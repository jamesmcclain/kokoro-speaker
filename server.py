"""
server.py — Minimal REST API around Kokoro-82M that plays synthesized speech
out of the host's speakers (via a PulseAudio socket mounted into the
container).

The host has exactly one speaker, so all playback is serialized behind a
speaker mutex: at most one utterance can be playing at a time. A /speak
request that asks for playback while the speaker is busy is REJECTED
immediately (409 Conflict) rather than queued; the error response includes
an estimate of how many seconds remain until the speaker is free, and the
same estimate can be polled at any time via GET /speaker.

There is no async/sync switch: the execution mode is determined entirely
by what the request is for. Playback ("play": true, the default) is
asynchronous per force — the request is validated, the speaker is claimed,
and 202 Accepted is returned before synthesis even starts; rendering and
playback then happen in a background thread, and the speaker mutex is
released only once playback has actually finished draining. WAV download
("play": false) is synchronous per force — the request blocks until
synthesis completes and the response body is the WAV bytes (e.g. `curl ...
-o out.wav`); it never touches the speaker, so it neither takes the mutex
nor can it fail with 409.

Endpoints:
  GET  /health   -> {"status": "ok"}
  GET  /speaker  -> {
                      "busy": true|false,
                      "estimated_seconds_until_free": <number>
                    }
                  Approximately how long until the speaker is free.
                  0.0 with busy=false means the speaker is free right now.
                  The estimate is derived from the word-count/RTF heuristic
                  described below, so it can undershoot: if it reaches 0.0
                  while "busy" is still true, playback is expected to end
                  imminently but hasn't yet — trust "busy", not the number,
                  for the question "can I speak right now?".
  POST /speak     body: {
                     "text": "...",           (required)
                     "voice": "af_heart",     (optional, default af_heart)
                     "speed": 1.0,            (optional, default 1.0, range 0.1-3.0)
                     "play": true             (optional, default true)
                   }
                  play=true — speak through the host speaker,
                  asynchronously: if the speaker is free, it is claimed
                  and 202 Accepted is returned immediately, before
                  synthesis starts. If the speaker is busy, 409 Conflict
                  is returned with "estimated_seconds_until_free" and
                  nothing is synthesized.

                  The 202 response includes "estimated_duration_seconds":
                  an estimate of total time until the utterance has
                  finished playing. This is the sum of (1) an estimated
                  playback duration, based on word count and a typical
                  speaking rate, and (2) an estimated rendering duration —
                  how long Kokoro itself will take to synthesize the audio
                  — derived from a running average of actual measured
                  render times on this server. Neither component is an
                  exact measurement of this specific request (the audio
                  doesn't exist yet when the response is sent), so treat
                  the total as a ballpark, not a guarantee. This same
                  estimate is what backs GET /speaker while the utterance
                  is playing.

                  play=false — render the text and return the WAV bytes,
                  synchronously: the response is the audio itself
                  (Content-Type: audio/wav), available only once synthesis
                  has finished. No speaker involvement, no mutex, no 409.

                  The former "async" and "return_audio" fields have been
                  removed; a request that still sends either gets a 400
                  explaining the new contract, rather than a silent
                  reinterpretation of what the caller meant.
"""

import io
import subprocess
import threading
import time

import numpy as np
import soundfile as sf
from flask import Flask, request, send_file, jsonify
from kokoro import KPipeline

app = Flask(__name__)

# lang_code "a" = American English; change as needed. This loads once at
# startup so requests don't pay model-load cost each time.
pipeline = KPipeline(lang_code="a")

DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

# Rough average speaking rate for Kokoro's English voices at speed=1.0,
# used only to produce an *estimate* of audio duration before synthesis
# has actually happened. This is a heuristic, not a measurement: real
# duration depends on punctuation/pauses, the specific voice, and how the
# text tokenizes into phonemes. Treat the estimate as a ballpark, not a
# guarantee.
WORDS_PER_MINUTE = 165

# American English voices available under lang_code='a' (see
# https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md). Voices
# from other languages aren't included here since this server's pipeline
# is fixed to English g2p and wouldn't pronounce them correctly. A handful
# of these (see Dockerfile) are baked into the image at build time for
# zero-network-latency use; the rest still work, they just download their
# voice pack from Hugging Face on first use.
VALID_VOICES = frozenset({
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
})

# --- Rendering-time estimation -------------------------------------------
#
# estimated_duration_seconds (returned to the caller) is meant to be "how
# long from now until this utterance has finished playing" — which is
# rendering time (Kokoro actually synthesizing the audio) PLUS playback
# time (the audio playing at real speed), not playback time alone.
#
# We don't know rendering time in advance, so we track it as a running
# "real-time factor" (RTF): seconds of rendering per second of resulting
# audio. RTF_ESTIMATE starts at a seeded guess and is refined via an
# exponential moving average using the actual measured render time of
# every synthesis this process performs (played or not), so the estimate
# adapts to the host's real hardware/load over the server's lifetime.
_rtf_lock = threading.Lock()
_rtf_estimate = 0.3  # seed guess: render time ~= 30% of audio duration
_RTF_SMOOTHING = 0.3  # weight given to each new measurement


def _get_rtf_estimate():
    with _rtf_lock:
        return _rtf_estimate


def _update_rtf_estimate(render_seconds, audio_duration_seconds):
    global _rtf_estimate
    if audio_duration_seconds <= 0:
        return
    measured_rtf = render_seconds / audio_duration_seconds
    with _rtf_lock:
        _rtf_estimate = (
            (1 - _RTF_SMOOTHING) * _rtf_estimate + _RTF_SMOOTHING * measured_rtf
        )


def _estimate_total_duration(text, speed):
    """Estimated seconds from 'now' until an utterance of this text has
    finished playing: estimated playback time (word count at a typical
    speaking rate, scaled by speed) plus estimated rendering time (the
    rolling RTF measured on this server). A ballpark, not a guarantee."""
    word_count = len(text.split())
    estimated_playback = (word_count / WORDS_PER_MINUTE) * 60 / speed
    estimated_rendering = estimated_playback * _get_rtf_estimate()
    return round(estimated_playback + estimated_rendering, 1)


# --- Speaker mutex ---------------------------------------------------------
#
# The host has one speaker, and this API's contract is fail-fast rather
# than queue: at most one utterance may hold the speaker at a time, and a
# playback request made while the speaker is held is rejected immediately
# (409) instead of blocking behind the current utterance.
#
# _speaker_mutex is claimed *synchronously in the request handler* (a
# non-blocking acquire, so two racing requests can't both win) before the
# 202 is sent, and released by the background thread only after playback
# has fully finished (paplay has drained and exited), not merely after
# synthesis. It is deliberately NOT used to serialize synthesis itself:
# play=false renders never touch it.
#
# _speaker_busy_until is the estimate backing "how long until the speaker
# is free": the monotonic time at which the current utterance is
# *expected* to finish, set from the same word-count/RTF heuristic that
# produces estimated_duration_seconds. Because it's a heuristic, the
# remaining-time figure can reach 0 while playback is still draining;
# "busy" (i.e. whether the mutex is actually held) is the ground truth.
_speaker_mutex = threading.Lock()
_speaker_state_lock = threading.Lock()  # guards _speaker_busy_until
_speaker_busy_until = 0.0  # time.monotonic() at which the speaker should free up


def _try_claim_speaker(estimated_duration_seconds):
    """Attempts to claim the speaker without blocking. On success, records
    when it's expected to be free again and returns True; the caller (or a
    thread the caller spawns) is then responsible for _release_speaker().
    Returns False if some other utterance currently holds it."""
    global _speaker_busy_until
    if not _speaker_mutex.acquire(blocking=False):
        return False
    with _speaker_state_lock:
        _speaker_busy_until = time.monotonic() + estimated_duration_seconds
    return True


def _release_speaker():
    global _speaker_busy_until
    with _speaker_state_lock:
        _speaker_busy_until = 0.0
    _speaker_mutex.release()


def _speaker_status():
    """Returns (busy, estimated_seconds_until_free). The estimate is
    approximate (see _speaker_busy_until above) and clamped to >= 0; it
    can be 0.0 while busy is still True if the heuristic undershot."""
    busy = _speaker_mutex.locked()
    if not busy:
        return False, 0.0
    with _speaker_state_lock:
        remaining = _speaker_busy_until - time.monotonic()
    return True, max(round(remaining, 1), 0.0)


def _open_paplay_stream():
    """Starts a single long-lived `paplay` process reading raw float32
    PCM from stdin, so audio chunks can be written to it as Kokoro
    produces them instead of waiting for the whole utterance to render
    first. Returns the Popen handle, or None if paplay can't be started
    (e.g. missing binary), in which case the caller should skip playback
    entirely rather than fail synthesis."""
    try:
        return subprocess.Popen(
            [
                "paplay",
                "--raw",
                "--format=float32le",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
            ],
            stdin=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        app.logger.warning("paplay not available: %s", e)
        return None


def _run_pipeline(text, voice, speed, play):
    """Synthesizes with Kokoro chunk-by-chunk. If play is True, each
    chunk is streamed to a live `paplay` process as soon as it's
    produced, so playback of chunk N starts while chunk N+1 is still
    being rendered — this is what actually reduces time-to-first-phoneme,
    as opposed to concatenating everything and playing it only once the
    entire utterance has been synthesized. When play is True, this
    function does not return until paplay has drained and exited, i.e.
    until the speaker is genuinely done — callers rely on that to know
    when it's safe to release the speaker mutex.

    Callers that pass play=True must already hold the speaker mutex;
    this function itself neither claims nor releases it.

    Measures total wall-clock render time and feeds it into the rolling
    RTF estimate used for future duration estimates. Returns the
    concatenated audio array (still needed for return_audio and for the
    duration measurement). Raises RuntimeError if synthesis produced no
    audio."""
    paplay_proc = _open_paplay_stream() if play else None

    start = time.time()
    audio_chunks = []
    try:
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            audio_chunks.append(audio)
            if paplay_proc is not None:
                try:
                    paplay_proc.stdin.write(
                        np.asarray(audio, dtype=np.float32).tobytes()
                    )
                except (BrokenPipeError, OSError) as e:
                    app.logger.warning("paplay write failed: %s", e)
                    paplay_proc = None
    finally:
        if paplay_proc is not None:
            try:
                paplay_proc.stdin.close()
                paplay_proc.wait()
            except (BrokenPipeError, OSError) as e:
                app.logger.warning("paplay close/wait failed: %s", e)
    render_seconds = time.time() - start

    if not audio_chunks:
        raise RuntimeError("synthesis produced no audio")

    full_audio = np.concatenate(audio_chunks)
    audio_duration_seconds = len(full_audio) / SAMPLE_RATE
    _update_rtf_estimate(render_seconds, audio_duration_seconds)
    return full_audio


def _synthesize_and_play(text, voice, speed):
    """Runs in a background thread with the speaker mutex already held by
    the request handler that spawned it: synthesizes with Kokoro,
    streaming each chunk to paplay as it's produced. Releases the speaker
    mutex once playback has fully finished (or failed) — this release
    must happen on every path, otherwise the speaker would be stuck busy
    forever. Any failure here is only logged, since the HTTP response has
    already been sent."""
    try:
        _run_pipeline(text, voice, speed, play=True)
    except Exception as e:
        app.logger.error("synthesis failed for voice=%s: %s", voice, e)
    finally:
        _release_speaker()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/speaker", methods=["GET"])
def speaker():
    """Approximately how long until the speaker is free. busy=false means
    it's free right now; busy=true with a 0.0 estimate means the current
    utterance overran its estimate and should finish imminently."""
    busy, seconds = _speaker_status()
    return jsonify({
        "busy": busy,
        "estimated_seconds_until_free": seconds,
    })


@app.route("/speak", methods=["POST"])
def speak():
    payload = request.get_json(force=True, silent=True) or {}
    text = payload.get("text")
    voice = payload.get("voice", DEFAULT_VOICE)
    speed = payload.get("speed", 1.0)
    # The single mode switch: play=true (default) means asynchronous
    # playback through the host speaker; play=false means synchronous
    # WAV download. There is no independent async knob.
    play = payload.get("play", True)

    # Validation that's cheap and immediate, done synchronously in both
    # modes so the caller always gets parameter errors right away.
    if "async" in payload:
        return jsonify({
            "error": "the 'async' field has been removed: playback "
                     "(play=true) is always asynchronous and WAV download "
                     "(play=false) is always synchronous. Drop the field "
                     "and choose the mode via 'play'.",
        }), 400
    if "return_audio" in payload:
        return jsonify({
            "error": "the 'return_audio' field has been removed: "
                     "play=false always returns the WAV bytes, and "
                     "play=true never can (the audio doesn't exist yet "
                     "when its 202 response is sent). Drop the field and "
                     "choose the mode via 'play'.",
        }), 400
    if not text:
        return jsonify({"error": "missing 'text' field"}), 400
    if voice not in VALID_VOICES:
        return jsonify({
            "error": f"unknown voice '{voice}'",
            "valid_voices": sorted(VALID_VOICES),
        }), 400
    if not isinstance(speed, (int, float)) or not (0.1 <= speed <= 3.0):
        return jsonify({"error": "'speed' must be a number between 0.1 and 3.0"}), 400
    if not isinstance(play, bool):
        return jsonify({"error": "'play' must be a boolean"}), 400

    if play:
        # Asynchronous playback. The speaker must be claimed *now*, in
        # the request handler, so that a busy speaker turns into an
        # immediate failure instead of overlapping audio or an invisible
        # queue. The non-blocking acquire is the arbiter when two
        # requests race.
        estimated_duration = _estimate_total_duration(text, speed)
        if not _try_claim_speaker(estimated_duration):
            _, seconds_until_free = _speaker_status()
            return jsonify({
                "error": "speaker is busy",
                "estimated_seconds_until_free": seconds_until_free,
            }), 409

        # From here on the speaker is ours; _synthesize_and_play releases
        # it when playback finishes. If the thread somehow fails to start,
        # release immediately so the speaker can't leak into a stuck-busy
        # state.
        try:
            threading.Thread(
                target=_synthesize_and_play,
                args=(text, voice, speed),
                daemon=True,
            ).start()
        except Exception:
            _release_speaker()
            raise
        return jsonify({
            "status": "accepted",
            "voice": voice,
            "played": True,
            "estimated_duration_seconds": estimated_duration,
        }), 202

    # Synchronous WAV download. Blocks until synthesis completes, then
    # returns the audio bytes. Never touches the speaker, so it neither
    # takes the mutex nor competes with playback.
    try:
        full_audio = _run_pipeline(text, voice, speed, play=False)
    except RuntimeError:
        return jsonify({"error": "synthesis produced no audio"}), 500

    buf = io.BytesIO()
    sf.write(buf, full_audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", download_name="speech.wav")


def _warm_up():
    """Runs one throwaway synthesis (no playback, so no speaker mutex)
    before the server starts accepting requests. The Dockerfile already
    bakes model weights and voice packs into the image at build time, but
    a fresh container still pays a first-inference tax when the process
    actually starts (e.g. lazy kernel/thread-pool init). Paying that cost
    here means the first real /speak request doesn't have to."""
    try:
        start = time.time()
        _run_pipeline("Warming up.", DEFAULT_VOICE, 1.0, play=False)
        app.logger.info("warm-up synthesis completed in %.2fs", time.time() - start)
    except Exception as e:
        app.logger.warning("warm-up synthesis failed (non-fatal): %s", e)


if __name__ == "__main__":
    _warm_up()
    # Threaded so GET /speaker can be answered while a playback thread is
    # running (Flask's dev server is threaded by default, but be explicit:
    # the speaker-busy contract depends on concurrent request handling).
    app.run(host="0.0.0.0", port=5001, threaded=True)
