"""
server.py — Minimal REST API around Kokoro-82M that plays synthesized speech
out of the host's speakers (via a PulseAudio socket mounted into the
container).

By default, /speak is asynchronous: it validates the request (missing text,
unknown voice, etc.) synchronously and returns immediately with 202 Accepted
once there's no reason to believe synthesis/playback will fail, then
performs the actual synthesis and playback in a background thread. Pass
"async": false in the request body to fall back to the old synchronous
behavior, which waits for synthesis to finish and can optionally return the
WAV bytes.

Endpoints:
  GET  /health   -> {"status": "ok"}
  POST /speak     body: {
                     "text": "...",           (required)
                     "voice": "af_heart",     (optional, default af_heart)
                     "speed": 1.0,            (optional, default 1.0, range 0.1-3.0)
                     "play": true,            (optional, default true)
                     "async": true,           (optional, default true)
                     "return_audio": true     (optional, default true;
                                                only honored when async=false,
                                                since the audio doesn't exist
                                                yet when the async response
                                                is sent)
                   }
                  In async mode (default), the response includes
                  "estimated_duration_seconds": an estimate of total time
                  until the utterance has finished playing. This is the
                  sum of (1) an estimated playback duration, based on word
                  count and a typical speaking rate, and (2) an estimated
                  rendering duration — how long Kokoro itself will take to
                  synthesize the audio — derived from a running average of
                  actual measured render times on this server. Neither
                  component is an exact measurement of this specific
                  request (the audio doesn't exist yet when the response
                  is sent), so treat the total as a ballpark, not a
                  guarantee. The response format is unchanged: still a
                  single number in "estimated_duration_seconds".
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
# every synthesis this process performs (sync or async), so the estimate
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
    entire utterance has been synthesized.

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


def _synthesize_and_play(text, voice, play, speed):
    """Runs in a background thread: synthesize with Kokoro, streaming
    each chunk to paplay as it's produced if play is requested. Any
    failure here is only logged, since the HTTP response has already
    been sent."""
    try:
        _run_pipeline(text, voice, speed, play)
    except Exception as e:
        app.logger.error("synthesis failed for voice=%s: %s", voice, e)
        return


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/speak", methods=["POST"])
def speak():
    payload = request.get_json(force=True, silent=True) or {}
    text = payload.get("text")
    voice = payload.get("voice", DEFAULT_VOICE)
    play = payload.get("play", True)
    is_async = payload.get("async", True)
    return_audio = payload.get("return_audio", True)
    speed = payload.get("speed", 1.0)

    # Validation that's cheap and immediate — this is the "immediate error"
    # part of the contract, done synchronously regardless of async/sync mode.
    if not text:
        return jsonify({"error": "missing 'text' field"}), 400
    if voice not in VALID_VOICES:
        return jsonify({
            "error": f"unknown voice '{voice}'",
            "valid_voices": sorted(VALID_VOICES),
        }), 400
    if not isinstance(speed, (int, float)) or not (0.1 <= speed <= 3.0):
        return jsonify({"error": "'speed' must be a number between 0.1 and 3.0"}), 400

    if is_async:
        # No reason left to believe this will fail — hand off to a
        # background thread and return immediately, before synthesis
        # (which is the slow part) has even started.
        if payload.get("return_audio") is not None:
            app.logger.warning(
                "return_audio was requested but is ignored in async mode "
                "(audio doesn't exist yet when this response is sent)"
            )
        word_count = len(text.split())
        estimated_playback = (word_count / WORDS_PER_MINUTE) * 60 / speed
        estimated_rendering = estimated_playback * _get_rtf_estimate()
        estimated_duration = round(estimated_playback + estimated_rendering, 1)
        threading.Thread(
            target=_synthesize_and_play,
            args=(text, voice, play, speed),
            daemon=True,
        ).start()
        return jsonify({
            "status": "accepted",
            "voice": voice,
            "played": play,
            "estimated_duration_seconds": estimated_duration,
        }), 202

    # Synchronous legacy path: wait for synthesis, optionally return bytes.
    # Playback (if requested) is streamed chunk-by-chunk inside
    # _run_pipeline rather than written to a temp file and played only
    # after the whole utterance has rendered.
    try:
        full_audio = _run_pipeline(text, voice, speed, play)
    except RuntimeError:
        return jsonify({"error": "synthesis produced no audio"}), 500

    if not return_audio:
        return jsonify({"status": "ok", "played": play}), 200

    buf = io.BytesIO()
    sf.write(buf, full_audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", download_name="speech.wav")


def _warm_up():
    """Runs one throwaway synthesis (no playback) before the server
    starts accepting requests. The Dockerfile already bakes model
    weights and voice packs into the image at build time, but a fresh
    container still pays a first-inference tax when the process actually
    starts (e.g. lazy kernel/thread-pool init). Paying that cost here
    means the first real /speak request doesn't have to."""
    try:
        start = time.time()
        _run_pipeline("Warming up.", DEFAULT_VOICE, 1.0, play=False)
        app.logger.info("warm-up synthesis completed in %.2fs", time.time() - start)
    except Exception as e:
        app.logger.warning("warm-up synthesis failed (non-fatal): %s", e)


if __name__ == "__main__":
    _warm_up()
    app.run(host="0.0.0.0", port=5001)
