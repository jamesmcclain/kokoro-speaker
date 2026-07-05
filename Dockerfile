# Dockerfile — Kokoro-82M TTS REST API that plays audio on the host via PulseAudio
#
# Build:
#   docker build -t kokoro-speaker .
#
# Run: use the accompanying run.sh script, which sets up the PulseAudio
# socket mapping and port forwarding described below.

FROM python:3.11-slim

# pulseaudio-utils gives us `paplay` for playback through the host's PulseAudio
# server; espeak-ng is used by Kokoro as a fallback phonemizer for
# out-of-dictionary words.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pulseaudio-utils \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch (CPU build) from PyTorch's own index first. Note: that
# index doesn't host flask/kokoro/soundfile, so it must be a separate
# pip install call, not combined with the packages below.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Kokoro's Python package + Flask for the API, from regular PyPI.
# NOTE: "kokoro>=0.7.9" must be quoted — unquoted, the shell interprets the
# unescaped '>' as output redirection instead of a version specifier.
RUN pip install --no-cache-dir \
    flask \
    "kokoro>=0.7.9" \
    soundfile

# Kokoro's English g2p (misaki) lazily downloads the spaCy "en_core_web_sm"
# model on first use if it's missing, via `pip install --user ...`. That
# breaks when the server later runs as an arbitrary non-root UID (see
# run.sh): the already-running Python process never sees the newly
# installed --user site-packages directory, so it "succeeds" and then
# immediately fails with "Can't find model 'en_core_web_sm'". Installing
# it here, at build time, as root, into the normal (writable) site-packages
# avoids that runtime download path entirely.
RUN python3 -m spacy download en_core_web_sm

# run.sh runs this container with --user "$(id -u):$(id -g)" so files
# written from inside the container end up owned by the host user. That
# UID has no corresponding entry in the container's /etc/passwd, which
# breaks anything that calls getpass.getuser()/pwd.getpwuid() — notably
# torch's cache-dir resolution. Point HOME and the relevant cache dirs at
# writable, world-writable locations so nothing needs to resolve a
# username at runtime.
ENV HOME=/tmp \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    HF_HOME=/tmp/huggingface

RUN mkdir -p /tmp/torchinductor /tmp/huggingface && chmod -R 777 /tmp

# Bake in the Kokoro model weights AND a set of commonly used voice packs
# at build time, by running one real synthesis per voice here, which
# populates the Hugging Face cache under HF_HOME. At container run time
# (potentially as a different arbitrary UID) these voices are already
# cached, so requests for them need no network access on `docker run`.
#
# Cache dirs are left world-writable (not locked down to read-only)
# afterward: hf_hub_download needs to create new subdirectories/lock files
# whenever a voice other than the baked-in ones below is requested at
# runtime, and the runtime UID won't match the build-time (root) owner.
RUN python3 -c "\
from kokoro import KPipeline; \
pipeline = KPipeline(lang_code='a'); \
voices = ['af_heart', 'am_santa', 'am_michael', 'am_onyx', 'af_river', 'af_alloy', 'af_nicole', 'am_adam', 'am_echo']; \
[list(pipeline('Hello world', voice=v)) for v in voices]" \
    && chmod -R 777 /tmp/huggingface /tmp/torchinductor

COPY server.py /app/server.py

ENV PULSE_SERVER=unix:/tmp/pulse-socket

EXPOSE 5001

CMD ["python3", "/app/server.py"]

# docker build -t kokoro-speaker -f Dockerfile .
