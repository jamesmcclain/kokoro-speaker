#!/usr/bin/env bash
#
# run.sh — Build (if needed) and run the kokoro-speaker container so that:
#   1. It can play audio out of the HOST's speakers, by sharing the host's
#      PulseAudio/PipeWire socket into the container.
#   2. Its REST API (port 5001) is reachable from the host and other
#      machines on the same network, via a port mapping.
#
# Usage:
#   ./run.sh
#
# Then, from the host (or anywhere that can reach the host on port 5001):
#
# Play through the host's speakers (always asynchronous; fails with 409
# if the speaker is already in use):
#   curl -X POST http://localhost:5001/speak \
#     -H "Content-Type: application/json" \
#     -d '{"text": "Hello from inside a container.", "voice": "af_heart"}'
#
# Ask approximately how long until the speaker is free:
#   curl http://localhost:5001/speaker
#
# Download a WAV rendering (always synchronous; never plays through the
# speaker):
#   curl -X POST http://localhost:5001/speak \
#     -H "Content-Type: application/json" \
#     -d '{"text": "Hello as a file.", "voice": "af_heart", "play": false}' \
#     -o output.wav

set -euo pipefail

IMAGE_NAME="kokoro-speaker"
CONTAINER_NAME="kokoro-speaker"
HOST_PORT=5001
CONTAINER_PORT=5001

PULSE_SOCKET="/run/user/$(id -u)/pulse/native"
PULSE_COOKIE="${HOME}/.config/pulse/cookie"

if [ ! -S "${PULSE_SOCKET}" ]; then
    echo "Error: PulseAudio/PipeWire socket not found at ${PULSE_SOCKET}." >&2
    echo "Make sure PulseAudio or PipeWire (with pipewire-pulse) is running on the host." >&2
    exit 1
fi

# # Build the image if it doesn't already exist locally.
# if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
#     echo "Image ${IMAGE_NAME} not found locally — building it now..."
#     docker build -t "${IMAGE_NAME}" .
# fi

# # Remove any previous container with the same name so re-running is idempotent.
# docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

MOUNT_ARGS=(
    -v "${PULSE_SOCKET}:/tmp/pulse-socket"
)

# Only mount the cookie if it exists; some setups don't require it for
# local Unix-socket connections owned by the same UID.
if [ -f "${PULSE_COOKIE}" ]; then
    MOUNT_ARGS+=(-v "${PULSE_COOKIE}:/root/.config/pulse/cookie")
fi

# echo "Starting container '${CONTAINER_NAME}' in the foreground (Ctrl-C to stop)."
# echo "REST API will be available at: http://localhost:${HOST_PORT}"
# echo
# echo "Try it from another terminal:"
# echo "  curl -X POST http://localhost:${HOST_PORT}/speak \\"
# echo "    -H 'Content-Type: application/json' \\"
# echo "    -d '{\"text\": \"Hello world\", \"voice\": \"af_heart\"}'"
# echo

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    -e PULSE_SERVER=unix:/tmp/pulse-socket \
    "${MOUNT_ARGS[@]}" \
    --user "$(id -u):$(id -g)" \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    "${IMAGE_NAME}"
