#!/usr/bin/env bash
#
# Pull Raspberry Pi camera recordings onto the training computer over the LAN.
#
# Videos are too large for GitHub's free Git LFS budget, so they do NOT go through
# GitHub. Record on the Pi with src/record_video.py (saves into pi_videos/), then
# run this on your Mac to copy them straight off the Pi with rsync.
#
# The Pi host and repo path can be overridden with environment variables:
#
#   PI_HOST=pi@192.168.254.42 PI_REPO=~/Autonomous-Sorting-Rover ./scripts/pull_pi_videos.sh
#
# rsync only transfers new/changed files, so re-running it is cheap.

set -euo pipefail

PI_HOST="${PI_HOST:-pi@raspberrypi.local}"
PI_REPO="${PI_REPO:-~/Autonomous-Sorting-Rover}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/pi_videos"
mkdir -p "${DEST}"

echo "Pulling recordings:"
echo "  from  ${PI_HOST}:${PI_REPO}/pi_videos/"
echo "  into  ${DEST}/"
echo

rsync -av --progress "${PI_HOST}:${PI_REPO%/}/pi_videos/" "${DEST}/"

echo
echo "Done. Sort the clips into data/raw_videos/<class>/ (or background/) and run Process Dataset."
