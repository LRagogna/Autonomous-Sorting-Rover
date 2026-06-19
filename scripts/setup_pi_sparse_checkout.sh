#!/usr/bin/env bash
set -euo pipefail

# Configure this checkout for Raspberry Pi runtime use.
#
# The full dataset is tracked in GitHub under data/ using Git LFS, but the Pi
# should only pull rover code and lightweight project files during normal use.
# Sparse checkout keeps data/ out of the Pi working tree entirely.

git lfs install --local --skip-smudge
git sparse-checkout init --no-cone
git sparse-checkout set "/*" "!/data/"

echo "Configured sparse checkout: this working tree will exclude data/."
echo "Use 'git pull' normally after this; dataset files will stay off the Pi."
