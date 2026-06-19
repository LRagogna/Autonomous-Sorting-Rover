#!/usr/bin/env bash
set -euo pipefail

# Configure this checkout for Raspberry Pi runtime use.
#
# The full dataset is tracked in GitHub under data/ using Git LFS. Project
# documentation images live under docs/images/. The Pi should only pull rover
# code and lightweight project files during normal use.
#
# Sparse checkout keeps data/ and docs/images/ out of the Pi working tree.

git lfs install --local --skip-smudge
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!/data/' '!/docs/images/'

echo "Configured sparse checkout: this working tree will exclude data/ and docs/images/."
echo "Use 'git pull' normally after this; dataset files and doc images will stay off the Pi."
