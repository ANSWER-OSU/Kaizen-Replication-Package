#!/bin/bash

set -e

# Change to project root directory
ORIGINAL_DIR=$(pwd)
cd "$(dirname "$0")/.."

mkdir -p "build"
apptainer build --fakeroot build/kaizen.sif apptainer/kaizen-apptainer.def

# Print the relative path from the original directory to the build directory
RELATIVE_PATH=$(realpath --relative-to="$ORIGINAL_DIR" build/kaizen.sif)
echo "Build complete. SIF file is at $RELATIVE_PATH"
