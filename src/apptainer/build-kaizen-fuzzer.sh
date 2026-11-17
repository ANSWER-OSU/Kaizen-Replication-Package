#!/bin/bash

# --- Script Start ---
set -e

# Parse command line arguments
CLEAN_BUILD=false
if [[ "$1" == "--clean" ]]; then
    CLEAN_BUILD=true
fi

# Change to project root directory
ORIGINAL_DIR=$(pwd)
cd "$(dirname "$0")/.."

# Create the host build directory if it doesn't exist
mkdir -p "build/apptainer"

# Clean build directory if --clean flag is provided
if [[ "$CLEAN_BUILD" == true ]]; then
    echo ">>> Cleaning build directory..."
    rm -rf build/apptainer/*
fi

apptainer exec \
    --fakeroot \
    --bind .:/usr/src/Kaizen \
    build/kaizen.sif \
    /bin/bash -c "
    set -e
    echo '--- Inside Container ---'
    # The build directory inside the container is the same as the source
    cd /usr/src/Kaizen/
    
    echo '>>> Configuring with CMake...'
    # Tell CMake to use the 'build' subdirectory for its output
    CC=clang-18 CXX=clang++-18 cmake -S apptainer -B build/apptainer
    
    echo '>>> Compiling with make...'
    cmake --build build/apptainer -- -j\$(nproc)
"

# Print the relative path from the original directory to the build directory
RELATIVE_PATH=$(realpath --relative-to="$ORIGINAL_DIR" build/apptainer/kaizen)
echo ">>> Build complete. Kaizen is at $RELATIVE_PATH"
