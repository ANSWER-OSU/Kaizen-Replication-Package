#!/bin/bash

# Build script for the kernel extractor using apptainer
# This script builds the kernel extractor inside the apptainer container

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if apptainer image exists
APPTAINER_IMAGE="$PROJECT_ROOT/build/kaizen.sif"
if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Error: Apptainer image '$APPTAINER_IMAGE' does not exist"
    echo "Please build the apptainer image first"
    exit 1
fi

echo "Building kernel extractor using apptainer..."
echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"

# Create build directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/build"

# Run the build inside the apptainer container
apptainer exec --bind "$PROJECT_ROOT:/work" \
    "$APPTAINER_IMAGE" \
    bash -c "
        cd /work/resources/mutated-kernel-extraction && \
        mkdir -p build && \
        cd build && \
        echo 'Running cmake...' && \
        cmake .. && \
        echo 'Running make...' && \
        make -j\$(nproc)
    "

BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "Build completed successfully!"
    echo "Kernel extractor executable created at: $SCRIPT_DIR/build/kernel-extractor"
    
    # Test if the executable exists and is executable
    if [ -x "$SCRIPT_DIR/build/kernel-extractor" ]; then
        echo "Build verification: SUCCESS"
        echo ""
        echo "Usage: ./build/kernel-extractor <input-folder> <output-file>"
        echo "Or use the wrapper script: ./run-kernel-extractor.sh <input-folder> <source-file> <output-file>"
    else
        echo "Warning: Executable not found or not executable"
        exit 1
    fi
else
    echo "Build failed with exit code: $BUILD_EXIT_CODE"
    exit 1
fi
