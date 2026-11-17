#!/bin/bash

# Script to run the kernel extractor using apptainer
# Usage: ./run-extractor.sh <input-folder> <output-file>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input-folder> <output-file>"
    echo "  input-folder: Path to the program folder containing source code and Makefile"
    echo "  output-file:  Path where extracted kernels will be saved"
    exit 1
fi

# Get absolute paths
INPUT_FOLDER="$(realpath "$1")"
OUTPUT_FILE="$(realpath "$2")"

# Get the absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' does not exist"
    exit 1
fi

# Check if apptainer image exists
APPTAINER_IMAGE="$PROJECT_ROOT/build/kaizen.sif"
if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Error: Apptainer image '$APPTAINER_IMAGE' does not exist"
    exit 1
fi

# Check if kernel extractor executable exists
KERNEL_EXTRACTOR="$SCRIPT_DIR/build/kernel-extractor"
if [ ! -f "$KERNEL_EXTRACTOR" ]; then
    echo "Error: Kernel extractor executable '$KERNEL_EXTRACTOR' does not exist"
    echo "Please build the kernel extractor first using: ./build.sh"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUTPUT_DIR"

echo "Extracting kernels from: $INPUT_FOLDER"
echo "Output file: $OUTPUT_FILE"

# Convert paths to container paths
CONTAINER_INPUT="/work$(echo "$INPUT_FOLDER" | sed "s|$PROJECT_ROOT||")"
CONTAINER_OUTPUT="/work$(echo "$OUTPUT_FILE" | sed "s|$PROJECT_ROOT||")"

echo "Container input path: $CONTAINER_INPUT"
echo "Container output path: $CONTAINER_OUTPUT"

# Run the kernel extractor using apptainer
apptainer exec --bind "$PROJECT_ROOT:/work" \
    "$APPTAINER_IMAGE" \
    "/work/resources/mutated-kernel-extraction/build/kernel-extractor" \
    "$CONTAINER_INPUT" \
    "$CONTAINER_OUTPUT"

if [ $? -eq 0 ]; then
    echo "Kernel extraction completed successfully"
    echo "Results saved to: $OUTPUT_FILE"
    
    # Show a preview of the results if the file exists and is not empty
    if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
        echo ""
        echo "Preview of extracted kernels:"
        echo "=============================="
        head -20 "$OUTPUT_FILE"
        if [ $(wc -l < "$OUTPUT_FILE") -gt 20 ]; then
            echo "... (file continues)"
        fi
    fi
else
    echo "Error: Kernel extraction failed"
    exit 1
fi
