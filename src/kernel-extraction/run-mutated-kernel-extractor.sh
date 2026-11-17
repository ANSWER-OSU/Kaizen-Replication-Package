#!/bin/bash

# Script to extract kernels from mutated files in AFL output directories
# Usage: ./run-mutated-kernel-extractor.sh <base-directory>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <base-directory>"
    echo "  base-directory: Path to directory containing AFL run folders (e.g., output/5-seed-run)"
    exit 1
fi

BASE_DIR="$1"

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory '$BASE_DIR' does not exist"
    exit 1
fi

# Get the absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR_SCRIPT="$SCRIPT_DIR/run-kernel-extractor.sh"

# Check if the kernel extractor script exists
if [ ! -f "$EXTRACTOR_SCRIPT" ]; then
    echo "Error: Kernel extractor script '$EXTRACTOR_SCRIPT' does not exist"
    exit 1
fi

echo "Processing base directory: $BASE_DIR"

# Counter for processed files
total_processed=0
total_folders=0
total_kernels=0

# CSV file for statistics
csv_file="$BASE_DIR/extracted-kernel-stats.csv"
echo "Example,Total_Kernels" > "$csv_file"

# Iterate through each folder in the base directory
for folder in "$BASE_DIR"/*; do
    if [ ! -d "$folder" ]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    echo "Processing folder: $folder_name"
    ((total_folders++))
    
    folder_kernel_count=0
    
    # Look for source-queue and working-copy directories
    source_queue_dir="$folder/source-queue"
    working_copy_dir="$folder/working-copy"
    extracted_kernels_dir="$folder/extracted-kernels"
    
    if [ ! -d "$source_queue_dir" ]; then
        echo "  Warning: source-queue directory not found in $folder_name, skipping"
        continue
    fi
    
    # Create working-copy directory if it doesn't exist
    if [ ! -d "$working_copy_dir" ]; then
        echo "  Creating working-copy directory"
        mkdir -p "$working_copy_dir"
    fi
    
    # Create extracted-kernels directory if it doesn't exist
    if [ ! -d "$extracted_kernels_dir" ]; then
        echo "  Creating extracted-kernels directory"
        mkdir -p "$extracted_kernels_dir"
    fi
    
    # Process each file in source-queue
    files_in_queue=$(find "$source_queue_dir" -maxdepth 1 -type f | wc -l)
    if [ $files_in_queue -eq 0 ]; then
        echo "  No files found in source-queue, skipping"
        continue
    fi
    
    echo "  Found $files_in_queue files in source-queue"
    
    for source_file in "$source_queue_dir"/*; do
        if [ ! -f "$source_file" ]; then
            continue
        fi
        
        source_filename=$(basename "$source_file")
        
        # Extract the ID from the filename (everything after the last dash before the extension)
        # e.g., "main-000.cu" -> ID is "000"
        if [[ $source_filename =~ ^(.+)-([0-9]+)\.(.+)$ ]]; then
            base_name="${BASH_REMATCH[1]}"
            file_id="${BASH_REMATCH[2]}"
            extension="${BASH_REMATCH[3]}"
            target_filename="${base_name}.${extension}"
        else
            echo "    Warning: Cannot parse ID from filename '$source_filename', skipping"
            continue
        fi
        
        target_file="$working_copy_dir/$target_filename"
        output_file="$extracted_kernels_dir/extracted-kernels-${file_id}.txt"
        
        echo "    Processing: $source_filename -> $target_filename (ID: $file_id)"
        
        # Copy the source file to working-copy with the new name
        cp "$source_file" "$target_file"
        if [ $? -ne 0 ]; then
            echo "      Error: Failed to copy $source_filename to $target_filename"
            continue
        fi
        
        # Run the kernel extractor on the working-copy directory
        echo "      Extracting kernels to: extracted-kernels-${file_id}.txt"
        "$EXTRACTOR_SCRIPT" "$working_copy_dir" "$output_file" > /dev/null 2>&1

        if [ $? -eq 0 ]; then
            # Count the number of kernels in the output file
            if [ -f "$output_file" ]; then
                kernel_count=$(grep -c "============<kernel" "$output_file" 2>/dev/null || echo "0")
                echo "      Success: $kernel_count kernels extracted"
                ((total_processed++))
                ((folder_kernel_count += kernel_count))
                ((total_kernels += kernel_count))
            else
                echo "      Success: Kernels extracted (file not found for counting)"
                ((total_processed++))
            fi
        else
            echo "      Error: Kernel extraction failed"
        fi
        
        # Clean up the copied file (optional - remove this line if you want to keep them)
        rm -f "$target_file"
    done
    
    # Add folder statistics to CSV
    echo "$folder_name,$folder_kernel_count" >> "$csv_file"
    echo "  Total kernels in $folder_name: $folder_kernel_count"
done

echo ""
echo "Processing complete!"
echo "Total folders processed: $total_folders"
echo "Total files processed: $total_processed"
echo "Total kernels extracted: $total_kernels"
echo "Output files are saved as 'extracted-kernels-<id>.txt' in 'extracted-kernels' subdirectories"
echo "Statistics saved to: $csv_file"
