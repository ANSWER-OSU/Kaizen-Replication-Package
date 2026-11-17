#!/usr/bin/env python3
"""
Setup benchmark for kernel fuzzing by:
1. Reading HECBENCH_SRC_DIR from .env file
2. Converting example name to folder path
3. Ensuring main program is main.cu
4. Inlining CUDA kernel headers into main file
NOTE: This is a partial automation script that works for many examples,
manual work to ensure compatibility will need to be done for many hecbench applications.
"""

import os
import sys
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_env_file(env_file_path: str = ".env") -> Dict[str, str]:
    """
    Read .env file and return a dictionary of environment variables.
    """
    env_vars = {}
    try:
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    # Handle $(pwd) substitution
                    if '$(pwd)' in value:
                        value = value.replace('$(pwd)', os.getcwd())
                    env_vars[key] = value
    except FileNotFoundError:
        print(f"Error: .env file not found at {env_file_path}")
        sys.exit(1)
    return env_vars


def get_example_folder_path(hecbench_src_dir: str, example_name: str) -> Path:
    """
    Convert example name to folder path: HECBENCH_SRC_DIR/<example name>-cuda
    """
    folder_name = f"{example_name}-cuda"
    folder_path = Path(hecbench_src_dir) / folder_name
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        sys.exit(1)
    
    return folder_path


def create_backup(folder_path: Path, example_name: str) -> Path:
    """
    Create a backup of the folder as <example name>-cuda-bak
    """
    backup_name = f"{example_name}-cuda-bak"
    backup_path = folder_path.parent / backup_name
    
    # Remove existing backup if it exists
    if backup_path.exists():
        shutil.rmtree(backup_path)
    
    # Create backup
    shutil.copytree(folder_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    return backup_path


def analyze_makefile(folder_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Analyze Makefile to check if source is main.cu.
    Returns (needs_rename, current_source_file)
    """
    makefile_path = folder_path / "Makefile"
    
    if not makefile_path.exists():
        print(f"Warning: No Makefile found in {folder_path}")
        return False, None
    
    try:
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Look for source = <filename> pattern
        source_match = re.search(r'source\s*=\s*(\S+)', content, re.IGNORECASE)
        
        if source_match:
            current_source = source_match.group(1)
            if current_source != "main.cu":
                return True, current_source
        else:
            # If no explicit source found, check for common patterns
            # Look for files that might be the main source
            cu_files = list(folder_path.glob("*.cu"))
            if cu_files and not (folder_path / "main.cu").exists():
                # Assume the first .cu file is the main source
                return True, cu_files[0].name
        
        return False, None
        
    except Exception as e:
        print(f"Error reading Makefile: {e}")
        return False, None


def rename_main_source(folder_path: Path, current_source: str) -> bool:
    """
    Rename the main source file to main.cu and update Makefile.
    Returns True if changes were made.
    """
    current_file = folder_path / current_source
    main_file = folder_path / "main.cu"
    makefile_path = folder_path / "Makefile"
    
    if not current_file.exists():
        print(f"Warning: Source file {current_file} does not exist")
        return False
    
    try:
        # Rename the file
        shutil.move(str(current_file), str(main_file))
        print(f"Renamed {current_source} to main.cu")
        
        # Update Makefile
        if makefile_path.exists():
            with open(makefile_path, 'r') as f:
                content = f.read()
            
            # Replace source declaration
            updated_content = re.sub(
                r'source\s*=\s*\S+',
                'source = main.cu',
                content,
                flags=re.IGNORECASE
            )
            
            with open(makefile_path, 'w') as f:
                f.write(updated_content)
            
            print("Updated Makefile to use main.cu")
        
        return True
        
    except Exception as e:
        print(f"Error renaming source file: {e}")
        return False


def find_cuda_kernel_files(folder_path: Path) -> List[Path]:
    """
    Find files containing CUDA kernels (identified by __global__ keyword).
    Excludes main.cu.
    """
    kernel_files = []
    
    # Check all .h, .hpp, .cuh, .cu files except main.cu
    for pattern in ["*.h", "*.hpp", "*.cuh", "*.cu"]:
        for file_path in folder_path.glob(pattern):
            if file_path.name == "main.cu":
                continue
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for __global__ keyword (possibly with whitespace)
                if re.search(r'__global__', content):
                    kernel_files.append(file_path)
                    print(f"Found CUDA kernels in: {file_path.name}")
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return kernel_files


def inline_header_files(folder_path: Path, kernel_files: List[Path]) -> bool:
    """
    Inline header files containing CUDA kernels into main.cu.
    Returns True if changes were made.
    """
    main_file = folder_path / "main.cu"
    
    if not main_file.exists():
        print("Error: main.cu does not exist")
        return False
    
    if not kernel_files:
        return False
    
    try:
        with open(main_file, 'r') as f:
            main_content = f.read()
        
        changes_made = False
        
        for kernel_file in kernel_files:
            include_pattern = f'#include\\s+["\']?{kernel_file.name}["\']?'
            
            if re.search(include_pattern, main_content):
                # Read the kernel file content
                with open(kernel_file, 'r') as f:
                    kernel_content = f.read()
                
                # Comment out the include and add the content
                replacement = f"""// #include "{kernel_file.name}"
/// inclusion of {kernel_file.name}
{kernel_content}
/// end of inclusion of {kernel_file.name}"""
                
                main_content = re.sub(include_pattern, replacement, main_content)
                changes_made = True
                print(f"Inlined {kernel_file.name} into main.cu")
        
        if changes_made:
            with open(main_file, 'w') as f:
                f.write(main_content)
        
        return changes_made
        
    except Exception as e:
        print(f"Error inlining header files: {e}")
        return False


def main():
    """
    Main function to setup benchmark for kernel fuzzing.
    """
    if len(sys.argv) != 2:
        print("Usage: python3 setup-benchmark-for-kernel-fuzzing.py <example_name>")
        sys.exit(1)
    
    example_name = sys.argv[1]
    
    # Step 1: Read .env file to get HECBENCH_SRC_DIR
    env_vars = read_env_file()
    
    if 'HECBENCH_SRC_DIR' not in env_vars:
        print("Error: HECBENCH_SRC_DIR not found in .env file")
        sys.exit(1)
    
    hecbench_src_dir = env_vars['HECBENCH_SRC_DIR']
    print(f"Using HECBENCH_SRC_DIR: {hecbench_src_dir}")
    
    # Step 2: Get folder path
    folder_path = get_example_folder_path(hecbench_src_dir, example_name)
    print(f"Working with folder: {folder_path}")
    
    # Step 3: Analyze what changes are needed
    needs_rename, current_source = analyze_makefile(folder_path)
    kernel_files = find_cuda_kernel_files(folder_path)
    
    changes_needed = needs_rename or len(kernel_files) > 0
    
    if not changes_needed:
        print("No changes needed.")
        return
    
    # Step 4: Create backup before making changes
    create_backup(folder_path, example_name)
    
    # Step 5: Rename main source file if needed
    if needs_rename and current_source:
        rename_main_source(folder_path, current_source)
    
    # Step 6: Inline kernel header files
    if kernel_files:
        inline_header_files(folder_path, kernel_files)
    
    print(f"Setup complete for {example_name}")


if __name__ == "__main__":
    main()