#!/usr/bin/env python3
"""
Setup script for benchmark differential fuzzing.
Processes all folders in example-bench to prepare them for fuzzing with AFL++.
NOTE: This is a partial automation script that works for many examples,
manual work to ensure compatibility will need to be done for many hecbench applications.
"""

import os
import shutil
import re
import argparse
from pathlib import Path

def create_ready_bench_dir(base_dir):
    """Create the ready-bench directory."""
    ready_bench_dir = os.path.join(base_dir, "ready-bench")
    if os.path.exists(ready_bench_dir):
        shutil.rmtree(ready_bench_dir)
    os.makedirs(ready_bench_dir)
    return ready_bench_dir

def is_cuda_folder(folder_name):
    """Check if folder is a CUDA variant."""
    return folder_name.endswith("-cuda")

def is_omp_folder(folder_name):
    """Check if folder is an OpenMP variant."""
    return folder_name.endswith("-omp")

def modify_makefile_cuda(makefile_path):
    """Modify makefile for CUDA variant."""
    with open(makefile_path, 'r') as f:
        content = f.read()
    
    # Change CC to CXX and add afl-clang-lto++
    content = re.sub(r'^CC\s*=.*$', 'CXX       = afl-clang-lto++', content, flags=re.MULTILINE)
    
    # Change ARCH to sm_86
    content = re.sub(r'^ARCH\s*=.*$', 'ARCH      = sm_86', content, flags=re.MULTILINE)
    
    # Change CFLAGS to CXXFLAGS first
    content = re.sub(r'CFLAGS', 'CXXFLAGS', content)
    
    # Now update the base CXXFLAGS line (after conversion from CFLAGS)
    content = re.sub(r'^CXXFLAGS := \$\(EXTRA_CXXFLAGS\).*$', 
                     'CXXFLAGS := $(EXTRA_CXXFLAGS) -stdlib=libc++ -std=c++14 --cuda-gpu-arch=$(ARCH)', 
                     content, flags=re.MULTILINE)
    
    # Change $(CC) to $(CXX) in build rules
    content = re.sub(r'\$\(CC\)', '$(CXX)', content)
    
    # Add libcuda.a target before clean target
    if 'clean:' in content and 'libcuda.a:' not in content:
        libcuda_target = """
# standalone add LIBCUDA definition
libcuda.a: CXXFLAGS += -Wall -DLIBCUDA

libcuda.a: $(obj)
	llvm-ar-18 rcs $@ $^

"""
        # Insert before clean target
        content = content.replace('clean:', libcuda_target + 'clean:')
        # Update clean target to include libcuda.a
        content = re.sub(r'clean:\s*\n\s*rm -rf \$\(program\) \$\(obj\)', 
                        'clean:\n\trm -rf $(program) $(obj) libcuda.a', content)
    
    # Add LDFLAGS for CUDA if it's empty
    content = re.sub(r'^LDFLAGS = $', 'LDFLAGS = -L/usr/local/cuda/lib64 -lcudart', content, flags=re.MULTILINE)
    
    with open(makefile_path, 'w') as f:
        f.write(content)

def modify_makefile_omp(makefile_path):
    """Modify makefile for OpenMP variant."""
    with open(makefile_path, 'r') as f:
        content = f.read()
    
    # Change CC to CXX and add afl-clang-lto++
    content = re.sub(r'^CC\s*=.*$', 'CXX       = afl-clang-lto++', content, flags=re.MULTILINE)
    
    # Change CFLAGS to CXXFLAGS first
    content = re.sub(r'CFLAGS', 'CXXFLAGS', content)
    
    # Now update the base CXXFLAGS line (after conversion from CFLAGS)
    content = re.sub(r'^CXXFLAGS := \$\(EXTRA_CXXFLAGS\).*$', 
                     'CXXFLAGS := $(EXTRA_CXXFLAGS) -stdlib=libc++ -std=c++14', 
                     content, flags=re.MULTILINE)
    
    # Change $(CC) to $(CXX) in build rules
    content = re.sub(r'\$\(CC\)', '$(CXX)', content)
    
    # Add libomp.a target before clean target
    if 'clean:' in content and 'libomp.a:' not in content:
        libomp_target = """
# standalone add LIBOMP definition
libomp.a: CXXFLAGS += -DLIBOMP

libomp.a: $(obj)
	llvm-ar-18 rcs $@ $^

"""
        # Insert before clean target
        content = content.replace('clean:', libomp_target + 'clean:')
        # Update clean target to include libomp.a
        content = re.sub(r'clean:\s*\n\s*rm -rf \$\(program\) \$\(obj\)', 
                        'clean:\n\trm -rf $(program) $(obj) libomp.a', content)
    
    # Update OpenMP flags to use fopenmp=libomp
    content = re.sub(r'-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__', 
                     '-fopenmp=libomp', content)
    content = re.sub(r'-qopenmp', '-fopenmp=libomp', content)
    
    with open(makefile_path, 'w') as f:
        f.write(content)

def modify_cuda_source(source_path):
    """Modify CUDA source file to add namespace and conditional compilation."""
    with open(source_path, 'r') as f:
        content = f.read()
    
    # Add namespace at the beginning (after includes)
    # Find the first non-include, non-define line
    lines = content.split('\n')
    insert_idx = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped and 
            not stripped.startswith('#include') and 
            not stripped.startswith('#define') and
            not stripped.startswith('//') and
            not stripped.startswith('/*') and
            not stripped.startswith('*') and
            not stripped.startswith('typedef')):
            insert_idx = i
            break
    
    # Insert namespace
    lines.insert(insert_idx, '')
    lines.insert(insert_idx + 1, '#ifdef LIBCUDA')
    lines.insert(insert_idx + 2, 'namespace cuda_program {')
    lines.insert(insert_idx + 3, '#endif // LIBCUDA')
    lines.insert(insert_idx + 4, '')
    
    # Change main function
    content = '\n'.join(lines)
    content = re.sub(r'int main\s*\(', 
                     '#ifdef LIBCUDA\nint cuda_main\n#else\nint main\n#endif // LIBCUDA\n(', 
                     content)
    
    # Add closing namespace at the end
    content += '\n\n#ifdef LIBCUDA\n}\n#endif // LIBCUDA\n'
    
    with open(source_path, 'w') as f:
        f.write(content)

def modify_omp_source(source_path):
    """Modify OpenMP source file to add namespace and conditional compilation."""
    with open(source_path, 'r') as f:
        content = f.read()
    
    # Add namespace at the beginning (after includes)
    lines = content.split('\n')
    insert_idx = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped and 
            not stripped.startswith('#include') and 
            not stripped.startswith('#define') and
            not stripped.startswith('//') and
            not stripped.startswith('/*') and
            not stripped.startswith('*') and
            not stripped.startswith('typedef')):
            insert_idx = i
            break
    
    # Insert namespace
    lines.insert(insert_idx, '')
    lines.insert(insert_idx + 1, '#ifdef LIBOMP')
    lines.insert(insert_idx + 2, 'namespace omp_program {')
    lines.insert(insert_idx + 3, '#endif // LIBOMP')
    lines.insert(insert_idx + 4, '')
    
    # Change main function
    content = '\n'.join(lines)
    content = re.sub(r'int main\s*\(', 
                     '#ifdef LIBOMP\nint omp_main\n#else\nint main\n#endif // LIBOMP\n(', 
                     content)
    
    # Add closing namespace at the end
    content += '\n\n#ifdef LIBOMP\n}\n#endif // LIBOMP\n'
    
    with open(source_path, 'w') as f:
        f.write(content)

def get_source_file_from_makefile(makefile_path):
    """Extract the source file name from the Makefile."""
    try:
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Look for "source = filename" pattern
        match = re.search(r'^source\s*=\s*(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for common source file extensions
        lines = content.split('\n')
        for line in lines:
            if '.cu' in line or '.cpp' in line or '.c' in line:
                # Try to extract filename from various patterns
                for ext in ['.cu', '.cpp', '.c']:
                    if ext in line:
                        # Look for word boundaries around filenames
                        pattern = r'\b\w+' + re.escape(ext) + r'\b'
                        match = re.search(pattern, line)
                        if match:
                            return match.group()
        
        return None
    except Exception as e:
        print(f"Error reading makefile {makefile_path}: {e}")
        return None

def process_folder(src_folder, dest_folder):
    """Process a single benchmark folder."""
    folder_name = os.path.basename(src_folder)
    
    # Copy entire folder
    shutil.copytree(src_folder, dest_folder)
    
    # Find and modify Makefile
    makefile_path = os.path.join(dest_folder, 'Makefile')
    if os.path.exists(makefile_path):
        if is_cuda_folder(folder_name):
            modify_makefile_cuda(makefile_path)
        elif is_omp_folder(folder_name):
            modify_makefile_omp(makefile_path)
    
    # Find and modify source files using Makefile information
    if os.path.exists(makefile_path):
        source_file = get_source_file_from_makefile(makefile_path)
        
        if source_file:
            source_path = os.path.join(dest_folder, source_file)
            
            if os.path.exists(source_path):
                if is_cuda_folder(folder_name) and source_file.endswith('.cu'):
                    modify_cuda_source(source_path)
                elif is_omp_folder(folder_name) and (source_file.endswith('.cpp') or source_file.endswith('.c')):
                    modify_omp_source(source_path)
                else:
                    print(f"  Warning: Source file {source_file} doesn't match expected extension for {folder_name}")
            else:
                print(f"  Warning: Source file {source_file} not found in {folder_name}")
        else:
            print(f"  Warning: Could not determine source file for {folder_name}")
    
    # Fallback to old method if makefile parsing fails
    else:
        if is_cuda_folder(folder_name):
            # Look for main.cu or any .cu file
            main_cu_path = os.path.join(dest_folder, 'main.cu')
            if os.path.exists(main_cu_path):
                modify_cuda_source(main_cu_path)
            else:
                # Find any .cu file
                for file in os.listdir(dest_folder):
                    if file.endswith('.cu'):
                        modify_cuda_source(os.path.join(dest_folder, file))
                        break
        elif is_omp_folder(folder_name):
            # Look for main.cpp or any .cpp file
            main_cpp_path = os.path.join(dest_folder, 'main.cpp')
            if os.path.exists(main_cpp_path):
                modify_omp_source(main_cpp_path)
            else:
                # Find any .cpp file
                for file in os.listdir(dest_folder):
                    if file.endswith('.cpp'):
                        modify_omp_source(os.path.join(dest_folder, file))
                        break

def handle_reference_header(example_bench_dir, ready_bench_src_dir, benchmark_name):
    """Check for reference.h in both CUDA and OpenMP folders and copy to both output folders."""
    cuda_folder = f"{benchmark_name}-cuda"
    omp_folder = f"{benchmark_name}-omp"
    
    # Check if both variants exist in source directory
    cuda_path = os.path.join(example_bench_dir, cuda_folder)
    omp_path = os.path.join(example_bench_dir, omp_folder)
    
    if not (os.path.isdir(cuda_path) and os.path.isdir(omp_path)):
        return False
    
    # Check if both variants exist in destination directory
    cuda_dest_folder = os.path.join(ready_bench_src_dir, cuda_folder)
    omp_dest_folder = os.path.join(ready_bench_src_dir, omp_folder)
    
    if not (os.path.isdir(cuda_dest_folder) and os.path.isdir(omp_dest_folder)):
        print(f"  Warning: Destination folders not found for {benchmark_name}, skipping reference.h")
        return False
    
    # Check if reference.h exists in either source folder
    cuda_ref_path = os.path.join(cuda_path, "reference.h")
    omp_ref_path = os.path.join(omp_path, "reference.h")
    
    ref_header_content = None
    ref_source = None
    
    # Check CUDA folder first
    if os.path.exists(cuda_ref_path):
        with open(cuda_ref_path, 'r') as f:
            ref_header_content = f.read()
        ref_source = "CUDA"
    
    # If not found in CUDA, check OpenMP folder
    elif os.path.exists(omp_ref_path):
        with open(omp_ref_path, 'r') as f:
            ref_header_content = f.read()
        ref_source = "OpenMP"
    
    # If found in either folder, write to both output folders
    if ref_header_content:
        cuda_dest_path = os.path.join(cuda_dest_folder, "reference.h")
        omp_dest_path = os.path.join(omp_dest_folder, "reference.h")
        
        # Write to CUDA output folder
        with open(cuda_dest_path, 'w') as f:
            f.write(ref_header_content)
        
        # Write to OpenMP output folder
        with open(omp_dest_path, 'w') as f:
            f.write(ref_header_content)
        
        print(f"  Copied reference.h from {ref_source} to both {cuda_folder} and {omp_folder}")
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Setup benchmark folders for differential fuzzing')
    parser.add_argument('--base-dir', default='.', help='Base directory containing example-bench')
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.base_dir)
    example_bench_dir = os.path.join(base_dir, 'example-bench', 'src')
    
    if not os.path.exists(example_bench_dir):
        print(f"Error: example-bench/src directory not found at {example_bench_dir}")
        return 1
    
    # Create ready-bench directory
    ready_bench_dir = create_ready_bench_dir(base_dir)
    ready_bench_src_dir = os.path.join(ready_bench_dir, 'src')
    os.makedirs(ready_bench_src_dir)
    
    # Process each folder in example-bench/src
    folders = [f for f in os.listdir(example_bench_dir) 
               if os.path.isdir(os.path.join(example_bench_dir, f))]
    
    # Group folders by benchmark name (removing -cuda/-omp suffix)
    benchmark_groups = {}
    
    for folder in sorted(folders):
        if is_cuda_folder(folder):
            benchmark_name = folder[:-5]  # Remove "-cuda"
            if benchmark_name not in benchmark_groups:
                benchmark_groups[benchmark_name] = {"cuda": folder, "omp": None}
            else:
                benchmark_groups[benchmark_name]["cuda"] = folder
        elif is_omp_folder(folder):
            benchmark_name = folder[:-4]  # Remove "-omp"
            if benchmark_name not in benchmark_groups:
                benchmark_groups[benchmark_name] = {"cuda": None, "omp": folder}
            else:
                benchmark_groups[benchmark_name]["omp"] = folder
    
    processed_count = 0
    
    # Process each benchmark group
    for benchmark_name, variants in benchmark_groups.items():
        print(f"Processing benchmark: {benchmark_name}")
        
        # Process CUDA variant
        if variants["cuda"]:
            folder = variants["cuda"]
            src_folder_path = os.path.join(example_bench_dir, folder)
            dest_folder_path = os.path.join(ready_bench_src_dir, folder)
            
            print(f"  Processing {folder}...")
            try:
                process_folder(src_folder_path, dest_folder_path)
                processed_count += 1
            except Exception as e:
                print(f"  Error processing {folder}: {e}")
        
        # Process OMP variant
        if variants["omp"]:
            folder = variants["omp"]
            src_folder_path = os.path.join(example_bench_dir, folder)
            dest_folder_path = os.path.join(ready_bench_src_dir, folder)
            
            print(f"  Processing {folder}...")
            try:
                process_folder(src_folder_path, dest_folder_path)
                processed_count += 1
            except Exception as e:
                print(f"  Error processing {folder}: {e}")
        
        # Handle reference.h files if both variants exist
        if variants["cuda"] and variants["omp"]:
            handle_reference_header(example_bench_dir, ready_bench_src_dir, benchmark_name)
    
    print(f"\nProcessed {processed_count} folders.")
    print(f"Ready-bench directory created at: {ready_bench_dir}")
    
    return 0

if __name__ == '__main__':
    exit(main())