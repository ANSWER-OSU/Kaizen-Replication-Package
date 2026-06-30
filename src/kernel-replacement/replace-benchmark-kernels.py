import os
import sys
import shutil
import re
from pathlib import Path
from process_file import verifier
from replace_function import replace_file
import argparse

def load_environment():
    script_dir = Path(__file__).parent
    env_file = script_dir / '..' / '..' / '.env'
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HECBENCH_SRC_DIR='):
                        value = line.split('=', 1)[1].strip('"\'')
                        return value
        except Exception as e:
            print(f"Error reading .env file: {e}")
    else:
        print(f"Warning: .env file not found at {env_file}")
    return None

def extract_example_name(folder_name):
    match = re.match(r'^(.+?)-seed\d+', folder_name)
    if match:
        return match.group(1)
    return None

def extract_id_from_filename(filename, llm_name):
    pattern = rf'translated-kernels-{re.escape(llm_name)}-(\d+)\.json'
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    # fallback: original format
    match = re.match(r'translated-kernels-(\d+)\.json', filename)
    if match:
        return match.group(1)
    return None

def copy_benchmark_source(hecbench_src_dir, example_name, target_dir):
    source_path = Path(hecbench_src_dir) / f"{example_name}-omp"
    target_path = Path(target_dir) / "working-copy-omp"
    if source_path.exists():
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)
        print(f"Copied {source_path} to {target_path}")
        return True
    else:
        print(f"Warning: Source benchmark {source_path} not found")
        return False

def process_translated_kernel(kernel_file, config_path, example_folder, kernel_id, hecbench_src_dir, llm_name):
    try:
        import json
        with open(kernel_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read {kernel_file}: {e}")
        return False

    try:
        with open(config_path, 'r') as f:
            function_list = json.load(f)
    except Exception as e:
        print(f"Failed to read config {config_path}: {e}")
        return False

    fname_pattern = r"(?:__global__\s+)?void\s+(\w+)\s*\("
    files_to_copy = set()

    for item in data:
        src_function_code = item.get('question', '')
        if not src_function_code:
            continue
        src_match = re.search(fname_pattern, src_function_code)
        if not src_match:
            continue
        src_function_name = src_match.group(1)
        for f in function_list:
            if src_function_name == f["function_name"]:
                file_path = Path(hecbench_src_dir) / f["path"]
                files_to_copy.add(file_path)
                print(f"Function {src_function_name} will modify {file_path}")
                break

    for file_path in files_to_copy:
        backup_path = Path(str(file_path) + '.bak')
        if backup_path.exists():
            print(f"Restoring {file_path} from {backup_path}")
            shutil.copy(backup_path, file_path)

    args = argparse.Namespace()
    args.json_path = str(kernel_file)
    args.config_path = config_path
    args.bench_path = hecbench_src_dir
    args.compiler_name = "clang++"
    args.target_lang = "omp"
    args.sycl_type = "cuda"
    args.nvidia_sm = "80"
    args.amd_arch = "gfx908"
    args.sanity_check = False
    args.save_summary = None
    args.load_filter = None
    args.restore_only = False
    args.list_kernels = False
    args.number_samples = 1
    args.input_key_name = "question"
    args.output_key_name = "answer"

    print(f"Processing {kernel_file} with llm={llm_name}")

    try:
        verifier(args)

        output_dir = Path(example_folder) / f"translated-outputs-{llm_name}"
        output_dir.mkdir(exist_ok=True)

        copied_count = 0
        for file_path in files_to_copy:
            if file_path.exists():
                base_name = file_path.stem
                extension = file_path.suffix
                output_filename = f"{base_name}-{kernel_id}{extension}"
                output_path = output_dir / output_filename
                shutil.copy2(file_path, output_path)
                print(f"Copied modified file to {output_path}")
                copied_count += 1
            else:
                print(f"Warning: Expected modified file not found: {file_path}")

        print(f"Copied {copied_count} specific modified files for kernel {kernel_id}")
        return True

    except Exception as e:
        print(f"Error processing {kernel_file}: {e}")
        return False

def process_example_folder(example_folder, config_path, hecbench_src_dir, llm_name):
    folder_name = Path(example_folder).name
    example_name = extract_example_name(folder_name)

    if not example_name:
        print(f"Could not extract example name from {folder_name}")
        return False

    print(f"Processing example: {example_name} from folder: {folder_name}")
    copy_benchmark_source(hecbench_src_dir, example_name, example_folder)

    translated_kernels_dir = Path(example_folder) / f"translated-kernels-{llm_name}"

    if not translated_kernels_dir.exists():
        print(f"No translated-kernels-{llm_name} directory found in {example_folder}")
        return False

    success = False
    for kernel_file in sorted(translated_kernels_dir.glob("translated-kernels-*.json")):
        kernel_id = extract_id_from_filename(kernel_file.name, llm_name)
        if kernel_id:
            result = process_translated_kernel(kernel_file, config_path, example_folder, kernel_id, hecbench_src_dir, llm_name)
            success = success or result
        else:
            print(f"Could not extract ID from {kernel_file.name}")

    return success

def restore_all_backups(hecbench_src_dir):
    hecbench_path = Path(hecbench_src_dir)
    backup_files = list(hecbench_path.glob("**/*.bak"))
    if backup_files:
        print(f"\nRestoring {len(backup_files)} backup files...")
        for bak_file in backup_files:
            orig_file = bak_file.with_suffix('')
            replace_file(str(orig_file), str(bak_file))
            print(f"Restored {orig_file}")
    else:
        print("\nNo backup files found to restore.")

def main():
    parser = argparse.ArgumentParser(description="Replace benchmark kernels with LLM translations.")
    parser.add_argument("benchmark_dir", help="Path to the benchmark output directory")
    parser.add_argument("--llm", required=True, choices=["codellama", "starcoder", "hpccoder"],
                        help="LLM name to use for translated-kernels-<llm> folder")
    args = parser.parse_args()

    if not os.path.isdir(args.benchmark_dir):
        print(f"Error: {args.benchmark_dir} is not a valid directory")
        sys.exit(1)

    hecbench_src_dir = load_environment()
    if not hecbench_src_dir:
        print("Error: Could not load HECBENCH_SRC_DIR from .env file")
        sys.exit(1)

    config_path = str(Path(__file__).parent / "omp-benchmarks.config")
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    print(f"Benchmark directory: {args.benchmark_dir}")
    print(f"LLM: {args.llm}")
    print(f"HeCBench source directory: {hecbench_src_dir}")
    print(f"Configuration file: {config_path}")

    benchmark_path = Path(args.benchmark_dir)
    example_folders = [d for d in benchmark_path.iterdir()
                      if d.is_dir() and re.match(r'.+-seed\d+', d.name)]

    if not example_folders:
        print(f"No example folders found in {args.benchmark_dir}")
        sys.exit(1)

    print(f"Found {len(example_folders)} example folders to process")

    processed_count = 0
    for example_folder in sorted(example_folders):
        print(f"\n{'='*60}")
        print(f"Processing: {example_folder}")
        print(f"{'='*60}")
        try:
            if process_example_folder(str(example_folder), config_path, hecbench_src_dir, args.llm):
                processed_count += 1
        except Exception as e:
            print(f"Error processing {example_folder}: {e}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed {processed_count}/{len(example_folders)} example folders")
    print(f"{'='*60}")

    restore_all_backups(hecbench_src_dir)

if __name__ == "__main__":
    main()