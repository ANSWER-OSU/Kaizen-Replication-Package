#!/usr/bin/env python3
"""
Simplified AFL fuzzing script for running inside Apptainer with custom fuzzers.

Usage:
    python run_afl_fuzzing.py <driver_path> <fuzzer_name> [--seeds SEED1 SEED2 ...] [--timeout 1200]

Example:
    # Single seed
    python run_afl_fuzzing.py /path/to/driver custom_fuzzer_cross --seeds 7632 --timeout 1200
    
    # Multiple seeds (parallel execution)
    python run_afl_fuzzing.py /path/to/driver custom_fuzzer_aidw --seeds 7632 1324 8742 9088 3739 --timeout 1200
    
For running with 2 GPUs, you can run 2-3 seeds per GPU (4-6 total).
Each AFL process will use one CPU core, so ensure you have enough cores.
"""

import os
import sys
import argparse
import subprocess
import threading
from pathlib import Path

# Default seeds
DEFAULT_SEEDS = [7632, 1324, 8742, 9088, 3739]

def run_afl_single_seed(driver_path, fuzzer_name, seed, timeout, exec_timeout, fuzzer_dir, shared_inputs=False):
    """Run AFL fuzzing for a single seed"""
    
    # Get the directory containing the driver
    driver_dir = Path(driver_path).parent.absolute()
    
    # Use shared or seed-specific directories
    if shared_inputs:
        inputs_dir = driver_dir / "inputs"
        outputs_dir = driver_dir / f"outputs_{seed}"
        
        # Create directories (inputs should already exist)
        inputs_dir.mkdir(exist_ok=True)
        outputs_dir.mkdir(exist_ok=True)
    else:
        # Create seed-specific directories
        inputs_dir = driver_dir / f"inputs_{seed}"
        outputs_dir = driver_dir / f"outputs_{seed}"
        
        # Create directories
        inputs_dir.mkdir(exist_ok=True)
        outputs_dir.mkdir(exist_ok=True)
        
        # Create a simple input file (adjust this based on your program)
        input_file = inputs_dir / "seed.txt"
        with open(input_file, 'w') as f:
            f.write("100 10 5")  # Default input, customize as needed
    
    # Set up environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fuzzer_dir)
    env["AFL_PYTHON_MODULE"] = fuzzer_name
    env["AFL_CUSTOM_MUTATOR_ONLY"] = "1"
    env["AFL_DISABLE_TRIM"] = "1"
    env["AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES"] = "1"
    env["AFL_NO_FORKSRV"] = "1"
    
    # Construct AFL command
    cmd = [
        "timeout", str(timeout),
        "afl-fuzz",
        "-s", str(seed),
        "-t", str(exec_timeout),  # Execution timeout in milliseconds
        "-i", str(inputs_dir),
        "-o", str(outputs_dir),
        "--",
        str(driver_path)
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting AFL fuzzing with seed {seed}")
    print(f"Driver: {driver_path}")
    print(f"Fuzzer: {fuzzer_name}")
    print(f"Session timeout: {timeout} seconds")
    print(f"Execution timeout: {exec_timeout} ms ({exec_timeout/1000:.1f} seconds)")
    print(f"Outputs: {outputs_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, env=env, cwd=driver_dir)
        
        if result.returncode == 0:
            print(f"\n✓ Seed {seed} completed successfully")
        elif result.returncode == 124:  # timeout exit code
            print(f"\n✓ Seed {seed} completed (timeout reached)")
        else:
            print(f"\n✗ Seed {seed} failed with exit code {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"\n✗ Seed {seed} failed with exception: {e}")
        return 1

def run_afl_parallel(driver_path, fuzzer_name, seeds, timeout, exec_timeout, fuzzer_dir, shared_inputs=False):
    """Run AFL fuzzing in parallel for multiple seeds"""
    
    print(f"\n{'='*60}")
    print(f"Running AFL fuzzing in parallel")
    print(f"Seeds: {seeds}")
    print(f"Total parallel instances: {len(seeds)}")
    print(f"Shared inputs: {shared_inputs}")
    print(f"{'='*60}\n")
    
    threads = []
    results = {}
    
    def worker(seed):
        exit_code = run_afl_single_seed(driver_path, fuzzer_name, seed, timeout, exec_timeout, fuzzer_dir, shared_inputs)
        results[seed] = exit_code
    
    # Start threads
    for seed in seeds:
        thread = threading.Thread(target=worker, args=(seed,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Report results
    print(f"\n{'='*60}")
    print("AFL Fuzzing Results Summary")
    print(f"{'='*60}")
    
    success_count = 0
    for seed in seeds:
        exit_code = results.get(seed, -1)
        if exit_code in [0, 124]:  # 0 = normal exit, 124 = timeout
            print(f"  ✓ Seed {seed}: SUCCESS")
            success_count += 1
        else:
            print(f"  ✗ Seed {seed}: FAILED (exit code {exit_code})")
    
    print(f"\nTotal: {success_count}/{len(seeds)} successful")
    print(f"{'='*60}\n")
    
    return success_count == len(seeds)

def main():
    parser = argparse.ArgumentParser(
        description='Run AFL fuzzing with custom mutators inside Apptainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single seed
  python run_afl_fuzzing.py ./driver custom_fuzzer_cross --seeds 7632 --timeout 1200
  
  # Multiple seeds (parallel) - all 5 default seeds
  python run_afl_fuzzing.py ./driver custom_fuzzer_aidw --seeds 7632 1324 8742 9088 3739 --timeout 1200
  
  # Use all default seeds
  python run_afl_fuzzing.py ./driver custom_fuzzer_lif --timeout 1200
  
GPU Usage:
  With 2 GPUs, you can run 2-3 seeds per GPU safely (4-6 total parallel instances).
  The script will distribute the load automatically.
  
Note:
  Run this script INSIDE Apptainer with --nv flag:
    apptainer shell --nv kaizen.sif
    python run_afl_fuzzing.py ./driver custom_fuzzer_name --timeout 1200
        """
    )
    
    parser.add_argument('driver', help='Path to the AFL driver executable')
    parser.add_argument('fuzzer', help='Custom fuzzer module name (without .py extension)')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                       help=f'Seeds for AFL fuzzing (default: {DEFAULT_SEEDS})')
    parser.add_argument('--timeout', type=int, default=1200,
                       help='Timeout in seconds for each AFL run (default: 1200 = 20 minutes)')
    parser.add_argument('--exec-timeout', type=int, default=6000,
                       help='Timeout in milliseconds per execution (default: 6000ms = 6s)')
    parser.add_argument('--shared-inputs', action='store_true',
                       help='Use shared inputs/ folder for all seeds (recommended for parallel runs)')
    parser.add_argument('--fuzzer-dir', type=str, 
                       default='/nfs/stak/users/motwanim/software/LLM-Code-Porting-Testing/resources/differential-testing/custom-input-fuzzing',
                       help='Directory containing custom fuzzer modules')
    
    args = parser.parse_args()
    
    # Validate driver exists
    driver_path = Path(args.driver).absolute()
    if not driver_path.exists():
        print(f"Error: Driver not found: {driver_path}")
        sys.exit(1)
    
    if not os.access(driver_path, os.X_OK):
        print(f"Error: Driver is not executable: {driver_path}")
        sys.exit(1)
    
    # Validate fuzzer directory exists
    fuzzer_dir = Path(args.fuzzer_dir)
    if not fuzzer_dir.exists():
        print(f"Error: Fuzzer directory not found: {fuzzer_dir}")
        sys.exit(1)
    
    # Check if fuzzer module exists
    fuzzer_file = fuzzer_dir / f"{args.fuzzer}.py"
    if not fuzzer_file.exists():
        print(f"Error: Fuzzer module not found: {fuzzer_file}")
        print(f"Available fuzzers in {fuzzer_dir}:")
        for f in fuzzer_dir.glob("*.py"):
            if not f.name.startswith("_"):
                print(f"  - {f.stem}")
        sys.exit(1)
    
    # Check AFL is available
    try:
        subprocess.run(["which", "afl-fuzz"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: afl-fuzz not found. Make sure AFL++ is installed.")
        sys.exit(1)
    
    # Warn about GPU usage
    if len(args.seeds) > 6:
        print(f"\nWarning: Running {len(args.seeds)} parallel instances with 2 GPUs")
        print("Recommended: 2-3 seeds per GPU (4-6 total)")
        print("Consider reducing the number of seeds to avoid resource contention\n")
    
    # Run fuzzing
    print(f"AFL Fuzzing Configuration:")
    print(f"  Driver: {driver_path}")
    print(f"  Fuzzer: {args.fuzzer}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Session timeout: {args.timeout} seconds ({args.timeout//60} minutes)")
    print(f"  Execution timeout: {args.exec_timeout} ms ({args.exec_timeout/1000:.1f} seconds)")
    print(f"  Shared inputs: {args.shared_inputs}")
    print(f"  Fuzzer directory: {fuzzer_dir}")
    
    if len(args.seeds) == 1:
        # Single seed mode
        exit_code = run_afl_single_seed(
            driver_path, args.fuzzer, args.seeds[0], args.timeout, args.exec_timeout, fuzzer_dir, args.shared_inputs
        )
        sys.exit(0 if exit_code in [0, 124] else 1)
    else:
        # Multi-seed parallel mode
        success = run_afl_parallel(
            driver_path, args.fuzzer, args.seeds, args.timeout, args.exec_timeout, fuzzer_dir, args.shared_inputs
        )
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
