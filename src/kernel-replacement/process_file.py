#====================================================================
# Author: Zheming Jin (jinz@ornl.gov)
#====================================================================

import argparse
import json
import re
import subprocess
from replace_function import replace_function, replace_file
# Don't need training json file during verification
# from metrics import correct_rate

# find the kernel name and path for a benchmark
def search_benchmark(name, function_list, prefix_path='HeCBench'):
    for f in function_list:
        if name == f["function_name"]:
            benchmark_name = f["path"].split("/")[-2] 
            # temp pring benchmark_name, prefix_path + '/' + f["path"]
            print(f"Found the benchmark name {benchmark_name} and path {prefix_path + '/' + f['path']}")
            return benchmark_name, prefix_path + '/' + f["path"]
    return None, None


def verifier (args):
    file_path = args.json_path
    config_path = args.config_path
    bench_path = args.bench_path
    compiler_name = args.compiler_name
    sanity_check = args.sanity_check
    save_summary = args.save_summary
    load_filter = args.load_filter
    restore_only = args.restore_only
    list_kernels = args.list_kernels
    output_key_name = args.output_key_name
    input_key_name = args.input_key_name
   
    if output_key_name == input_key_name:
        number_samples = 1
    else:
        number_samples = args.number_samples

    print(file_path)
    print(config_path)
    print(bench_path)

    try:
        with open(config_path, 'r') as file:
            function_list = json.load(file)
    except Exception as e:
        print(f"Failed to open the file {config_path}: {e}")
        return

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Failed to open the file {file_path}: {e}")
        return

    if load_filter:
        try:
            with open(load_filter, 'r') as file:
                summary = json.load(file)
                print(summary) 
        except Exception as e:
            print(f"Failed to open the file {load_summary}: {e}")
            return

    # function name pattern does not care about template
    # Assumption: there is no "\n" after "(", for example, attention_kernel2 (\n ...) 
    # fname_pattern = r"\s*void\s+(\w+)\s*\(.*?\)"  
    # New pattern matcher
    fname_pattern = r"(?:__global__\s+)?void\s+(\w+)\s*\("

    # KeyError may indicate errors in subset.json
    error_pattern = r"\bFAIL\b|\bMismatch\b"
    transform_pattern = r"\bFAILED(?!=transform-warning)\b"
    assert_pattern = r"\bfailed(?! as expected)\b"

    # Set the options for the OpenMP and SYCL compilers
    if "omp" in args.target_lang: 
        if "clang" in compiler_name:
            backend_option = "--amd-arch"
            arch_option = args.amd_arch
        elif "nvc" in compiler_name:
            backend_option = "--nvidia-sm"
            arch_option = args.nvidia_sm

    elif "sycl" in args.target_lang: 
        if "hip" in args.sycl_type:
            backend_option = "--amd-arch"
            arch_option = args.amd_arch
        elif "cuda" in args.sycl_type:
            backend_option = "--nvidia-sm"
            arch_option = args.nvidia_sm

    for n in range(number_samples):

        # log of benchmarks status
        bench_status = {'fail' : [],
                        'pass' : [],
                        'skip' : [],
                        'misc' : [],
                        'any'  : [] }
        success = 0

        # Debug print file name
        print(f"Processing file: {file_path}")

        for item in data:  
            src_function_code = (item[args.input_key_name])
            print("Source function code:", src_function_code)
            match1 = re.search(fname_pattern, src_function_code)
            # debug print match1
            print("Match1:", match1)


            if list_kernels:
                function_name = match1.group(1)
                print(f"Found the function name {function_name}")
                # count the number of lines in a function
                loc = src_function_code.count('\n')
                print(f"Lines of code: {loc}")
                continue

            if sanity_check:
                match2 = match1
            else:
                if output_key_name == input_key_name:
                     function_code = (item[output_key_name])
                else:
                     # function code from the n-th model response
                     function_code = (item[output_key_name][str(n)])
                match2 = re.search(fname_pattern, function_code)

            if match1 and match2:
                src_function_name = match1.group(1)
                function_name = match2.group(1)
                if src_function_name != function_name:  # Check that LLM produced a different function name
                    print(f"Warning: LLM produced a different function name: {function_name} != {src_function_name}")
                    print(f"Warning: Replace {function_name} with {src_function_name}")
                    function_code = function_code.replace(function_name, src_function_name)  # replace the function name in the LLM response
                else:
                    print(f"Found the function name {function_name}")

                # use the reference function name 
                # sanity check also verifies whether a kernel name exists in the function path
                benchmark_name, function_path = search_benchmark(src_function_name, function_list, prefix_path=bench_path)
                if benchmark_name and function_path:
                    print(f"Found the function path {function_path} and benchmark name {benchmark_name}")

                    if restore_only:
                        replace_file(function_path, function_path + '.bak')
                        continue

                    # verify the benchmarks that failed
                    bench = benchmark_name + ":" + src_function_name
                    bench_status['any'].append(bench)
                    if load_filter:
                        if bench not in summary['fail']:
                            bench_status['skip'].append(bench)
                            continue

                    if not sanity_check:
                        status = replace_function(function_path, src_function_name, function_code)
                        if not status:
                            bench_status['misc'].append(benchmark_name + ':' + src_function_name)
                            continue

                    # cause = subprocess.run(["python",
                    #                         bench_path + "/scripts/autohecbench.py",
                    #                         "--yes-prompt",
                    #                         "--bench-dir",
                    #                         bench_path,
                    #                         "--clean",  
                    #                         "--compiler-name",
                    #                         compiler_name,
                    #                         "--sycl-type",
                    #                         args.sycl_type,
                    #                         # backend_option,
                    #                         # arch_option,
                    #                         "--verbose",  # debugging
                    #                         benchmark_name],
                    #                         #check=False, capture_output=True,
                    #                         stdout=subprocess.PIPE,
                    #                         encoding="utf-8")
                    # print(cause.stdout)
                    # fail1 = re.search(error_pattern, cause.stdout, re.IGNORECASE)
                    # fail2 = re.search(transform_pattern, cause.stdout, re.IGNORECASE)
                    # fail3 = re.search(assert_pattern, cause.stdout, re.IGNORECASE)
                    # # TODO verification for each benchmark in HeCBench
                    # if fail1 or fail2 and fail3:
                    #     bench_status['fail'].append(bench)
                    # else:
                    #     success += 1
                    #     bench_status['pass'].append(bench)
                else:
                    print(f"Failed to find the function path {function_path}")
                    if benchmark_name is None:
                        benchmark_name = 'NONE'
                    bench_status['misc'].append(benchmark_name + ':' + src_function_name)
            else:
                print(f"Failed to extract the function name from the question: {src_function_code}")
                bench_status['misc'].append("NONE:NONE")

        if save_summary:
            summary_path = "sample_ID" + str(n) + "_" + save_summary
            with open(summary_path , "w") as f:
                json.dump(bench_status, f, indent=4, sort_keys=True)
            print(f"Wrote the verification results to {summary_path}.")

        total = len(data) # data is a list
        print(file_path)
        print(f"Sample ID: {n}")
        print(f"Total number of benchmarks: {total}")
        print(f"Total number of pass benchmarks: {success}")
        print(f"Correct rate {(100 * success / total):>0.1f}%")
        print(f"List of benchmarks that cause miscellaneous issues: {bench_status['misc']}")
        print(f"List of benchmarks that failed in compile, run and/or verification: {bench_status['fail']}")
        print("\n");
        print(f"List of benchmarks that are successful in compile and run: {bench_status['pass']}")

        # Don't need training json file during verification
        # correct_rate(args, bench_status)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify the HeCBench benchmarks generated by LLMs')
    parser.add_argument('--json_path', required=True,
                        help='Path to the JSON file with LLM responses')
    # parser.add_argument('--train_path', required=True,
    #                     help='Path to the JSON file for fine tuning a LLM')
    parser.add_argument('--config_path', default='benchmarks.config',
                        help='Path to the benchmark configuration file')
    parser.add_argument('--bench_path', default='/noback/5zj/HeCBench-SYCL/src/',
                        help='Path to the benchmark directory')
    parser.add_argument('--compiler_name', default='icpx',
                        help='Compiler to build a benchmark')
    parser.add_argument('--target_lang', choices=['sycl', 'omp'], default='omp',
                        help='target programming language')
    parser.add_argument('--sycl_type', choices=['cuda', 'hip', 'opencl', 'cpu'], default='cuda',
                        help='Type of SYCL device to use (default is cuda)')
    parser.add_argument('--nvidia_sm', default='80',
                        help='NVIDIA SM version (default is 80)')
    parser.add_argument('--amd_arch', default='gfx908',
                        help='AMD Architecture (default is gfx908)')
    parser.add_argument('--sanity_check', action='store_true',
                        help='Verify reference benchmarks in HeCBench')
    parser.add_argument('--number_samples', type=int, default=1,
                        help='Number of LLM responses per question')
    parser.add_argument('--list_kernels', action='store_true',
                        help='List kernel names and their lines of code in a dataset')
    parser.add_argument('--restore_only', action='store_true',
                        help='Restore reference benchmarks only')
    parser.add_argument('--save_summary',
                        help='Save the verification results')
    parser.add_argument('--load_filter',
                        help='Load a JSON file to verify the benchmarks that failed previously')
    parser.add_argument('--input_key_name', default="question",
                        help='name of input key')
    parser.add_argument('--output_key_name', default="answer",
                        help='name of output key')
    
    args = parser.parse_args()
    verifier(args)
