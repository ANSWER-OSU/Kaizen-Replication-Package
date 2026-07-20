import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any

import torch
from dotenv import dotenv_values
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Constants
SAMPLES = 1
BATCH_SIZE = 50
LORA_RANK = 64
MAX_TOKENS = 5120

def find_kernel_files(base_dir: str) -> List[str]:
    """Recursively finds all 'extracted-kernels-*.txt' files."""
    kernel_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("extracted-kernels-") and file.endswith(".txt"):
                kernel_files.append(os.path.join(root, file))
    return kernel_files

def parse_kernels(file_path: str) -> List[Dict[str, Any]]:
    """Parses a kernel file and returns a list of task objects."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []

    pattern = re.compile(
        r"============\d+:([^=]+)============\n(.*?)(?=(?:============\d+:)|$)",
        re.DOTALL
    )
    
    tasks = []
    for match in pattern.finditer(content):
        name = match.group(1).strip()
        code = match.group(2).strip()
        
        if not code:
            continue
            
        # Filter comments (keep lines without '///')
        filtered_code = '\n'.join(
            line for line in code.split('\n') if '///' not in line
        )
        
        if not filtered_code.strip():
            continue
            
        question = f"Can you translate this CUDA code to OpenMP offload? \n{filtered_code}"
        context = "CUDA to OpenMP offload translation"
        
        tasks.append({
            "source_file": file_path,
            "kernel_name": name,
            "data": {
                "question": question,
                "context": context,
                "answer": {}
            }
        })
    return tasks

def extract_answer(response: str) -> str:
    """Extracts the code block from the LLM response."""
    parts = response.split("### Response:\n")
    return parts[1] if len(parts) > 1 else response

def format_prompt(item: Dict[str, Any]) -> str:
    """Formats the prompt for the model."""
    src, tgt = "CUDA", "OpenMP"
    context = item['context']
    if "CUDA to SYCL" in context:
        tgt = "SYCL"
    elif "SYCL to OpenMP" in context:
        src = "SYCL"
        
    return (
        f"You are a powerful LLM model for translating {src} codes to {tgt} codes.\n\n"
        f"You must output the {tgt} code for the given {src} code.\n\n"
        f"### Input:\n{item['question']}\n\n"
        f"### Context:\n{context}\n\n"
        "### Response:\n"
    )

def get_model_name_from_checkpoint(checkpoint_dir: str) -> str:
    """Reads base_model_name_or_path from adapter_config.json located in the checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
            if "base_model_name_or_path" not in config:
                print(f"Error: 'base_model_name_or_path' not found in {config_path}")
                sys.exit(1)
                
            return config["base_model_name_or_path"]
    except Exception as e:
        print(f"Error reading {config_path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run batch translation on extracted kernels using vLLM.")
    parser.add_argument("benchmark_path", help="Path to the benchmark output directory")
    parser.add_argument("checkpoint", help="Path to the LoRA checkpoint directory")
    args = parser.parse_args()

    if not os.path.isdir(args.benchmark_path):
        print(f"Error: Directory not found: {args.benchmark_path}")
        sys.exit(1)

    # Setup paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Load env (optional based on usage but kept for compatibility)
    env_file = os.path.join(project_root, '.env')
    if os.path.exists(env_file):
        dotenv_values(env_file)

    checkpoint_dir = args.checkpoint
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    print(f"Checkpoint Directory: {checkpoint_dir}")
    model_name = get_model_name_from_checkpoint(checkpoint_dir)
    print(f"Base Model Name: {model_name}")
    
    # Scan files
    print(f"Scanning '{args.benchmark_path}' for kernels...")
    kernel_files = find_kernel_files(args.benchmark_path)
    if not kernel_files:
        print("No kernel files found.")
        sys.exit(0)

    # Parse all tasks
    all_tasks = []
    print(f"Parsing {len(kernel_files)} files...")
    for f in kernel_files:
        all_tasks.extend(parse_kernels(f))
    
    total_tasks = len(all_tasks)
    print(f"Total extracted kernels: {total_tasks}")
    if total_tasks == 0:
        print("No valid kernels found in files.")
        sys.exit(0)

    # Initialize vLLM
    print("Initializing vLLM...")
    try:
        llm = LLM(
            model=model_name,
            enable_lora=True,
            max_lora_rank=LORA_RANK
        )
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        sys.exit(1)

    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=MAX_TOKENS, 
        n=SAMPLES
    )
    lora_request = LoRARequest("adapter", 1, checkpoint_dir)

    # Pre-organize tasks by source file for easier partial saving
    tasks_by_file = {}
    for task in all_tasks:
        tasks_by_file.setdefault(task['source_file'], []).append(task['data'])

    # Batch Processing
    print(f"Starting batch translation (Batch Size: {BATCH_SIZE})...")
    
    # Process in chunks
    for i in range(0, total_tasks, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_tasks)
        chunk = all_tasks[i : batch_end]
        
        print(f"Processing batch {i//BATCH_SIZE + 1} / {(total_tasks + BATCH_SIZE - 1)//BATCH_SIZE} "
              f"({len(chunk)} items)...")
        
        prompts = [format_prompt(t['data']) for t in chunk]
        
        affected_files = set()
        
        try:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            
            # Map outputs back to tasks
            for j, output in enumerate(outputs):
                full_text = output.outputs[0].text
                answer = extract_answer(full_text)
                chunk[j]['data']['answer']['0'] = answer
                affected_files.add(chunk[j]['source_file'])
                
            # Incremental Save
            print(f"Saving progress for {len(affected_files)} files...")
            for source_file in affected_files:
                data = tasks_by_file[source_file]
                
                # Determine output path (same logic as before)
                parent_dir = os.path.dirname(os.path.dirname(source_file))
                translated_dir = os.path.join(parent_dir, "translated-kernels-codellama")
                os.makedirs(translated_dir, exist_ok=True)
                
                basename = os.path.basename(source_file)
                match = re.search(r'extracted-kernels-(\d+)\.txt', basename)
                
                if match:
                    out_name = f"translated-kernels-codellama-{match.group(1)}.json"
                    out_path = os.path.join(translated_dir, out_name)
                    try:
                        with open(out_path, 'w') as f:
                            json.dump(data, f, indent=2)
                    except IOError as e:
                        print(f"Error writing to {out_path}: {e}")
                else:
                    print(f"Warning: Could not determine output filename for {basename}")

        except Exception as e:
            print(f"Error processing batch: {e}")

    print("Translation pipeline complete.")

if __name__ == "__main__":
    main()
