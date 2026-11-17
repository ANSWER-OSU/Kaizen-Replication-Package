# Setup Guide (Clean Install on a New System)

This guide documents how to set up, build, and run the **Kaizen** framework from a clean clone of `Kaizen-Replication-Package` on a Linux machine or HPC cluster. Because several scripts contain cluster-specific commands or placeholder paths that match their original build layout, follow the configuration and path edits below before running each stage.

> **Reproducing paper figures & tables from precomputed results only:** If you only want to regenerate the paper's RQ1–RQ4 tables, statistics, and figures from the shipped result data, you do **not** need Apptainer, CUDA, or LLMs—see [`src/analysis-scripts/README.md`](src/analysis-scripts/README.md).

---

## 1. Prerequisites

### Host System & Hardware
- **OS:** Linux (x86_64)
- **Container runtime:** [Apptainer](https://apptainer.org/) (or compatible Singularity runtime)
  - Building `build/kaizen.sif` requires either `--fakeroot` support (`/etc/subuid` and `/etc/subgid` configured with unprivileged user namespaces) **or** `root`/`sudo` access (or building on another machine and copying the `.sif` file).
  - Internet access is needed during container build to pull `docker://nvidia/cuda:12.0.1-devel-ubuntu22.04`, Ubuntu packages, LLVM 18 (`apt.llvm.org`), and PyPI packages.
- **GPU:** NVIDIA GPU(s) with a host driver compatible with CUDA 12.x (`--nv` passthrough for Apptainer).
  - *Optional / LLM Translation:* Multi-GPU or high-VRAM NVIDIA GPUs (e.g., for CodeLlama-13B, HPC-Coder-6.7B, StarCoderBase-7B with `vllm`).
- **Disk space:** Plan for at least ~15–20 GB for the Apptainer image (`build/kaizen.sif` is ~7.5 GB) and build artifacts, plus space for external dependencies (AFL++, HeCBench, Hugging Face model weights, and experiment outputs).

### Host Python Environment (Only Needed Outside the Container)
The Apptainer image includes PyTorch, `transformers`, `peft`, `accelerate`, and `bitsandbytes` (from `src/apptainer/requirements.txt`), **but it does not include `vllm`, `python-dotenv`, or `pandas`**. If you run the batch LLM translation scripts (`src/translation/batch-translate-*.py`) on the host, create a Python virtual environment (`python3 >= 3.10`) and install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install vllm torch python-dotenv pandas
```

---

## 2. Repository Structure & External Downloads

### Included in `Kaizen-Replication-Package`
- `src/apptainer/` — Apptainer definition file, CMake configuration, Python requirements, and build scripts
- `src/kernel-fuzzing/` — C++ source for `kaizen` (kernel identification, mutation, workflow orchestrator), `benchmark-setup/` (HeCBench preparation scripts), `config/` (`kaizen.cfg`, `mutation-options.cfg`), `resources/mutation/*.txt`, and `run.sh`
- `src/kernel-extraction/` — Standalone Clang AST tool (`kernel-extractor`) and wrapper scripts to extract mutated CUDA kernels from `source-queue/`
- `src/translation/` — vLLM batch translation scripts (`batch-translate-*.py`) and fine-tuned PEFT/LoRA checkpoints (`checkpoints/codellama-13b/`, `checkpoints/hpc-coder-v2-6.7b/`, `checkpoints/starcoderbase-7b/`)
- `src/kernel-replacement/` — Cross-file kernel replacement scripts (`replace-benchmark-kernels.py`, `omp-benchmarks.config`)
- `src/differential-testing/` — Differential testing driver (`differential-driver.cpp`), 14 benchmark-specific AFL++ custom Python mutators (`AFL-custom-fuzzers/`), seed inputs (`seeds/`), and per-variant AFL++ runners (`launch-afl.sh`, `run_afl_fuzzing.py`)
- `src/analysis-scripts/` — Python and R scripts reproducing RQ1–RQ4 tables, statistics, and figures from precomputed data
- `results/`, `experiment_logs/` — Precomputed experimental results and execution logs

### External Resources You Must Download or Supply
1. **AFL++ source tree** (originally built against `v4.33c`):
   - Download from <https://github.com/AFLplusplus/AFLplusplus> (e.g., into `resources/AFLplusplus/` inside the repo root, or wherever you point `src/apptainer/kaizen-apptainer.def`).
2. **HeCBench benchmark suite**:
   - Download from <https://github.com/zjin-lcf/HeCBench/tree/HeCBench-OMP> (e.g., placing its `src/` directory at `example-bench/src/`), then prepare `ready-bench/src/` using `src/kernel-fuzzing/benchmark-setup/` (see Step 4.2).
   - *(Note: `results/RQ4/omp-original-programs-(developer-written)/` contains unmodified OpenMP reference programs only and no CUDA sources, so it does not replace a full HeCBench checkout.)*
3. **Base LLM weights (for Stage 3 translation):**
   - Hugging Face access or local weights for the base models referenced by `src/translation/checkpoints/*/adapter_config.json` (`meta-llama/CodeLlama-13b-hf`, `hpcgroup/hpc-coder-v2-6.7b`, `bigcode/starcoderbase-7b`).
4. **Built/generated locally:**
   - `build/kaizen.sif` — Apptainer container image
   - `build/apptainer/kaizen` — Compiled `kaizen` binary
   - `src/kernel-extraction/build/kernel-extractor` — Compiled standalone kernel extractor binary
   - `.env` — Host/container environment configuration file (see Step 4.1)

---

## 3. Build the Container and Binaries

### Step 3.1: Download AFL++ and Build the Apptainer Image (`src/apptainer/build-apptainer.sh`)

1. Download or clone AFL++ into `resources/AFLplusplus/` (relative to the repository root):
   ```bash
   mkdir -p resources
   git clone --branch v4.33c https://github.com/AFLplusplus/AFLplusplus.git resources/AFLplusplus
   ```
2. Edit **`src/apptainer/kaizen-apptainer.def`**:
   - Line 5: change `apptainer/requirements.txt /apptainer/requirements.txt` to:
     ```text
     src/apptainer/requirements.txt /apptainer/requirements.txt
     ```
   - Line 6: verify `resources/AFLplusplus /apptainer/AFLplusplus` matches where you downloaded AFL++.
   - *(Optional)* If your GPU requires a newer CUDA toolkit (e.g., Blackwell `sm_100`/`sm_120`, which is newer than CUDA 12.0.1 supports), update `From: nvidia/cuda:12.0.1-devel-ubuntu22.04` on line 2 and adjust `src/apptainer/requirements.txt` if needed.
3. Edit **`src/apptainer/build-apptainer.sh`**:
   - Line 7: change `cd "$(dirname "$0")/.."` to `cd "$(dirname "$0")/../.."` so it resolves to the repository root.
   - Line 10: change `apptainer/kaizen-apptainer.def` to `src/apptainer/kaizen-apptainer.def`.
   - **If `--fakeroot` fails on your system** (e.g., user namespaces disabled or `/etc/subuid` not configured), change line 10 to use `sudo`:
     ```bash
     sudo apptainer build build/kaizen.sif src/apptainer/kaizen-apptainer.def
     ```
     (Or build `build/kaizen.sif` on another machine with `--fakeroot`/`root` and copy it to `build/kaizen.sif`.)
4. Run from the repository root:
   ```bash
   src/apptainer/build-apptainer.sh
   ```

**What `src/apptainer/kaizen-apptainer.def` installs:**
- Base image: `nvidia/cuda:12.0.1-devel-ubuntu22.04`
- Build tools, CMake, Python 3, pip, gfortran
- LLVM/Clang 18 (`llvm-18-dev`, `libclang-18-dev`, `libomp-18-dev`, `libc++-18-dev`)
- AFL++ compiled from `resources/AFLplusplus` using `LLVM_CONFIG=llvm-config-18`
- Python packages from `src/apptainer/requirements.txt`

---

### Step 3.2: Build the Kaizen Binary (`src/apptainer/build-kaizen-fuzzer.sh`)

1. Edit **`src/apptainer/CMakeLists.txt`**:
   - Line 24: change `file(GLOB_RECURSE SOURCES "../src/*.cpp")` to:
     ```cmake
     file(GLOB_RECURSE SOURCES "../kernel-fuzzing/*.cpp")
     ```
     *(This ensures CMake compiles only the Kaizen fuzzer sources and does not pick up `src/differential-testing/differential-driver.cpp`, which defines its own `main()`.)*
2. Edit **`src/apptainer/build-kaizen-fuzzer.sh`**:
   - Line 14: change `cd "$(dirname "$0")/.."` to `cd "$(dirname "$0")/../.."` (repository root).
   - Line 37: change `cmake -S apptainer -B build/apptainer` to:
     ```bash
     CC=clang-18 CXX=clang++-18 cmake -S src/apptainer -B build/apptainer
     ```
   - **If `--fakeroot` fails during `apptainer exec`** (common on HPC compute/login nodes without fakeroot mapping), remove `--fakeroot` from line 26. (`build/apptainer` is owned by your user on the host.)
3. Compile `build/apptainer/kaizen` inside `build/kaizen.sif`:
   ```bash
   bash src/apptainer/build-kaizen-fuzzer.sh
   # Or to force a clean rebuild:
   bash src/apptainer/build-kaizen-fuzzer.sh --clean
   ```

---

### Step 3.3: Build the Standalone Kernel Extractor (`src/kernel-extraction/build.sh`)

If you plan to run Stage 2 (`src/kernel-extraction/`) to extract CUDA kernels from mutated `source-queue/` files into `extracted-kernels-*.txt`:

1. Edit **`src/kernel-extraction/build.sh`**:
   - Line 29: change `cd /work/resources/mutated-kernel-extraction` to:
     ```bash
     cd /work/src/kernel-extraction && \
     ```
2. Edit **`src/kernel-extraction/run-kernel-extractor.sh`**:
   - Line 59: change `"/work/resources/mutated-kernel-extraction/build/kernel-extractor"` to:
     ```bash
     "/work/src/kernel-extraction/build/kernel-extractor" \
     ```
3. Build `src/kernel-extraction/build/kernel-extractor`:
   ```bash
   bash src/kernel-extraction/build.sh
   ```

---

## 4. Configure `.env`, `kaizen.cfg`, and HeCBench Benchmarks

### Step 4.1: Create `.env` and Update `kaizen.cfg`

1. Edit **`src/kernel-fuzzing/config/kaizen.cfg`** (lines 10–26) so every path points to `src/kernel-fuzzing/` inside `/app`:
   ```ini
   comment_options_file = /app/src/kernel-fuzzing/resources/mutation/comment-options.txt
   mutation_options_config = /app/src/kernel-fuzzing/config/mutation-options.cfg
   false_conditions_file = /app/src/kernel-fuzzing/resources/mutation/false-conditions.txt
   dead_code_statements_file = /app/src/kernel-fuzzing/resources/mutation/dead-code-statements.txt
   dead_for_loops_file = /app/src/kernel-fuzzing/resources/mutation/dead-for-loops.txt
   dead_while_loops_file = /app/src/kernel-fuzzing/resources/mutation/dead-while-loops.txt
   variable_types_file = /app/src/kernel-fuzzing/resources/mutation/variable-types.txt
   variable_values_file = /app/src/kernel-fuzzing/resources/mutation/variable-values.txt
   variable_names_file = /app/src/kernel-fuzzing/resources/mutation/variable-names.txt
   ```
2. Create a **`.env`** file in the repository root:
   ```dotenv
   # --- Host System Paths ---
   HECBENCH_SRC_DIR="/absolute/path/to/ready-bench/src"
   HOST_PROJECT_DIR=$(pwd)
   APPTAINER_IMAGE="build/kaizen.sif"

   # --- Configuration Files (container paths) ---
   KAIZEN_CONFIG_FILE="/app/src/kernel-fuzzing/config/kaizen.cfg"

   # --- Container Mounts & Binary Paths ---
   CONTAINER_APP_DIR="/app"
   CONTAINER_DATA_DIR="/data"
   KERNEL_EXTRACTOR_BIN="/app/build/apptainer/kaizen"

   # --- Required by WorkflowOrchestrator::validateEnvironment() (unused while inline LLM stages are disabled) ---
   TRANSLATOR_SCRIPT="unused"
   LLM_INFERENCE_SCRIPT_DIR="/unused"
   LLM_INFERENCE_INPUT_FILE="/unused"
   LLM_INFERENCE_OUTPUT_FILE="/unused"
   PEFT_CHECKPOINT_DIR=""
   ```

Notes:
- `EnvironmentConfig` (`src/kernel-fuzzing/kaizen/environment-config.cpp`) expands `$(pwd)` and `${VAR}`, and `src/kernel-fuzzing/run.sh` sources `.env` on the host before passing paths into `apptainer exec`.
- `WorkflowOrchestrator::validateEnvironment()` (`src/kernel-fuzzing/kaizen/workflow-orchestrator.cpp`, lines 88–103) checks that `TRANSLATOR_SCRIPT` and `LLM_INFERENCE_*` are non-empty strings even though inline LLM translation Stages 3/4/3.5 (lines 595–619) are commented out; keeping the placeholder values above satisfies validation without requiring extra directories.
- `src/kernel-fuzzing/kernel-identification/CompilationCommandGenerator.cpp` and `src/kernel-extraction/main.cpp` hardcode `/usr/bin/clang++-18` and `--cuda-path=/usr/local/cuda`, which match `build/kaizen.sif`.

---

### Step 4.2: Download and Prepare HeCBench (`ready-bench/src/`)

1. Clone or download HeCBench from <https://github.com/zjin-lcf/HeCBench/tree/HeCBench-OMP> so its benchmark directory is at `<base-dir>/example-bench/src/`.
2. Run `src/kernel-fuzzing/benchmark-setup/setup-benchmark-fuzzing.py` to generate `ready-bench/src/` with AFL++ Makefiles (`CXX = afl-clang-lto++`, `libcuda.a` and `libomp.a` targets) and `#ifdef LIBCUDA` / `#ifdef LIBOMP` namespace wrappers (`cuda_program::cuda_main`, `omp_program::omp_main`):
   ```bash
   python3 src/kernel-fuzzing/benchmark-setup/setup-benchmark-fuzzing.py --base-dir /path/to/dir-containing-example-bench
   ```
   Point `HECBENCH_SRC_DIR` in `.env` to the generated `ready-bench/src` directory.
3. For benchmarks whose CUDA entry source file is not named `main.cu` or whose `__global__` kernels are defined in separate headers, run `src/kernel-fuzzing/benchmark-setup/setup-benchmark-for-kernel-fuzzing.py` (reads `HECBENCH_SRC_DIR` from `.env` in the repo root) to rename the main source to `main.cu` and inline kernel headers:
   ```bash
   python3 src/kernel-fuzzing/benchmark-setup/setup-benchmark-for-kernel-fuzzing.py <benchmark_name>
   ```
4. **Verify `ready-bench/src/` Makefiles for your GPU and compiler:**
   - **CUDA GPU Architecture (`ARCH`):** `setup-benchmark-fuzzing.py` sets `ARCH = sm_86` in `*-cuda/Makefile`. Update `ARCH` if your GPU requires a different compute capability.
   - **OpenMP Makefiles:** Verify each target `<benchmark>-omp/Makefile` uses `CXX = afl-clang-lto++`, `-fopenmp=libomp`, and defines a `libomp.a` target (some upstream HeCBench OpenMP Makefiles, e.g., ones originally using `icpx`, may require manual alignment with standard `afl-clang-lto++` Makefiles). Verify `make clean && make`, `make run`, and `make libomp.a` inside `build/kaizen.sif` before fuzzing.

---

## 5. Required Modifications & Usage by Pipeline Stage

The framework runs as a 6-stage pipeline. Wrapper scripts here operate on individual runs or directories so you can integrate them into your own automation or cluster job scheduler.

### Stage 1: Kernel Mutation Loop (`src/kernel-fuzzing/run.sh`)

**Modifications required in `src/kernel-fuzzing/run.sh`:**
- Line 19: change `cd "$(dirname "$0")/.."` to `cd "$(dirname "$0")/../.."` (repository root).
- Lines 55–56: **remove** the two `--bind` flags (for `/llm_inference` and `/llm_inference/checkpoint-400`). Those source directories do not exist and inline LLM stages are commented out in `src/kernel-fuzzing/kaizen/workflow-orchestrator.cpp`; leaving those `--bind` flags will cause Apptainer to abort on startup.

**Usage (from repository root):**
```bash
bash src/kernel-fuzzing/run.sh <benchmark_name> <runtime_seconds> [seed] [gpu_id]
# Example:
bash src/kernel-fuzzing/run.sh cross 3600 10292 0
```
Outputs are created under `kaizen-output/<benchmark>-seed<seed>-<timestamp>/` with mutated CUDA variants in `source-queue/main-*.cu`. *(In the paper's evaluation, runs used 3600s across seeds `10292`, `11687`, `68351`, `90221`, and `91313`.)*

---

### Stage 2: Mutated Kernel Extraction (`src/kernel-extraction/`)

Extract `__global__` CUDA kernels from `source-queue/` into `extracted-kernels/extracted-kernels-<id>.txt`:

```bash
# Single directory containing source code and Makefile:
bash src/kernel-extraction/run-kernel-extractor.sh <input-folder> <output-file>

# Batch across all run folders in a base directory inside the repo root:
bash src/kernel-extraction/run-mutated-kernel-extractor.sh kaizen-output
```

**Important path requirement:** `src/kernel-extraction/run-kernel-extractor.sh` mounts `$PROJECT_ROOT` to `/work` and computes container paths by stripping `$PROJECT_ROOT`. Therefore, `<input-folder>`, `<output-file>`, and `<base-directory>` **must live inside the repository root** (e.g., `kaizen-output/`), or you must add an extra `--bind` flag in `run-kernel-extractor.sh`.

---

### Stage 3: Batch LLM Translation (`src/translation/`)

Run on the **host Python environment** (with `vllm`, `torch`, `python-dotenv`, and `pandas` installed, outside `build/kaizen.sif`):
- `src/translation/batch-translate-codellama.py`
- `src/translation/batch-translate-hpccoder.py`
- `src/translation/batch-translate-starcoder.py`
- `src/translation/batch-translate-starcoder-parallel.py`

```bash
# Reassemble the split CodeLlama LoRA weights first (once after cloning):
cat src/translation/checkpoints/codellama-13b/adapter_model.safetensors.part_* \
  > src/translation/checkpoints/codellama-13b/adapter_model.safetensors

source venv/bin/activate
python3 src/translation/batch-translate-codellama.py kaizen-output src/translation/checkpoints/codellama-13b
python3 src/translation/batch-translate-hpccoder.py kaizen-output src/translation/checkpoints/hpc-coder-v2-6.7b
python3 src/translation/batch-translate-starcoder.py kaizen-output src/translation/checkpoints/starcoderbase-7b
```

**What you need to set up / adjust:**
1. **Model weights & Hugging Face auth:** Each script reads `base_model_name_or_path` from `<checkpoint>/adapter_config.json` (in `src/translation/checkpoints/`). Ensure Hugging Face access (`export HF_TOKEN=...` or `huggingface-cli login`) or local base model weights (`meta-llama/CodeLlama-13b-hf`, `hpcgroup/hpc-coder-v2-6.7b`, `bigcode/starcoderbase-7b`), and sufficient GPU VRAM. (For `codellama-13b`, make sure you reassemble `adapter_model.safetensors` from `adapter_model.safetensors.part_*` as shown above.)
2. **GPU count & context length:**
   - `src/translation/batch-translate-starcoder.py` sets `tensor_parallel_size=2` (line 164) — change to `1` for single-GPU systems (or use `batch-translate-starcoder-parallel.py`, which uses `tensor_parallel_size=1`).
   - `src/translation/batch-translate-hpccoder.py` and `src/translation/batch-translate-vllm-hpccoder.py` set `max_model_len=53872` (line 164) — reduce if your GPU VRAM is smaller.

This produces `translated-kernels-<llm>/translated-kernels-<llm>-<id>.json` inside each benchmark run folder.

---

### Stage 4: Cross-File Kernel Replacement (`src/kernel-replacement/`)

Insert translated OpenMP kernels back into the benchmark source files:

```bash
python3 src/kernel-replacement/replace-benchmark-kernels.py kaizen-output --llm codellama
```

- Reads `HECBENCH_SRC_DIR` from `.env` at the repository root and maps kernel names using `src/kernel-replacement/omp-benchmarks.config`.
- Temporarily modifies files in `HECBENCH_SRC_DIR` (creating `.bak` backups) and restores them at the end. **Do not run multiple instances concurrently against the same `HECBENCH_SRC_DIR`.** If interrupted, restore any remaining `*.bak` files in `HECBENCH_SRC_DIR` before running again.
- Saves complete translated OpenMP files to `translated-outputs-<llm>/main-<id>.cpp` and copies the benchmark build files into `working-copy-omp/`.

---

### Stage 5: Variant Compilation Testing

Inside `build/kaizen.sif`, test compilation of each variant produced in Stage 4:
- **OpenMP compilation check (and static library for Stage 6):** Copy `translated-outputs-<llm>/main-<id>.cpp` into `working-copy-omp/main.cpp` (or an isolated per-worker copy of `<benchmark>-omp`) and run:
  - `make clean && make` inside `build/kaizen.sif` to verify compilation and collect compiler diagnostics.
  - `make clean && make libomp.a` inside `build/kaizen.sif` to build the static library linked by the differential driver in Stage 6.
- **CUDA binary (for Stage 6):** Copy `source-queue/main-<id>.cu` into `working-copy/main.cu` (from `<benchmark>-cuda`) and run `make clean && make` inside `build/kaizen.sif` (`--nv` if executing tests) to produce the `cuda_program` executable.

---

### Stage 6: Differential Testing / AFL++ Fuzzing (`src/differential-testing/`)

`src/differential-testing/` provides the C++ differential driver (`differential-driver.cpp`), 14 benchmark custom AFL++ Python mutators (`AFL-custom-fuzzers/`), seed inputs (`seeds/`), and the per-variant fuzzing helpers (`launch-afl.sh`, `run_afl_fuzzing.py`).

**Required modifications before compiling and running:**
1. **`src/differential-testing/differential-driver.cpp` (line 18):**
   Update `LOG_PATH` to a valid, writable directory inside/bound into the container:
   ```cpp
   constexpr const char* LOG_PATH = "/absolute/path/to/logs/differential-testing";
   ```
   Ensure that directory exists on the host (and is bound into the container when running inside Apptainer), otherwise per-input logs and diffs (`diff -u omp.log cuda.log >> ...`) will fail to open that path.
2. **`src/differential-testing/launch-afl.sh`:**
   - Lines 7–11: remove or update the `module load ...` lines (`slurm`, `gcc/10.3`, `cuda/12.8`, `mpich/4.0h_gcc-10`, `python/3.12`) to match your environment (they will fail on standalone machines without Environment Modules/Lmod).
   - Line 23: replace the hardcoded path to `afldriver-separate-cuda.cpp` with the path to `src/differential-testing/differential-driver.cpp` in your checkout:
     ```bash
     afl-clang-lto++ -std=c++14 /path/to/Kaizen-Replication-Package/src/differential-testing/differential-driver.cpp libomp.a -o driver_new -stdlib=libc++ -fopenmp=libomp -L/usr/local/cuda/lib64 -lcudart_static -lrt -ldl -pthread
     ```
   - *(Optional)* Adjust `DEFAULT_SEEDS`, `--timeout`, or `--exec-timeout` as needed.
3. **`src/differential-testing/run_afl_fuzzing.py`:**
   - Line 187: update the hardcoded default `--fuzzer-dir` path to point to `src/differential-testing/AFL-custom-fuzzers` in your checkout (or pass `--fuzzer-dir /path/to/Kaizen-Replication-Package/src/differential-testing/AFL-custom-fuzzers` explicitly).

**Per-variant execution:**
In a variant test directory containing `cuda_program`, `libomp.a`, `launch-afl.sh`, `run_afl_fuzzing.py`, and `inputs/` (populated with `seed*.txt` from `src/differential-testing/seeds/<benchmark>/` when using `--shared-inputs`), run inside `build/kaizen.sif` with GPU passthrough (`--nv`):

```bash
apptainer exec --nv --bind /path/to/repo:/path/to/repo build/kaizen.sif \
  bash -c "cd '/path/to/test-dir' && bash launch-afl.sh custom_fuzzer_<benchmark>"
```

*(Note: `afl-clang-lto++` builds `driver_new` using `differential-driver.cpp` and `libomp.a`, and `differential-driver.cpp` invokes `./cuda_program` in the current working directory. Use `-D STANDALONE` when compiling `differential-driver.cpp` if you want to replay a single input file directly via `./driver <input_file>`.)*

---

## 6. Quick-Reference Table of Files Requiring Modification

| File | Hardcoded / Site-Specific Item | What to Change |
| :--- | :--- | :--- |
| `.env` *(create)* | Does not exist on clean clone | Create in repo root (see Step 4.1); set `HECBENCH_SRC_DIR`, `HOST_PROJECT_DIR`, and `KAIZEN_CONFIG_FILE` |
| `src/apptainer/kaizen-apptainer.def` | `apptainer/requirements.txt`, `resources/AFLplusplus` | Update line 5 to `src/apptainer/requirements.txt`; supply AFL++ at `resources/AFLplusplus` (or update line 6) |
| `src/apptainer/CMakeLists.txt` | `file(GLOB_RECURSE SOURCES "../src/*.cpp")` | Change line 24 to `../kernel-fuzzing/*.cpp` |
| `src/apptainer/build-apptainer.sh` | `# !/bin/bash`, `..`, `apptainer/kaizen-apptainer.def`, `--fakeroot` | Change line 7 to `../..`, line 10 to `src/apptainer/kaizen-apptainer.def`; run with `bash`; use `sudo` if `--fakeroot` fails |
| `src/apptainer/build-kaizen-fuzzer.sh` | `..`, `--fakeroot`, `-S apptainer` | Change line 14 to `../..`, line 37 to `-S src/apptainer`; remove `--fakeroot` on line 26 if unsupported |
| `src/kernel-fuzzing/config/kaizen.cfg` | `/app/config/...` and `/app/resources/mutation/...` | Point lines 10–26 to `/app/src/kernel-fuzzing/config/...` and `/app/src/kernel-fuzzing/resources/mutation/...` |
| `src/kernel-fuzzing/run.sh` | `..`, non-existent LLM/checkpoint `--bind` lines | Change line 19 to `../..`; remove lines 55–56 |
| `src/kernel-extraction/build.sh` | `/work/resources/mutated-kernel-extraction` | Change line 29 to `/work/src/kernel-extraction` |
| `src/kernel-extraction/run-kernel-extractor.sh` | `/work/resources/mutated-kernel-extraction/build/...` | Change line 59 to `/work/src/kernel-extraction/build/kernel-extractor` |
| `src/translation/batch-translate-*.py` | Needs `vllm`, `python-dotenv`, LoRA checkpoints, HF models | Install host venv deps; adjust `tensor_parallel_size` (`batch-translate-starcoder.py` line 164) and `max_model_len` (`batch-translate-hpccoder.py` line 164) |
| `src/differential-testing/differential-driver.cpp` | Hardcoded `LOG_PATH` (line 18) | Set to a valid writable log directory bound into the container |
| `src/differential-testing/launch-afl.sh` | `module load ...`, hardcoded driver C++ path (line 23) | Remove/update `module load` lines; point line 23 to `src/differential-testing/differential-driver.cpp` |
| `src/differential-testing/run_afl_fuzzing.py` | Hardcoded default `--fuzzer-dir` (line 187) | Update default or pass `--fuzzer-dir` pointing to `src/differential-testing/AFL-custom-fuzzers` |
| `ready-bench/src/*-cuda/Makefile` & `*-omp/Makefile` | `ARCH = sm_86`, upstream OpenMP flags | Update `ARCH` for your GPU; verify OpenMP Makefiles use `afl-clang-lto++`, `-fopenmp=libomp`, and `libomp.a` |

---

## 7. Common Pitfalls & Troubleshooting

- **Apptainer `--fakeroot` errors:** If `apptainer build --fakeroot` or `apptainer exec --fakeroot` fails with `user namespaces` or `subuid` errors, build `build/kaizen.sif` with `sudo apptainer build` (or on another machine with fakeroot enabled) and remove `--fakeroot` from `src/apptainer/build-kaizen-fuzzer.sh`.
- **Bind-mount errors (`source path doesn't exist`):** Check that every `--bind src:dst` source path exists on your host (especially removing lines 55–56 in `src/kernel-fuzzing/run.sh`).
- **`KAIZEN_CONFIG_FILE` / static init exit:** `KernelMutator.cpp` reads `KAIZEN_CONFIG_FILE` during static initialization; ensure `.env` sets `KAIZEN_CONFIG_FILE` to the container path of `kaizen.cfg` and that all paths inside `kaizen.cfg` exist inside the container.
- **`module: command not found`:** Remove `module load ...` lines from `src/differential-testing/launch-afl.sh` on systems without Environment Modules/Lmod.
- **AFL++ host warnings/aborts:** If running `afl-fuzz` manually outside `run_afl_fuzzing.py`, export `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1` and `AFL_SKIP_CPUFREQ=1` if you do not have root access to configure `/proc/sys/kernel/core_pattern` or the CPU frequency governor.
- **Container path resolution in `run-kernel-extractor.sh`:** Keep `kaizen-output/` inside the repository root so `/work` bind-mounting maps paths cleanly.
