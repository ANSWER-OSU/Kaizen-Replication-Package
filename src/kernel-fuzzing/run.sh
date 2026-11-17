#!/bin/bash
set -e # Exit immediately if a command fails

# Check if required arguments are provided
if [ $# -lt 2 ]; then
  echo "Usage: $0 <benchmark_name> <runtime_seconds> [seed] [gpu_id]"
  echo "Example: $0 cross 300"
  echo "Example: $0 adam 600 1234"
  echo "Example: $0 adam 600 1234 0"
  exit 1
fi

BENCHMARK_NAME="$1"
RUNTIME_SECONDS="$2"
SEED="$3"  # Optional third argument
GPU_ID="$4"  # Optional fourth argument for GPU selection

# Change to project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo ">>> Running mutation loop for benchmark: ${BENCHMARK_NAME}"
echo ">>> Runtime: ${RUNTIME_SECONDS} seconds"
if [ -n "$SEED" ]; then
  echo ">>> Using seed: ${SEED}"
fi
if [ -n "$GPU_ID" ]; then
  echo ">>> Using GPU: ${GPU_ID}"
fi
echo ">>> Project root: ${PROJECT_ROOT}"

# --- Configuration ---
# Source environment variables from .env file
if [ -f .env ]; then
  set -a # Automatically export all variables
  source .env
  set +a # Stop automatically exporting
else
  echo "Error: .env file not found."
  exit 1
fi

# Run the mutation loop using the C++ program in apptainer
echo ">>> Starting mutation loop via apptainer..."

# Build the command arguments
KAIZEN_ARGS="$RUNTIME_SECONDS --benchmark=$BENCHMARK_NAME"
if [ -n "$SEED" ]; then
  KAIZEN_ARGS="$KAIZEN_ARGS --seed=$SEED"
fi

apptainer exec --nv \
  --bind ${HOST_PROJECT_DIR}:${CONTAINER_APP_DIR} \
  --bind ${HECBENCH_SRC_DIR}:/hecbench_src \
  --bind ${HOST_PROJECT_DIR}/resources/chatport-test/HeCBench_LLM-main:/llm_inference \
  --bind ${HOST_PROJECT_DIR}/resources/chatport-test/checkpoint-400:/llm_inference/checkpoint-400 \
  --env HECBENCH_SRC_DIR="/hecbench_src" \
  --env HOST_PROJECT_DIR="${CONTAINER_APP_DIR}" \
  --env TRANSLATOR_SCRIPT="${CONTAINER_APP_DIR}/${TRANSLATOR_SCRIPT}" \
  --env APPTAINER_IMAGE="${CONTAINER_APP_DIR}/${APPTAINER_IMAGE}" \
  --env KAIZEN_CONFIG_FILE="${KAIZEN_CONFIG_FILE}" \
  --env CONTAINER_APP_DIR="${CONTAINER_APP_DIR}" \
  --env CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR}" \
  --env KERNEL_EXTRACTOR_BIN="${KERNEL_EXTRACTOR_BIN}" \
  --env LLM_INFERENCE_SCRIPT_DIR="/llm_inference/scripts" \
  --env LLM_INFERENCE_INPUT_FILE="/llm_inference/dataset/hecbench_test.json" \
  --env LLM_INFERENCE_OUTPUT_FILE="/llm_inference/scripts/hecbench-test-omp-ft-CodeLlama-13b-hf-sample1.json" \
  --env PEFT_CHECKPOINT_DIR="/llm_inference/checkpoint-400" \
  ${GPU_ID:+--env CUDA_VISIBLE_DEVICES="$GPU_ID"} \
  "$APPTAINER_IMAGE" \
  "$KERNEL_EXTRACTOR_BIN" \
    $KAIZEN_ARGS

echo ">>> Mutation loop complete."