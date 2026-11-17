# the purpose of this script is to launch custom fuzzer developed using AFL 
# on a single pair of CUDA and LLM-translated OpenMP program using 5 seeds, 
# 30 min timeout per seed and execution time per input of 60 sec 

#!/bin/bash

module load slurm
module load gcc/10.3
module load cuda/12.8
module load mpich/4.0h_gcc-10
module load python/3.12

CUSTOM_FUZZER=$1 # custom_fuzzer_aidw

command -v python3 >/dev/null || {
  echo "ERROR: python3 not found in PATH"
  echo "PATH=$PATH"
  exit 1
}

DEFAULT_SEEDS=(7632 1324 8742 9088 3739)

afl-clang-lto++ -std=c++14 /nfs/stak/users/motwanim/software/LLM-Code-Porting-Testing/resources/differential-testing/afldriver-separate-cuda.cpp libomp.a -o driver_new -stdlib=libc++ -fopenmp=libomp -L/usr/local/cuda/lib64 -lcudart_static -lrt -ldl -pthread

# Loop over seeds
for seed in "${DEFAULT_SEEDS[@]}"; do
  echo "Starting fuzzing with seed ${seed}"

  # Run python in background, redirect output to a per-seed log
  python3 run_afl_fuzzing.py \
    ./driver_new $CUSTOM_FUZZER \
    --seeds "${seed}" \
    --timeout 1800 \
    --exec-timeout 60000 \
    --shared-inputs \
    > "fuzz_seed_${seed}.log" 2>&1 &

  # Capture PID of background job
  pid=$!

  # Wait for this seed to finish before starting next
  wait "${pid}"

  echo "Finished fuzzing with seed ${seed}"
done
