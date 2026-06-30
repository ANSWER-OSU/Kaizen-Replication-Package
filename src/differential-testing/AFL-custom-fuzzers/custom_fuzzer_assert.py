"""
Dummy AFL++ fuzzer for assert benchmark.

assert takes no runtime arguments - all parameters (Nblocks, Nthreads, N)
are hardcoded. The benchmark intentionally triggers CUDA assertion failures
as part of its test. This dummy fuzzer always returns an empty input to
satisfy the pipeline.

The program being tested: ./driver (no arguments)
"""

import random
import platform

node = ""

def init(seed):
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Dummy fuzzer (assert) initialized on {node} with seed {seed}")
    print("assert takes no runtime arguments - outputs empty input")

def deinit():
    pass

def fuzz(buf, add_buf, max_size):
    return bytearray(b"")

def fuzz_count(buf):
    return 1