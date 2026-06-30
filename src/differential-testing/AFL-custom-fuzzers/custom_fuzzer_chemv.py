"""
Dummy AFL++ fuzzer for chemv benchmark.

chemv takes no runtime arguments - all parameters (N=370, REPEAT=1000,
alpha, beta) are hardcoded compile-time constants. This dummy fuzzer
always returns an empty input to satisfy the pipeline.

The program being tested: ./driver (no arguments)
"""

import random
import platform

node = ""

def init(seed):
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Dummy fuzzer (chemv) initialized on {node} with seed {seed}")
    print("chemv takes no runtime arguments - outputs empty input")

def deinit():
    pass

def fuzz(buf, add_buf, max_size):
    return bytearray(b"")

def fuzz_count(buf):
    return 1