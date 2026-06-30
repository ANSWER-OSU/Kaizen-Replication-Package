"""
Custom AFL++ fuzzer for ACE (Phase-field simulation).
Program: ./driver <repeat>
- repeat: number of iterations (must be positive)
"""

import random
import platform

MAX_REPEAT = 1000
node = ""

def init(seed):
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer ace initialized on {node} with seed {seed}")

def deinit():
    pass

def parse_input(buf_str):
    try:
        parts = buf_str.strip().split()
        return [int(p) for p in parts if p.lstrip('-').isdigit()]
    except:
        return []

def fuzz(buf, add_buf, max_size):
    try:
        buf_str = buf.decode('utf-8', errors='ignore').strip()
    except:
        buf_str = str(buf).strip()

    numbers = parse_input(buf_str)
    if not numbers:
        numbers = [100]

    mutation_type = random.randint(1, 6)
    if mutation_type == 1:
        numbers[0] = random.choice([1, 2, 5, 10])
    elif mutation_type == 2:
        numbers[0] = random.choice([10, 20, 50, 100])
    elif mutation_type == 3:
        numbers[0] = random.choice([100, 200, 500])
    elif mutation_type == 4:
        exp = random.randint(1, 9)
        numbers[0] = 2 ** exp
    else:
        offset = random.choice([1, 5, 10, 50])
        numbers[0] = max(1, numbers[0] + random.choice([-offset, offset]))

    numbers[0] = max(1, min(numbers[0], MAX_REPEAT))
    result = str(numbers[0])
    if len(result) > max_size:
        result = result[:max_size]
    return bytearray(result.encode('utf-8'))

def fuzz_count(buf):
    return 1