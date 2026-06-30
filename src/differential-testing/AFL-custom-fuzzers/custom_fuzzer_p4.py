"""
Custom AFL++ fuzzer for P4 (PointPillars postprocess) benchmark testing.

This fuzzer is designed to test the correctness of the postprocess kernel
by generating valid integer inputs for the repeat parameter.

The program being tested: ./driver <repeat>
- repeat: number of times to run the postprocess kernel (must be positive)

Key insights:
- All grid/anchor/feature parameters are compile-time constants from params.h
- repeat only controls how many times the kernel is launched
- Work per repeat is fixed by compile-time constants
- Timeout protection: total work scales linearly with repeat

Mutation strategy focuses on:
1. Small repeat values (fast execution, edge cases)
2. Medium repeat values (typical benchmark range)
3. Large repeat values (stress testing)

Setup:
1. Rename this file to: custom_fuzzer_p4.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_p4"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Conservative limits for 6000ms timeout
# Work per repeat is fixed; repeat scales linearly with total time
MAX_SAFE_REPEAT = 100000   # generous upper bound
MIN_REPEAT = 1

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer (p4) initialized on {node} with seed {seed}")
    print(f"Using limits: repeat<={MAX_SAFE_REPEAT}")

def deinit():
    """Cleanup on fuzzer shutdown"""
    pass

def parse_input(buf_str):
    """Parse space-separated integers from input string"""
    try:
        parts = buf_str.strip().split()
        numbers = []
        for part in parts:
            try:
                num = int(part)
                numbers.append(num)
            except ValueError:
                pass
        return numbers
    except:
        return []

def mutate_repeat(original_value):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        # Very small values (edge cases)
        new_value = random.choice([1, 2, 5, 10])

    elif mutation_type <= 4:
        # Small values (fast execution)
        new_value = random.choice([10, 50, 100, 200, 500])

    elif mutation_type <= 6:
        # Medium values (typical benchmark range)
        new_value = random.choice([1000, 2000, 5000, 10000])

    elif mutation_type <= 8:
        # Larger values (stress testing)
        new_value = random.choice([20000, 50000, 100000])

    elif mutation_type == 9:
        # Power of 2
        exp = random.randint(0, 16)  # 1 to 65536
        new_value = 2 ** exp

    else:
        # Add/subtract
        offset = random.choice([1, 10, 100, 1000, 5000])
        new_value = max(MIN_REPEAT, original_value + random.choice([-offset, offset]))

    return max(MIN_REPEAT, min(new_value, MAX_SAFE_REPEAT))

def fuzz(buf, add_buf, max_size):
    """
    Main fuzzing function called by AFL++

    Args:
        buf: input buffer (bytes)
        add_buf: additional buffer for splicing (not used here)
        max_size: maximum size of output

    Returns:
        bytearray: mutated input
    """
    try:
        buf_str = buf.decode('utf-8', errors='ignore').strip()
    except:
        buf_str = str(buf).strip()

    numbers = parse_input(buf_str)

    # Safe default matching Makefile run target
    if not numbers:
        numbers = [10000]

    # Ensure we have at least 1 parameter
    if len(numbers) == 0:
        numbers = [10000]

    # Enforce minimum
    numbers[0] = max(MIN_REPEAT, numbers[0])

    # Mutate repeat
    numbers[0] = mutate_repeat(numbers[0])

    # Final safety check
    numbers[0] = max(MIN_REPEAT, min(numbers[0], MAX_SAFE_REPEAT))

    # Keep only first parameter
    numbers = numbers[:1]

    mutated_string = str(numbers[0])

    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]

    return bytearray(mutated_string.encode('utf-8'))

def fuzz_count(buf):
    """Return number of mutations to perform"""
    return 1