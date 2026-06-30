"""
Custom AFL++ fuzzer for Attention (scaled dot-product) benchmark testing.

This fuzzer is designed to test the correctness of attention kernel calculations
by generating valid integer inputs for n, d, impl, and repeat parameters.

The program being tested: ./driver <n> <d> <impl> <repeat>
- n: number of rows / sequence length (must be positive)
- d: number of columns / embedding dimension (must be positive)
- impl: implementation variant (0=naive, 1=block reduce, 2=warp reduce, 3=mixed reduce)
- repeat: number of kernel repetitions (must be positive)

Key insights:
- Memory: 2 * n * d floats (key + value) + d floats (query)
- Computation: n * d * repeat operations
- impl is a discrete choice in {0, 1, 2, 3} — not a continuous parameter
- Timeout protection: total work = n * d * repeat

Mutation strategy focuses on:
1. Various n values (sequence lengths)
2. Various d values (embedding dimensions)
3. All 4 impl variants (rotated through)
4. Small to medium repeat values
5. Timeout protection via total work budget

Setup:
1. Rename this file to: custom_fuzzer_attention.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_attention"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Conservative limits for 6000ms timeout
# Total work = n * d * repeat
MAX_SAFE_N      = 131072   # 128K sequence length
MAX_SAFE_D      = 16384    # 16K embedding dim
MAX_SAFE_REPEAT = 10000
MAX_TOTAL_WORK  = 1000000000  # 1B operations

VALID_IMPLS = [0, 1, 2, 3]

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer (attention) initialized on {node} with seed {seed}")
    print(f"Using limits: n<={MAX_SAFE_N}, d<={MAX_SAFE_D}, repeat<={MAX_SAFE_REPEAT}")
    print(f"Max total work: {MAX_TOTAL_WORK} (n * d * repeat)")

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

def calculate_total_work(n, d, repeat):
    """Calculate total work: n * d * repeat"""
    try:
        return n * d * repeat
    except:
        return MAX_TOTAL_WORK + 1

def mutate_n(original_value):
    """Apply targeted mutations for n (sequence length) parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        # Very small sequence lengths (edge cases)
        new_value = random.choice([1, 2, 4, 8, 16, 32])

    elif mutation_type <= 4:
        # Small sequence lengths
        new_value = random.choice([64, 128, 256, 512, 1024])

    elif mutation_type <= 6:
        # Medium sequence lengths (typical)
        new_value = random.choice([2048, 4096, 8192, 16384])

    elif mutation_type <= 8:
        # Large sequence lengths (from Makefile)
        new_value = random.choice([32768, 65536, 131072])

    elif mutation_type == 9:
        # Power of 2
        exp = random.randint(0, 17)  # 1 to 131072
        new_value = 2 ** exp

    else:
        # Add/subtract
        offset = random.choice([1, 64, 256, 1024, 4096])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    return max(1, min(new_value, MAX_SAFE_N))

def mutate_d(original_value):
    """Apply targeted mutations for d (embedding dimension) parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        # Very small dimensions (edge cases)
        new_value = random.choice([1, 2, 4, 8, 16, 32])

    elif mutation_type <= 4:
        # Small dimensions
        new_value = random.choice([64, 128, 256, 512])

    elif mutation_type <= 6:
        # Medium dimensions (typical embedding sizes)
        new_value = random.choice([768, 1024, 2048, 4096])

    elif mutation_type <= 8:
        # Large dimensions (from Makefile)
        new_value = random.choice([8192, 16384])

    elif mutation_type == 9:
        # Power of 2
        exp = random.randint(0, 14)  # 1 to 16384
        new_value = 2 ** exp

    else:
        # Add/subtract
        offset = random.choice([1, 32, 128, 512, 1024])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    return max(1, min(new_value, MAX_SAFE_D))

def mutate_impl(original_value):
    """Rotate through valid implementation variants"""
    return random.choice(VALID_IMPLS)

def mutate_repeat(original_value, n, d):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 3:
        # Very small repeat counts
        new_value = random.choice([1, 2, 5, 10])

    elif mutation_type <= 5:
        # Small repeat counts
        new_value = random.choice([10, 50, 100, 500])

    elif mutation_type <= 7:
        # Medium repeat counts
        new_value = random.choice([1000, 2000, 5000])

    elif mutation_type == 8:
        # Power of 2
        exp = random.randint(0, 13)  # 1 to 8192
        new_value = 2 ** exp

    elif mutation_type == 9:
        # Add/subtract
        offset = random.choice([1, 10, 100, 500])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    else:
        # Multiply/divide
        factor = random.choice([2, 5, 10])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)

    new_value = max(1, new_value)

    # Timeout safety
    total_work = calculate_total_work(n, d, new_value)
    if total_work > MAX_TOTAL_WORK:
        nd = n * d
        if nd > 0:
            safe_repeat = max(1, MAX_TOTAL_WORK // nd)
            new_value = min(new_value, safe_repeat)

    # Additional workload-based throttling
    nd = n * d
    if nd > 100000000:
        new_value = min(new_value, 5)
    elif nd > 10000000:
        new_value = min(new_value, 20)
    elif nd > 1000000:
        new_value = min(new_value, 100)

    return max(1, min(new_value, MAX_SAFE_REPEAT))

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

    # Safe defaults matching Makefile run target
    if not numbers:
        numbers = [65536, 2048, 0, 1000]

    while len(numbers) < 4:
        if len(numbers) == 0:
            numbers.append(65536)  # n
        elif len(numbers) == 1:
            numbers.append(2048)   # d
        elif len(numbers) == 2:
            numbers.append(0)      # impl
        else:
            numbers.append(1000)   # repeat

    # Enforce constraints
    numbers[0] = max(1, numbers[0])
    numbers[1] = max(1, numbers[1])
    numbers[2] = numbers[2] % 4   # clamp impl to {0,1,2,3}
    numbers[3] = max(1, numbers[3])

    mutation_choice = random.random()

    if mutation_choice < 0.30:
        # Mutate n (30%)
        numbers[0] = mutate_n(numbers[0])
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])

    elif mutation_choice < 0.55:
        # Mutate d (25%)
        numbers[1] = mutate_d(numbers[1])
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])

    elif mutation_choice < 0.70:
        # Mutate impl (15%)
        numbers[2] = mutate_impl(numbers[2])

    else:
        # Mutate repeat (30%)
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])

    # Sometimes mutate multiple parameters (20% chance)
    if random.random() < 0.2:
        numbers[0] = mutate_n(numbers[0])
        numbers[1] = mutate_d(numbers[1])
        numbers[2] = mutate_impl(numbers[2])
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])

    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_N))
    numbers[1] = max(1, min(numbers[1], MAX_SAFE_D))
    numbers[2] = numbers[2] % 4
    numbers[3] = max(1, min(numbers[3], MAX_SAFE_REPEAT))

    total_work = calculate_total_work(numbers[0], numbers[1], numbers[3])
    if total_work > MAX_TOTAL_WORK:
        nd = numbers[0] * numbers[1]
        if nd > 0:
            safe_repeat = max(1, MAX_TOTAL_WORK // nd)
            numbers[3] = min(numbers[3], safe_repeat)

    numbers = numbers[:4]

    mutated_string = " ".join(str(num) for num in numbers)

    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]

    return bytearray(mutated_string.encode('utf-8'))

def fuzz_count(buf):
    """Return number of mutations to perform"""
    return 1