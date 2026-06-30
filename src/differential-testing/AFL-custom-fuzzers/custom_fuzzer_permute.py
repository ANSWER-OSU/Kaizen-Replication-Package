"""
Custom AFL++ fuzzer for Permute (QKV data permutation) benchmark testing.

This fuzzer is designed to test the correctness of QKV tensor permutation
by generating valid integer inputs for batch size and repeat parameters.

The program being tested: ./driver <batch_size> <repeat>
- batch_size (B): number of sequences in the batch (must be positive)
- repeat: number of benchmark repetitions (must be positive)

Key insights:
- T=1024, C=768, NH=12 are hardcoded constants
- Memory: B * T * C * 3 * sizeof(float) = B * 1024 * 768 * 3 * 4 bytes
        = B * 9,437,184 bytes (~9MB per batch item)
- Work: B * T * C * repeat operations
- block_sizes {32,64,128,256,512} are swept internally
- Timeout protection: total work = B * 1024 * 768 * repeat

Mutation strategy focuses on:
1. Various batch sizes (memory driver)
2. Small to medium repeat values
3. Timeout protection via B * repeat budget

Setup:
1. Rename this file to: custom_fuzzer_permute.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_permute"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# T, C are hardcoded in the binary
T = 1024
C = 768

# Conservative limits for 6000ms timeout
MAX_SAFE_B      = 256
MAX_SAFE_REPEAT = 10000
# Budget for B * repeat (T*C amplification is fixed at 786432)
MAX_B_REPEAT    = 50000  # 50K

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer (permute) initialized on {node} with seed {seed}")
    print(f"Using limits: B<={MAX_SAFE_B}, repeat<={MAX_SAFE_REPEAT}")
    print(f"Max B*repeat budget: {MAX_B_REPEAT}")

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

def mutate_B(original_value):
    """Apply targeted mutations for batch size parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        # Very small batch sizes (edge cases)
        new_value = random.choice([1, 2, 4])

    elif mutation_type <= 4:
        # Small batch sizes
        new_value = random.choice([4, 8, 16, 32])

    elif mutation_type <= 6:
        # Medium batch sizes (typical)
        new_value = random.choice([32, 64, 128])

    elif mutation_type <= 8:
        # Larger batch sizes
        new_value = random.choice([128, 192, 256])

    elif mutation_type == 9:
        # Power of 2
        exp = random.randint(0, 8)  # 1 to 256
        new_value = 2 ** exp

    else:
        # Add/subtract
        offset = random.choice([1, 2, 4, 8, 16])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    return max(1, min(new_value, MAX_SAFE_B))

def mutate_repeat(original_value, B):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 3:
        new_value = random.choice([1, 2, 5, 10])

    elif mutation_type <= 5:
        new_value = random.choice([10, 50, 100, 500])

    elif mutation_type <= 7:
        new_value = random.choice([1000, 2000, 5000])

    elif mutation_type == 8:
        exp = random.randint(0, 13)
        new_value = 2 ** exp

    elif mutation_type == 9:
        offset = random.choice([1, 10, 50, 100])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    else:
        factor = random.choice([2, 5, 10])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)

    new_value = max(1, new_value)

    # Timeout safety based on B * repeat budget
    if B * new_value > MAX_B_REPEAT:
        safe_repeat = max(1, MAX_B_REPEAT // B)
        new_value = min(new_value, safe_repeat)

    # Additional throttling for large batch sizes
    if B >= 128:
        new_value = min(new_value, 100)
    elif B >= 64:
        new_value = min(new_value, 500)
    elif B >= 32:
        new_value = min(new_value, 1000)

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
        numbers = [8, 100]

    while len(numbers) < 2:
        if len(numbers) == 0:
            numbers.append(8)    # B
        else:
            numbers.append(100)  # repeat

    numbers[0] = max(1, numbers[0])
    numbers[1] = max(1, numbers[1])

    mutation_choice = random.random()

    if mutation_choice < 0.50:
        # Mutate B (50%) — main memory/work driver
        numbers[0] = mutate_B(numbers[0])
        numbers[1] = mutate_repeat(numbers[1], numbers[0])

    else:
        # Mutate repeat (50%)
        numbers[1] = mutate_repeat(numbers[1], numbers[0])

    # Sometimes mutate both (20% chance)
    if random.random() < 0.2:
        numbers[0] = mutate_B(numbers[0])
        numbers[1] = mutate_repeat(numbers[1], numbers[0])

    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_B))

    if numbers[0] * numbers[1] > MAX_B_REPEAT:
        safe_repeat = max(1, MAX_B_REPEAT // numbers[0])
        numbers[1] = min(numbers[1], safe_repeat)

    numbers[1] = max(1, min(numbers[1], MAX_SAFE_REPEAT))

    numbers = numbers[:2]

    mutated_string = " ".join(str(num) for num in numbers)

    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]

    return bytearray(mutated_string.encode('utf-8'))

def fuzz_count(buf):
    """Return number of mutations to perform"""
    return 1