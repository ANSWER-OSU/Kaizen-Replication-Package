"""
Custom AFL++ fuzzer for ChannelSum benchmark testing.

This fuzzer is designed to test the correctness of channel sum (NCHW and NHWC)
calculations by generating valid integer inputs for width, height, and repeat parameters.

The program being tested: ./driver <width> <height> <repeat>
- width: spatial width of the input tensor (must be positive)
- height: spatial height of the input tensor (must be positive)
- repeat: number of kernel repetitions (must be positive)

Key insights:
- N and C are swept internally: N in {1,4,16,64}, C in {32,128,512}
- Max internal multiplier: N=64, C=512 => 32768x amplification on W*H
- Memory: N * C * W * H * sizeof(int) bytes
- Total work scales as N * C * W * H * repeat (N,C internal)
- Timeout protection: keep W * H * repeat within budget

Mutation strategy focuses on:
1. Various width values (spatial dimension)
2. Various height values (spatial dimension)
3. Small to medium repeat values
4. Timeout protection via W * H * repeat budget

Setup:
1. Rename this file to: custom_fuzzer_channelSum.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_channelSum"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Conservative limits for 6000ms timeout
# N and C are swept internally up to 64 * 512 = 32768
# Total work = 32768 * W * H * repeat
MAX_SAFE_W      = 4096
MAX_SAFE_H      = 4096
MAX_SAFE_REPEAT = 10000
# Budget for W * H * repeat (accounting for internal N*C amplification)
MAX_WH_REPEAT   = 10000000  # 10M

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer (channelSum) initialized on {node} with seed {seed}")
    print(f"Using limits: W<={MAX_SAFE_W}, H<={MAX_SAFE_H}, repeat<={MAX_SAFE_REPEAT}")
    print(f"Max W*H*repeat budget: {MAX_WH_REPEAT}")

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

def calculate_wh_repeat(W, H, repeat):
    """Calculate effective work proxy: W * H * repeat"""
    try:
        return W * H * repeat
    except:
        return MAX_WH_REPEAT + 1

def mutate_W(original_value):
    """Apply targeted mutations for width parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        # Very small widths (edge cases)
        new_value = random.choice([1, 2, 4, 7, 8, 16])

    elif mutation_type <= 4:
        # Small widths
        new_value = random.choice([32, 56, 64, 112, 128])

    elif mutation_type <= 6:
        # Medium widths (common image sizes)
        new_value = random.choice([224, 256, 320, 512])

    elif mutation_type <= 8:
        # Larger widths
        new_value = random.choice([1024, 2048, 4096])

    elif mutation_type == 9:
        # Power of 2
        exp = random.randint(0, 12)  # 1 to 4096
        new_value = 2 ** exp

    else:
        # Add/subtract
        offset = random.choice([1, 8, 32, 64, 128])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    return max(1, min(new_value, MAX_SAFE_W))

def mutate_H(original_value):
    """Apply targeted mutations for height parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        new_value = random.choice([1, 2, 4, 7, 8, 16])

    elif mutation_type <= 4:
        new_value = random.choice([32, 56, 64, 112, 128])

    elif mutation_type <= 6:
        new_value = random.choice([224, 256, 320, 512])

    elif mutation_type <= 8:
        new_value = random.choice([1024, 2048, 4096])

    elif mutation_type == 9:
        exp = random.randint(0, 12)
        new_value = 2 ** exp

    else:
        offset = random.choice([1, 8, 32, 64, 128])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    return max(1, min(new_value, MAX_SAFE_H))

def mutate_repeat(original_value, W, H):
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

    # Timeout safety
    wh_repeat = calculate_wh_repeat(W, H, new_value)
    if wh_repeat > MAX_WH_REPEAT:
        wh = W * H
        if wh > 0:
            safe_repeat = max(1, MAX_WH_REPEAT // wh)
            new_value = min(new_value, safe_repeat)

    # Additional throttling for large spatial dims
    wh = W * H
    if wh > 1000000:
        new_value = min(new_value, 5)
    elif wh > 100000:
        new_value = min(new_value, 50)
    elif wh > 10000:
        new_value = min(new_value, 500)

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
        numbers = [224, 224, 100]

    while len(numbers) < 3:
        if len(numbers) == 0:
            numbers.append(224)   # W
        elif len(numbers) == 1:
            numbers.append(224)   # H
        else:
            numbers.append(100)   # repeat

    numbers[0] = max(1, numbers[0])
    numbers[1] = max(1, numbers[1])
    numbers[2] = max(1, numbers[2])

    mutation_choice = random.random()

    if mutation_choice < 0.35:
        # Mutate W (35%)
        numbers[0] = mutate_W(numbers[0])
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])

    elif mutation_choice < 0.65:
        # Mutate H (30%)
        numbers[1] = mutate_H(numbers[1])
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])

    else:
        # Mutate repeat (35%)
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])

    # Sometimes mutate all parameters (20% chance)
    if random.random() < 0.2:
        numbers[0] = mutate_W(numbers[0])
        numbers[1] = mutate_H(numbers[1])
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])

    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_W))
    numbers[1] = max(1, min(numbers[1], MAX_SAFE_H))

    wh_repeat = calculate_wh_repeat(numbers[0], numbers[1], numbers[2])
    if wh_repeat > MAX_WH_REPEAT:
        wh = numbers[0] * numbers[1]
        if wh > 0:
            safe_repeat = max(1, MAX_WH_REPEAT // wh)
            numbers[2] = min(numbers[2], safe_repeat)

    numbers[2] = max(1, min(numbers[2], MAX_SAFE_REPEAT))

    numbers = numbers[:3]

    mutated_string = " ".join(str(num) for num in numbers)

    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]

    return bytearray(mutated_string.encode('utf-8'))

def fuzz_count(buf):
    """Return number of mutations to perform"""
    return 1