"""
Custom AFL++ fuzzer for Ising (2D Ising model Monte Carlo) benchmark testing.

This fuzzer is designed to test the correctness of 2D Ising model spin updates
by generating valid inputs for lattice dimensions and iteration counts.

The program being tested: ./driver -x <nx> -y <ny> -w <nwarmup> -n <niters>
- nx: number of lattice rows (must be positive and EVEN)
- ny: number of lattice columns (must be positive and EVEN)
- nwarmup: number of warmup iterations (must be positive)
- niters: number of trial iterations (must be positive)

Key insights:
- Both nx and ny MUST be even (program exits with error if odd)
- Memory: nx * ny / 2 * sizeof(float) for randvals + lattices
- Total work: nx * ny * (nwarmup + niters)
- Timeout protection: keep nx * ny * (nwarmup + niters) within budget

Mutation strategy focuses on:
1. Various even nx values (lattice rows)
2. Various even ny values (lattice columns)
3. Small to medium nwarmup values
4. Small to medium niters values
5. Timeout protection via total work budget

Setup:
1. Rename this file to: custom_fuzzer_ising.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_ising"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Both nx and ny must be even
MIN_DIM = 2

# Conservative limits for 6000ms timeout
# Total work = nx * ny * (nwarmup + niters)
MAX_SAFE_NX      = 10240
MAX_SAFE_NY      = 10240
MAX_SAFE_WARMUP  = 1000
MAX_SAFE_NITERS  = 1000
MAX_TOTAL_WORK   = 500000000  # 500M operations

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer (ising) initialized on {node} with seed {seed}")
    print(f"Using limits: nx<={MAX_SAFE_NX}, ny<={MAX_SAFE_NY}, warmup<={MAX_SAFE_WARMUP}, niters<={MAX_SAFE_NITERS}")
    print(f"Max total work: {MAX_TOTAL_WORK} (nx * ny * (nwarmup + niters))")

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

def make_even(value):
    """Ensure value is even"""
    if value % 2 != 0:
        value += 1
    return value

def calculate_total_work(nx, ny, nwarmup, niters):
    """Calculate total work: nx * ny * (nwarmup + niters)"""
    try:
        return nx * ny * (nwarmup + niters)
    except:
        return MAX_TOTAL_WORK + 1

def mutate_nx(original_value):
    """Apply targeted mutations for nx (lattice rows) parameter - must be even"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        # Very small lattices (edge cases)
        new_value = random.choice([2, 4, 8, 16, 32])

    elif mutation_type <= 4:
        # Small lattices
        new_value = random.choice([64, 128, 256, 512])

    elif mutation_type <= 6:
        # Medium lattices (typical)
        new_value = random.choice([1024, 2048, 4096, 5120])

    elif mutation_type <= 8:
        # Large lattices
        new_value = random.choice([6144, 8192, 10240])

    elif mutation_type == 9:
        # Power of 2
        exp = random.randint(1, 13)  # 2 to 8192
        new_value = 2 ** exp

    else:
        # Add/subtract (keep even)
        offset = random.choice([2, 4, 16, 64, 256, 1024])
        new_value = max(MIN_DIM, original_value + random.choice([-offset, offset]))

    new_value = make_even(max(MIN_DIM, min(new_value, MAX_SAFE_NX)))
    return new_value

def mutate_ny(original_value):
    """Apply targeted mutations for ny (lattice columns) parameter - must be even"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        new_value = random.choice([2, 4, 8, 16, 32])

    elif mutation_type <= 4:
        new_value = random.choice([64, 128, 256, 512])

    elif mutation_type <= 6:
        new_value = random.choice([1024, 2048, 4096, 5120])

    elif mutation_type <= 8:
        new_value = random.choice([6144, 8192, 10240])

    elif mutation_type == 9:
        exp = random.randint(1, 13)
        new_value = 2 ** exp

    else:
        offset = random.choice([2, 4, 16, 64, 256, 1024])
        new_value = max(MIN_DIM, original_value + random.choice([-offset, offset]))

    new_value = make_even(max(MIN_DIM, min(new_value, MAX_SAFE_NY)))
    return new_value

def mutate_warmup(original_value, nx, ny):
    """Apply targeted mutations for nwarmup parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 3:
        new_value = random.choice([1, 2, 5, 10])

    elif mutation_type <= 5:
        new_value = random.choice([10, 50, 100])

    elif mutation_type <= 7:
        new_value = random.choice([100, 200, 500, 1000])

    elif mutation_type == 8:
        exp = random.randint(0, 9)
        new_value = 2 ** exp

    elif mutation_type == 9:
        offset = random.choice([1, 5, 10, 50])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    else:
        factor = random.choice([2, 5, 10])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)

    new_value = max(1, min(new_value, MAX_SAFE_WARMUP))

    # Throttle based on lattice size
    lattice = nx * ny
    if lattice > 10000000:
        new_value = min(new_value, 5)
    elif lattice > 1000000:
        new_value = min(new_value, 20)
    elif lattice > 100000:
        new_value = min(new_value, 100)

    return max(1, new_value)

def mutate_niters(original_value, nx, ny, nwarmup):
    """Apply targeted mutations for niters parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 3:
        new_value = random.choice([1, 2, 5, 10])

    elif mutation_type <= 5:
        new_value = random.choice([10, 50, 100])

    elif mutation_type <= 7:
        new_value = random.choice([100, 200, 500, 1000])

    elif mutation_type == 8:
        exp = random.randint(0, 9)
        new_value = 2 ** exp

    elif mutation_type == 9:
        offset = random.choice([1, 5, 10, 50])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    else:
        factor = random.choice([2, 5, 10])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)

    new_value = max(1, new_value)

    # Timeout safety
    total_work = calculate_total_work(nx, ny, nwarmup, new_value)
    if total_work > MAX_TOTAL_WORK:
        lattice = nx * ny
        if lattice > 0:
            safe_iters = max(1, MAX_TOTAL_WORK // lattice - nwarmup)
            new_value = min(new_value, safe_iters)

    # Throttle based on lattice size
    lattice = nx * ny
    if lattice > 10000000:
        new_value = min(new_value, 5)
    elif lattice > 1000000:
        new_value = min(new_value, 20)
    elif lattice > 100000:
        new_value = min(new_value, 100)

    return max(1, min(new_value, MAX_SAFE_NITERS))

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
        numbers = [5120, 5120, 10, 1000]

    while len(numbers) < 4:
        if len(numbers) == 0:
            numbers.append(5120)   # nx
        elif len(numbers) == 1:
            numbers.append(5120)   # ny
        elif len(numbers) == 2:
            numbers.append(10)     # nwarmup
        else:
            numbers.append(1000)   # niters

    # Enforce even dimensions
    numbers[0] = make_even(max(MIN_DIM, numbers[0]))
    numbers[1] = make_even(max(MIN_DIM, numbers[1]))
    numbers[2] = max(1, numbers[2])
    numbers[3] = max(1, numbers[3])

    mutation_choice = random.random()

    if mutation_choice < 0.30:
        # Mutate nx (30%)
        numbers[0] = mutate_nx(numbers[0])
        numbers[3] = mutate_niters(numbers[3], numbers[0], numbers[1], numbers[2])

    elif mutation_choice < 0.55:
        # Mutate ny (25%)
        numbers[1] = mutate_ny(numbers[1])
        numbers[3] = mutate_niters(numbers[3], numbers[0], numbers[1], numbers[2])

    elif mutation_choice < 0.70:
        # Mutate nwarmup (15%)
        numbers[2] = mutate_warmup(numbers[2], numbers[0], numbers[1])

    else:
        # Mutate niters (30%)
        numbers[3] = mutate_niters(numbers[3], numbers[0], numbers[1], numbers[2])

    # Sometimes mutate all (20% chance)
    if random.random() < 0.2:
        numbers[0] = mutate_nx(numbers[0])
        numbers[1] = mutate_ny(numbers[1])
        numbers[2] = mutate_warmup(numbers[2], numbers[0], numbers[1])
        numbers[3] = mutate_niters(numbers[3], numbers[0], numbers[1], numbers[2])

    # Final safety checks
    numbers[0] = make_even(max(MIN_DIM, min(numbers[0], MAX_SAFE_NX)))
    numbers[1] = make_even(max(MIN_DIM, min(numbers[1], MAX_SAFE_NY)))
    numbers[2] = max(1, min(numbers[2], MAX_SAFE_WARMUP))

    total_work = calculate_total_work(numbers[0], numbers[1], numbers[2], numbers[3])
    if total_work > MAX_TOTAL_WORK:
        lattice = numbers[0] * numbers[1]
        if lattice > 0:
            safe_iters = max(1, MAX_TOTAL_WORK // lattice - numbers[2])
            numbers[3] = min(numbers[3], safe_iters)

    numbers[3] = max(1, min(numbers[3], MAX_SAFE_NITERS))
    numbers = numbers[:4]

    # Output as flag-style arguments
    mutated_string = f"-x {numbers[0]} -y {numbers[1]} -w {numbers[2]} -n {numbers[3]}"

    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]

    return bytearray(mutated_string.encode('utf-8'))

def fuzz_count(buf):
    """Return number of mutations to perform"""
    return 1