"""
Custom AFL++ fuzzer for Burger's equation (2D Navier-Stokes) testing.

This fuzzer is designed to test the correctness of 2D Burger's equation
calculations by generating valid integer inputs for dim_x, dim_y, and nt parameters.

The program being tested: ./driver <dim_x> <dim_y> <nt>
- dim_x: number of grid points in the x axis (must be >= 2)
- dim_y: number of grid points in the y axis (must be >= 2)
- nt: number of time steps (must be positive)

Key insights:
- Grid size: dim_x * dim_y elements
- Memory: 8 * dim_x * dim_y * 4 arrays of doubles
- Computation: dim_x * dim_y * nt operations
- Boundary conditions index 0 and dim-1, so both dims must be >= 2

Mutation strategy focuses on:
1. Various dim_x values (grid x resolution)
2. Various dim_y values (grid y resolution)
3. Small to medium nt (time steps)
4. Timeout protection: total work = dim_x * dim_y * nt

Setup:
1. Rename this file to: custom_fuzzer_burger.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_burger"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Configuration
MIN_DIM = 2  # Both dims must be >= 2 due to boundary conditions

# Conservative limits for 6000ms timeout
# Total work = dim_x * dim_y * nt
MAX_SAFE_DIM_X = 10000
MAX_SAFE_DIM_Y = 10000
MAX_SAFE_NT    = 1000
MAX_TOTAL_WORK = 100000000  # 100M operations

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer (burger) initialized on {node} with seed {seed}")
    print(f"Using limits: dim_x<={MAX_SAFE_DIM_X}, dim_y<={MAX_SAFE_DIM_Y}, nt<={MAX_SAFE_NT}")
    print(f"Max total work: {MAX_TOTAL_WORK} (dim_x * dim_y * nt)")

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

def calculate_total_work(dim_x, dim_y, nt):
    """Calculate total work: dim_x * dim_y * nt"""
    try:
        return dim_x * dim_y * nt
    except:
        return MAX_TOTAL_WORK + 1

def mutate_dim_x(original_value):
    """Apply targeted mutations for dim_x parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        # Minimum/tiny grids (edge cases)
        new_value = random.choice([2, 3, 4, 5, 10])

    elif mutation_type <= 4:
        # Small grids
        new_value = random.choice([16, 32, 64, 100, 128])

    elif mutation_type <= 6:
        # Medium grids (typical)
        new_value = random.choice([256, 512, 1000, 1024, 2000])

    elif mutation_type <= 8:
        # Larger grids
        new_value = random.choice([4096, 5000, 8000, 8200, 10000])

    elif mutation_type == 9:
        # Power of 2
        exp = random.randint(1, 13)  # 2 to 8192
        new_value = 2 ** exp

    else:
        # Add/subtract
        offset = random.choice([1, 10, 100, 500, 1000])
        new_value = max(MIN_DIM, original_value + random.choice([-offset, offset]))

    return max(MIN_DIM, min(new_value, MAX_SAFE_DIM_X))

def mutate_dim_y(original_value):
    """Apply targeted mutations for dim_y parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        new_value = random.choice([2, 3, 4, 5, 10])

    elif mutation_type <= 4:
        new_value = random.choice([16, 32, 64, 100, 128])

    elif mutation_type <= 6:
        new_value = random.choice([256, 512, 1000, 1024, 2000])

    elif mutation_type <= 8:
        new_value = random.choice([4096, 5000, 8000, 8100, 10000])

    elif mutation_type == 9:
        exp = random.randint(1, 13)
        new_value = 2 ** exp

    else:
        offset = random.choice([1, 10, 100, 500, 1000])
        new_value = max(MIN_DIM, original_value + random.choice([-offset, offset]))

    return max(MIN_DIM, min(new_value, MAX_SAFE_DIM_Y))

def mutate_nt(original_value, dim_x, dim_y):
    """Apply targeted mutations for nt parameter"""
    mutation_type = random.randint(1, 10)

    grid_size = dim_x * dim_y

    if mutation_type <= 3:
        # Very small step counts
        new_value = random.choice([1, 2, 5, 10])

    elif mutation_type <= 5:
        # Small step counts
        new_value = random.choice([10, 20, 50, 100])

    elif mutation_type <= 7:
        # Medium step counts
        new_value = random.choice([100, 200, 500, 1000])

    elif mutation_type == 8:
        # Power of 2
        exp = random.randint(0, 9)  # 1 to 512
        new_value = 2 ** exp

    elif mutation_type == 9:
        # Add/subtract
        offset = random.choice([1, 5, 10, 50, 100])
        new_value = max(1, original_value + random.choice([-offset, offset]))

    else:
        # Multiply/divide
        factor = random.choice([2, 5, 10])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)

    new_value = max(1, new_value)

    # Timeout safety based on total work
    total_work = calculate_total_work(dim_x, dim_y, new_value)
    if total_work > MAX_TOTAL_WORK:
        safe_nt = max(1, MAX_TOTAL_WORK // grid_size)
        new_value = min(new_value, safe_nt)

    # Additional workload-based throttling
    if grid_size > 5000000:
        new_value = min(new_value, 5)
    elif grid_size > 1000000:
        new_value = min(new_value, 20)
    elif grid_size > 500000:
        new_value = min(new_value, 50)
    elif grid_size > 100000:
        new_value = min(new_value, 200)

    return max(1, min(new_value, MAX_SAFE_NT))

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
        numbers = [8200, 8100, 100]

    while len(numbers) < 3:
        if len(numbers) == 0:
            numbers.append(8200)   # dim_x
        elif len(numbers) == 1:
            numbers.append(8100)   # dim_y
        else:
            numbers.append(100)    # nt

    # Enforce minimums
    numbers[0] = max(MIN_DIM, numbers[0])
    numbers[1] = max(MIN_DIM, numbers[1])
    numbers[2] = max(1, numbers[2])

    mutation_choice = random.random()

    if mutation_choice < 0.35:
        # Mutate dim_x (35%)
        numbers[0] = mutate_dim_x(numbers[0])
        numbers[2] = mutate_nt(numbers[2], numbers[0], numbers[1])

    elif mutation_choice < 0.65:
        # Mutate dim_y (30%)
        numbers[1] = mutate_dim_y(numbers[1])
        numbers[2] = mutate_nt(numbers[2], numbers[0], numbers[1])

    else:
        # Mutate nt (35%)
        numbers[2] = mutate_nt(numbers[2], numbers[0], numbers[1])

    # Sometimes mutate all parameters (20% chance)
    if random.random() < 0.2:
        numbers[0] = mutate_dim_x(numbers[0])
        numbers[1] = mutate_dim_y(numbers[1])
        numbers[2] = mutate_nt(numbers[2], numbers[0], numbers[1])

    # Final safety checks
    numbers[0] = max(MIN_DIM, min(numbers[0], MAX_SAFE_DIM_X))
    numbers[1] = max(MIN_DIM, min(numbers[1], MAX_SAFE_DIM_Y))

    total_work = calculate_total_work(numbers[0], numbers[1], numbers[2])
    if total_work > MAX_TOTAL_WORK:
        grid_size = numbers[0] * numbers[1]
        safe_nt = max(1, MAX_TOTAL_WORK // grid_size)
        numbers[2] = min(numbers[2], safe_nt)

    numbers[2] = max(1, min(numbers[2], MAX_SAFE_NT))

    numbers = numbers[:3]

    mutated_string = " ".join(str(num) for num in numbers)

    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]

    return bytearray(mutated_string.encode('utf-8'))

def fuzz_count(buf):
    """Return number of mutations to perform"""
    return 1