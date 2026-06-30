"""
Custom AFL++ fuzzer for Winograd convolution benchmark testing.

This fuzzer is designed to test the correctness of Winograd 2D convolution
by generating valid integer inputs for the three required parameters.

The program being tested: ./driver <map_size> <dim_local_work_group_x> <dim_local_work_group_y>
- map_size: size of the 2D input map (must be positive integer, typically > 2)
- dim_local_work_group_x: local work group size in X dimension (typically power of 2, <= 1024)
- dim_local_work_group_y: local work group size in Y dimension (typically power of 2, <= 1024)

Mutation strategy focuses on:
1. Valid map sizes (must be > 2 for the algorithm to work)
2. Work group sizes that are powers of 2 (common in GPU/parallel programming)
3. Work group sizes that fit within typical GPU limits (total threads <= 1024)
4. Edge cases around small and large map sizes
5. Timeout-aware mutations (avoids combinations that exceed 6000ms)

Setup:
1. Rename this file to: custom_fuzzer_winograd.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_winograd"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1

Run inside Apptainer with GPU access:
apptainer shell --nv kaizen.sif
"""

import random
import platform

# Configuration
POSITIVE_ONLY = True  # numk and iterations must be positive

# Timeout budget: 6000ms total
# The Winograd algorithm complexity is roughly O(map_size^2 * iterations)
# where iterations = 101 (sweep from 0 to 100)
MAX_SAFE_MAP_SIZE = 2048       # Large enough for thorough testing
MIN_MAP_SIZE = 3               # Must be > 2 for algorithm to work
MAX_WORK_GROUP_SIZE = 1024     # Typical GPU limit for total threads per block
MIN_WORK_GROUP_SIZE = 1        # Minimum work group size

# Common power-of-2 work group sizes used in GPU programming
COMMON_WORK_GROUP_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer initialized on {node} with seed {seed}")

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


def mutate_map_size(original_value):
    """Apply targeted mutations for map_size parameter"""
    mutation_type = random.randint(1, 10)

    if mutation_type <= 2:
        # Small map sizes (edge cases, fast execution)
        new_value = random.choice([3, 4, 5, 6, 7, 8, 10, 12, 15, 16, 31, 32, 63, 64])

    elif mutation_type <= 4:
        # Medium map sizes (good balance)
        new_value = random.choice([100, 128, 200, 256, 300, 400, 512, 600, 700, 800])

    elif mutation_type <= 6:
        # Large map sizes (thorough testing)
        new_value = random.choice([900, 1000, 1024, 1200, 1500, 1800, 2000, 2048])

    elif mutation_type <= 7:
        # Power of 2 values (common edge cases)
        exp = random.randint(2, 11)  # 4 to 2048
        new_value = 2 ** exp

    elif mutation_type <= 8:
        # Power of 2 minus/plus small offset (boundary testing)
        base = random.choice([64, 128, 256, 512, 1024, 2048])
        offset = random.choice([-2, -1, 1, 2])
        new_value = base + offset

    elif mutation_type == 9:
        # Add/subtract for boundary exploration
        offset = random.choice([1, 2, 5, 10, 50, 100])
        new_value = original_value + random.choice([-offset, offset])

    else:
        # Multiply/divide by small factors
        factor = random.choice([2, 3, 4, 5])
        if random.choice([True, False]) and original_value < MAX_SAFE_MAP_SIZE // factor:
            new_value = original_value * factor
        else:
            new_value = max(MIN_MAP_SIZE, original_value // factor)

    # Ensure within safe bounds
    new_value = max(MIN_MAP_SIZE, min(new_value, MAX_SAFE_MAP_SIZE))
    return new_value

def mutate_work_group_size(original_value, is_x_dimension=True):
    """
    Apply targeted mutations for work group size parameters.
    
    Args:
        original_value: current work group size
        is_x_dimension: whether this is the X dimension (unused for now, but may be useful)
    """
    mutation_type = random.randint(1, 10)

    if mutation_type <= 5:
        # Use common power-of-2 sizes (most important for GPU efficiency)
        new_value = random.choice(COMMON_WORK_GROUP_SIZES)

    elif mutation_type <= 7:
        # Small odd numbers (edge cases for work group alignment)
        new_value = random.choice([1, 3, 5, 7, 9, 11, 15, 17, 31, 33])

    elif mutation_type == 8:
        # Multiply/divide by 2 (common scaling pattern)
        if random.choice([True, False]) and original_value < MAX_WORK_GROUP_SIZE // 2:
            new_value = original_value * 2
        else:
            new_value = max(MIN_WORK_GROUP_SIZE, original_value // 2)

    elif mutation_type == 9:
        # Add/subtract small amounts
        offset = random.choice([1, 2, 4, 8, 16])
        new_value = original_value + random.choice([-offset, offset])

    else:
        # Random value in reasonable range
        new_value = random.randint(MIN_WORK_GROUP_SIZE, 256)

    # Ensure within bounds
    new_value = max(MIN_WORK_GROUP_SIZE, min(new_value, MAX_WORK_GROUP_SIZE))
    return new_value

def validate_and_adjust_params(map_size, work_group_x, work_group_y):
    """
    Validate and adjust parameters to ensure they're safe and reasonable.
    
    Returns:
        tuple: (adjusted_map_size, adjusted_work_group_x, adjusted_work_group_y)
    """
    # Ensure map_size is valid
    map_size = max(MIN_MAP_SIZE, min(map_size, MAX_SAFE_MAP_SIZE))
    
    # Ensure work group sizes are valid
    work_group_x = max(MIN_WORK_GROUP_SIZE, min(work_group_x, MAX_WORK_GROUP_SIZE))
    work_group_y = max(MIN_WORK_GROUP_SIZE, min(work_group_y, MAX_WORK_GROUP_SIZE))
    
    # GPU constraint: total threads per block typically <= 1024
    total_threads = work_group_x * work_group_y
    if total_threads > MAX_WORK_GROUP_SIZE:
        # Scale down proportionally
        scale_factor = (MAX_WORK_GROUP_SIZE / total_threads) ** 0.5
        work_group_x = max(1, int(work_group_x * scale_factor))
        work_group_y = max(1, int(work_group_y * scale_factor))
    
    # Timeout safety: if map_size is very large, ensure it won't timeout
    # The algorithm runs 101 iterations with O(map_size^2) work per iteration
    estimated_work = map_size * map_size * 101
    if estimated_work > 500000000:  # Rough threshold for 6-second timeout
        # Scale down map_size
        map_size = min(map_size, 1500)
    
    return map_size, work_group_x, work_group_y

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
    # Decode input buffer to string
    try:
        buf_str = buf.decode('utf-8', errors='ignore').strip()
    except:
        buf_str = str(buf).strip()

    # Parse input numbers
    numbers = parse_input(buf_str)

    # If no valid input, start with reasonable defaults
    if not numbers:
        numbers = [128, 16, 16]  # Reasonable defaults for Winograd

    # Ensure we have exactly 3 parameters
    while len(numbers) < 3:
        numbers.append(16)  # Add default work group size
    
    # Keep only first 3 parameters
    numbers = numbers[:3]
    
    map_size, work_group_x, work_group_y = numbers

    # Decide which parameter(s) to mutate
    mutation_choice = random.randint(1, 10)
    
    if mutation_choice <= 3:
        # Mutate map_size only
        map_size = mutate_map_size(map_size)
        
    elif mutation_choice <= 6:
        # Mutate work_group_x
        work_group_x = mutate_work_group_size(work_group_x, is_x_dimension=True)
        
    elif mutation_choice <= 8:
        # Mutate work_group_y
        work_group_y = mutate_work_group_size(work_group_y, is_x_dimension=False)
        
    elif mutation_choice == 9:
        # Mutate both work group sizes (keep them related)
        if random.choice([True, False]):
            # Make them equal (square work groups)
            size = mutate_work_group_size(work_group_x)
            work_group_x = work_group_y = size
        else:
            # Mutate independently
            work_group_x = mutate_work_group_size(work_group_x, is_x_dimension=True)
            work_group_y = mutate_work_group_size(work_group_y, is_x_dimension=False)
    
    else:
        # Mutate all three parameters
        map_size = mutate_map_size(map_size)
        work_group_x = mutate_work_group_size(work_group_x, is_x_dimension=True)
        work_group_y = mutate_work_group_size(work_group_y, is_x_dimension=False)

    # Validate and adjust parameters for safety
    map_size, work_group_x, work_group_y = validate_and_adjust_params(
        map_size, work_group_x, work_group_y
    )

    # Convert back to string format
    mutated_string = f"{map_size} {work_group_x} {work_group_y}"

    # Ensure we don't exceed max_size
    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]

    # Return as bytearray
    return bytearray(mutated_string.encode('utf-8'))

def fuzz_count(buf):
    """
    Optional: Return number of mutations to perform
    We'll do just 1 mutation per call for simplicity
    """
    return 1
