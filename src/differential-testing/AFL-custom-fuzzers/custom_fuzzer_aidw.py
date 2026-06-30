"""
Custom AFL++ fuzzer for AIDW (Adaptive Inverse Distance Weighting) interpolation testing.

This fuzzer is designed to test the correctness of AIDW interpolation calculations
by generating valid integer inputs for numk, check, and iterations parameters.

The program being tested: ./driver <numk> <check> <iterations>
- numk: number of points in thousands (e.g., 10 = 10K points, must be positive)
- check: enable verification (0 or 1, where 1 enables correctness checking)
- iterations: number of kernel repetitions (must be positive)

Key insights:
- dnum = inum = numk * 1024 (total points)
- Memory allocation: multiple arrays of size dnum (dx, dy, dz, avg_dist, ix, iy, iz)
- Computation complexity: O(dnum²) per iteration (nested loops)
- Total work: dnum² * iterations = (numk*1024)² * iterations
- EXTREMELY expensive for large numk!

Mutation strategy focuses on:
1. Small numk values (1-100) for reasonable execution times
2. Check parameter (0 or 1) to test with/without verification
3. Small iterations (1-10) due to O(n²) complexity
4. Aggressive timeout protection due to quadratic complexity

Setup:
1. Rename this file to: custom_fuzzer_aidw.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_aidw"
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

# VERY conservative limits due to O(n²) complexity
# Total work = (numk * 1024)² * iterations
# Example: numk=10 → 10K points → 100M operations per iteration
# Example: numk=100 → 100K points → 10B operations per iteration
MAX_SAFE_NUMK = 100              # 100K points maximum
MAX_SAFE_ITERATIONS = 20         # Low iterations due to O(n²)
MAX_TOTAL_OPS = 500000000000     # 500B operations max (numk²*1024²*iterations)

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer initialized on {node} with seed {seed}")
    print(f"Using limits: numk<={MAX_SAFE_NUMK}, iterations<={MAX_SAFE_ITERATIONS}")
    print(f"Max total ops: {MAX_TOTAL_OPS} (WARNING: O(n^2) complexity!)")

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

def calculate_total_ops(numk, iterations):
    """
    Calculate total operations: (numk * 1024)² * iterations
    This is QUADRATIC in dnum!
    """
    try:
        dnum = numk * 1024
        ops_per_iter = dnum * dnum  # O(n²) nested loop
        total_ops = ops_per_iter * iterations
        return total_ops
    except:
        return MAX_TOTAL_OPS + 1

def mutate_numk(original_value):
    """Apply targeted mutations for numk parameter (number of points in K)"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 3:
        # Very small values (fast execution, good for testing)
        new_value = random.choice([1, 2, 3, 5, 10])
    
    elif mutation_type <= 5:
        # Small values (10K-50K points)
        new_value = random.choice([10, 20, 30, 50])
    
    elif mutation_type <= 7:
        # Medium values (50K-100K points) - getting expensive
        new_value = random.choice([50, 75, 100])
    
    elif mutation_type <= 8:
        # Power of 2 values (good for cache/memory testing)
        exp = random.randint(0, 6)  # 1 to 64
        new_value = 2 ** exp
    
    elif mutation_type == 9:
        # Add/subtract small amounts
        offset = random.choice([1, 5, 10])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Multiply/divide by small factors
        factor = random.choice([2, 5])
        if random.choice([True, False]) and original_value < MAX_SAFE_NUMK // factor:
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    # Ensure positive and within bounds
    new_value = max(1, min(new_value, MAX_SAFE_NUMK))
    return new_value

def mutate_check(original_value):
    """
    Apply targeted mutations for check parameter (0 or 1)
    0 = no verification (faster)
    1 = with verification (slower but tests correctness)
    """
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 6:
        # 60% of the time: enable verification (we want to catch bugs!)
        new_value = 1
    elif mutation_type <= 9:
        # 30% of the time: disable verification (faster fuzzing)
        new_value = 0
    else:
        # 10% of the time: flip current value
        new_value = 1 - original_value
    
    # Ensure it's 0 or 1
    new_value = 1 if new_value != 0 else 0
    return new_value

def mutate_iterations(original_value, numk):
    """Apply targeted mutations for iterations parameter"""
    mutation_type = random.randint(1, 10)
    
    # Calculate workload
    dnum = numk * 1024
    ops_per_iter = dnum * dnum
    
    if mutation_type <= 5:
        # Most of the time: very small iteration counts
        new_value = random.choice([1, 1, 2, 3, 5])
    
    elif mutation_type <= 7:
        # Sometimes: medium iteration counts
        new_value = random.choice([5, 10, 20])
    
    elif mutation_type == 8:
        # Power of 2 values
        exp = random.randint(0, 4)  # 1 to 16
        new_value = 2 ** exp
    
    elif mutation_type == 9:
        # Add/subtract
        offset = random.choice([1, 2, 5])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Multiply/divide
        factor = random.choice([2, 5])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    # Ensure positive
    new_value = max(1, new_value)
    
    # AGGRESSIVE timeout safety based on O(n²) complexity
    total_ops = calculate_total_ops(numk, new_value)
    
    if total_ops > MAX_TOTAL_OPS:
        # Calculate safe iterations
        safe_iterations = max(1, MAX_TOTAL_OPS // ops_per_iter)
        new_value = min(new_value, safe_iterations)
    
    # Additional workload-based throttling
    # For large numk, force very small iterations
    if numk >= 75:
        new_value = min(new_value, 3)
    elif numk >= 50:
        new_value = min(new_value, 5)
    elif numk >= 30:
        new_value = min(new_value, 10)
    elif numk >= 20:
        new_value = min(new_value, 15)
    
    # Apply bounds
    new_value = min(new_value, MAX_SAFE_ITERATIONS)
    
    return new_value

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
    
    # If no valid input, start with safe defaults
    if not numbers:
        numbers = [10, 1, 5]  # numk=10 (10K points), check=1, iterations=5
    
    # Ensure we have at least 3 parameters
    while len(numbers) < 3:
        if len(numbers) == 0:
            numbers.append(10)   # numk
        elif len(numbers) == 1:
            numbers.append(1)    # check (enable verification)
        else:
            numbers.append(5)    # iterations
    
    # Ensure numk and iterations are positive, check is 0 or 1
    numbers[0] = max(1, numbers[0])
    numbers[1] = 1 if numbers[1] != 0 else 0
    numbers[2] = max(1, numbers[2])
    
    # Decide which parameter to mutate
    mutation_choice = random.random()
    
    if mutation_choice < 0.4:
        # Mutate numk (40% probability) - most important parameter
        numbers[0] = mutate_numk(numbers[0])
        # MUST adjust iterations after changing numk
        numbers[2] = mutate_iterations(numbers[2], numbers[0])
    
    elif mutation_choice < 0.6:
        # Mutate check (20% probability)
        numbers[1] = mutate_check(numbers[1])
    
    else:
        # Mutate iterations (40% probability)
        numbers[2] = mutate_iterations(numbers[2], numbers[0])
    
    # Sometimes mutate multiple parameters (15% chance)
    if random.random() < 0.15:
        numbers[0] = mutate_numk(numbers[0])
        numbers[1] = mutate_check(numbers[1])
        numbers[2] = mutate_iterations(numbers[2], numbers[0])
    
    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_NUMK))
    numbers[1] = 1 if numbers[1] != 0 else 0
    
    # Re-check iterations with final numk value
    total_ops = calculate_total_ops(numbers[0], numbers[2])
    if total_ops > MAX_TOTAL_OPS:
        dnum = numbers[0] * 1024
        ops_per_iter = dnum * dnum
        safe_iterations = max(1, MAX_TOTAL_OPS // ops_per_iter)
        numbers[2] = min(numbers[2], safe_iterations)
    
    numbers[2] = max(1, min(numbers[2], MAX_SAFE_ITERATIONS))
    
    # Keep only first three parameters
    numbers = numbers[:3]
    
    # Convert back to string format
    mutated_string = " ".join(str(num) for num in numbers)
    
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
