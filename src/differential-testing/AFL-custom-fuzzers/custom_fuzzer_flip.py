"""
Custom AFL++ fuzzer for flip kernel benchmark testing.

This fuzzer is designed to test the correctness of tensor flip operations
by generating valid integer inputs for num_dims, dim_size, and repeat parameters.

The program being tested: ./driver <num_dims> <dim_size> <repeat>
- num_dims: number of dimensions in the tensor (must be positive, typically 1-5)
- dim_size: size of each dimension (must be positive)
- repeat: number of benchmark repetitions (must be positive integer)

Key insights:
- Total elements: n = dim_size^num_dims (exponential growth!)
- Memory allocation: n * sizeof(scalar_t) * 2 (input + output)
- num_dims is typically small (1-5) for realistic tensors
- Large num_dims with large dim_size causes exponential memory explosion

Mutation strategy focuses on:
1. Small num_dims (1-5) which are realistic for tensors
2. Various dim_sizes (small, medium, powers of 2)
3. Edge cases: num_dims=1, dim_size=1
4. Aggressive timeout protection due to exponential complexity

Setup:
1. Rename this file to: custom_fuzzer_flip.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_flip"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform
import math

# Configuration
POSITIVE_ONLY = True  # All parameters must be positive

# Conservative limits due to EXPONENTIAL complexity: n = dim_size^num_dims
# Example: dim_size=10, num_dims=5 → n=100,000 elements
# Example: dim_size=100, num_dims=3 → n=1,000,000 elements
MAX_SAFE_NUM_DIMS = 6          # Beyond 6 dims is rarely used and risky
MAX_SAFE_DIM_SIZE = 1000       # Keep reasonable
MAX_SAFE_REPEAT = 10           # Low repeat for faster fuzzing
MAX_TOTAL_ELEMENTS = 10000000  # 10M elements maximum (n = dim_size^num_dims)

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer initialized on {node} with seed {seed}")
    print(f"Using limits: num_dims<={MAX_SAFE_NUM_DIMS}, dim_size<={MAX_SAFE_DIM_SIZE}, repeat<={MAX_SAFE_REPEAT}")
    print(f"Max total elements: {MAX_TOTAL_ELEMENTS} (dim_size^num_dims)")

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

def calculate_total_elements(num_dims, dim_size):
    """Calculate total elements: n = dim_size^num_dims"""
    try:
        # Use float to avoid integer overflow
        result = float(dim_size) ** float(num_dims)
        if result > MAX_TOTAL_ELEMENTS or result != result:  # Check for overflow or NaN
            return MAX_TOTAL_ELEMENTS + 1  # Return value that will trigger safety checks
        return int(result)
    except:
        return MAX_TOTAL_ELEMENTS + 1

def adjust_for_total_elements(num_dims, dim_size):
    """
    Adjust dim_size to keep total elements reasonable
    If dim_size^num_dims > MAX_TOTAL_ELEMENTS, reduce dim_size
    """
    total = calculate_total_elements(num_dims, dim_size)
    
    if total > MAX_TOTAL_ELEMENTS:
        # Calculate safe dim_size: dim_size = MAX_TOTAL_ELEMENTS^(1/num_dims)
        safe_dim_size = int(MAX_TOTAL_ELEMENTS ** (1.0 / num_dims))
        # Add some margin for safety
        safe_dim_size = max(1, int(safe_dim_size * 0.8))
        return safe_dim_size
    
    return dim_size

def mutate_num_dims(original_value):
    """Apply targeted mutations for num_dims parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 4:
        # Most common: 2D, 3D, 4D tensors (very common in ML)
        new_value = random.choice([2, 3, 4])
    
    elif mutation_type <= 6:
        # Edge cases: 1D and 5D
        new_value = random.choice([1, 5])
    
    elif mutation_type <= 8:
        # Test boundary: 6D (maximum safe)
        new_value = 6
    
    elif mutation_type == 9:
        # Small adjustments
        offset = random.choice([1, 2])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Random small value
        new_value = random.randint(1, 6)
    
    # Ensure bounds
    new_value = max(1, min(new_value, MAX_SAFE_NUM_DIMS))
    return new_value

def mutate_dim_size(original_value, num_dims):
    """Apply targeted mutations for dim_size parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Edge cases: very small dimensions
        new_value = random.choice([1, 2, 3, 4, 5])
    
    elif mutation_type <= 4:
        # Small dimensions (common for testing)
        new_value = random.choice([8, 10, 16, 20, 32])
    
    elif mutation_type <= 6:
        # Medium dimensions
        new_value = random.choice([50, 64, 100, 128, 200, 256])
    
    elif mutation_type <= 8:
        # Larger dimensions (be careful with num_dims)
        if num_dims <= 2:
            new_value = random.choice([500, 1000])
        elif num_dims == 3:
            new_value = random.choice([100, 200])
        else:
            new_value = random.choice([10, 20, 50])
    
    elif mutation_type == 9:
        # Add/subtract
        offset = random.choice([1, 5, 10, 50])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Power of 2 values
        exp = random.randint(0, 10)  # 1 to 1024
        new_value = 2 ** exp
    
    # Ensure positive
    new_value = max(1, new_value)
    
    # CRITICAL: Adjust for total elements constraint
    new_value = adjust_for_total_elements(num_dims, new_value)
    
    # Apply bounds
    new_value = min(new_value, MAX_SAFE_DIM_SIZE)
    
    return new_value

def mutate_repeat(original_value, num_dims, dim_size):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)
    
    # Calculate workload
    total_elements = calculate_total_elements(num_dims, dim_size)
    
    if mutation_type <= 6:
        # Most of the time: very small repeat counts
        new_value = random.choice([1, 1, 2, 3, 5])
    
    elif mutation_type <= 8:
        # Sometimes: medium repeat counts
        new_value = random.choice([10, 20])
    
    elif mutation_type == 9:
        # Rarely: small adjustments
        offset = random.choice([1, 2, 5])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Very rarely: test higher values
        new_value = random.choice([1, 5, 10])
    
    # Ensure positive
    new_value = max(1, new_value)
    
    # AGGRESSIVE timeout safety based on total elements
    if total_elements > 5000000:
        new_value = 1  # Force to 1 for very large tensors
    elif total_elements > 1000000:
        new_value = min(new_value, 2)
    elif total_elements > 500000:
        new_value = min(new_value, 3)
    elif total_elements > 100000:
        new_value = min(new_value, 5)
    
    # Apply bounds
    new_value = min(new_value, MAX_SAFE_REPEAT)
    
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
        numbers = [3, 10, 1]  # num_dims=3, dim_size=10, repeat=1 → 1000 elements
    
    # Ensure we have at least 3 parameters
    while len(numbers) < 3:
        if len(numbers) == 0:
            numbers.append(3)   # num_dims
        elif len(numbers) == 1:
            numbers.append(10)  # dim_size
        else:
            numbers.append(1)   # repeat
    
    # Ensure all values are positive
    numbers = [max(1, n) for n in numbers]
    
    # Adjust dim_size to keep total elements reasonable
    numbers[1] = adjust_for_total_elements(numbers[0], numbers[1])
    
    # Decide which parameter to mutate
    mutation_choice = random.random()
    
    if mutation_choice < 0.3:
        # Mutate num_dims (30% probability)
        numbers[0] = mutate_num_dims(numbers[0])
        # MUST adjust dim_size after changing num_dims
        numbers[1] = adjust_for_total_elements(numbers[0], numbers[1])
        # Adjust repeat based on new workload
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    elif mutation_choice < 0.7:
        # Mutate dim_size (40% probability)
        numbers[1] = mutate_dim_size(numbers[1], numbers[0])
        # Adjust repeat based on new workload
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    else:
        # Mutate repeat (30% probability)
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    # Sometimes mutate multiple parameters (15% chance)
    if random.random() < 0.15:
        numbers[0] = mutate_num_dims(numbers[0])
        numbers[1] = mutate_dim_size(numbers[1], numbers[0])
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_NUM_DIMS))
    numbers[1] = adjust_for_total_elements(numbers[0], numbers[1])
    numbers[1] = max(1, min(numbers[1], MAX_SAFE_DIM_SIZE))
    numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
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
