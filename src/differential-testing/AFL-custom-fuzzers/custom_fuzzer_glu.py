"""
Custom AFL++ fuzzer for GLU (Gated Linear Unit) kernel testing.

This fuzzer is designed to test the correctness of GLU activation function
by generating valid integer inputs for ndims, dim_size, and repeat parameters.

The program being tested: ./driver <ndims> <dim_size> <repeat>
- ndims: number of dimensions in the tensor (must be positive, typically 2-5)
- dim_size: size of each dimension (must be positive, preferably even for split)
- repeat: number of benchmark repetitions (must be positive)

Key insights:
- Total elements: nelems = dim_size^ndims (exponential growth!)
- The program tests multiple split dimensions: 3 * (ndims - 1) + 1 iterations
- Each iteration requires: m * split_dim_size * n operations
- Memory allocation: nelems * sizeof(float) * 3 (X, Y, Y_ref)
- Important constraint: Split dimension sizes should be even (divisible by 2)

Mutation strategy focuses on:
1. Small ndims (2-5) which are realistic for ML tensors
2. Even dim_sizes (to satisfy split dimension requirement)
3. Various dim_sizes (small, medium, powers of 2)
4. Timeout protection due to exponential complexity and multiple iterations

Setup:
1. Rename this file to: custom_fuzzer_glu.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_glu"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Configuration
POSITIVE_ONLY = True  # All parameters must be positive

# Conservative limits due to EXPONENTIAL complexity: nelems = dim_size^ndims
# Plus the program runs 3*(ndims-1)+1 iterations per execution
MAX_SAFE_NDIMS = 6           # Beyond 6 dims is rarely used
MAX_SAFE_DIM_SIZE = 1000     # Keep reasonable
MAX_SAFE_REPEAT = 20         # Moderate repeat
MAX_TOTAL_ELEMENTS = 10000000  # 10M elements maximum

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer initialized on {node} with seed {seed}")
    print(f"Using limits: ndims<={MAX_SAFE_NDIMS}, dim_size<={MAX_SAFE_DIM_SIZE}, repeat<={MAX_SAFE_REPEAT}")
    print(f"Max total elements: {MAX_TOTAL_ELEMENTS} (dim_size^ndims)")
    print(f"NOTE: Program tests 3*(ndims-1)+1 split dimensions per execution")

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

def calculate_total_elements(ndims, dim_size):
    """Calculate total elements: dim_size^ndims"""
    try:
        result = float(dim_size) ** float(ndims)
        if result > MAX_TOTAL_ELEMENTS or result != result:  # Check for overflow or NaN
            return MAX_TOTAL_ELEMENTS + 1
        return int(result)
    except:
        return MAX_TOTAL_ELEMENTS + 1

def adjust_for_total_elements(ndims, dim_size):
    """
    Adjust dim_size to keep total elements reasonable
    If dim_size^ndims > MAX_TOTAL_ELEMENTS, reduce dim_size
    """
    total = calculate_total_elements(ndims, dim_size)
    
    if total > MAX_TOTAL_ELEMENTS:
        # Calculate safe dim_size: dim_size = MAX_TOTAL_ELEMENTS^(1/ndims)
        safe_dim_size = int(MAX_TOTAL_ELEMENTS ** (1.0 / ndims))
        # Add margin for safety
        safe_dim_size = max(2, int(safe_dim_size * 0.8))
        # Ensure it's even (for split dimension requirement)
        if safe_dim_size % 2 != 0:
            safe_dim_size -= 1
        return max(2, safe_dim_size)
    
    return dim_size

def mutate_ndims(original_value):
    """Apply targeted mutations for ndims parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 3:
        # Most common: 2D, 3D, 4D tensors (very common in ML)
        new_value = random.choice([2, 3, 4])
    
    elif mutation_type <= 5:
        # Edge case: 1D tensor
        new_value = 1
    
    elif mutation_type <= 7:
        # 5D tensor (still reasonable)
        new_value = 5
    
    elif mutation_type == 8:
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
    new_value = max(1, min(new_value, MAX_SAFE_NDIMS))
    return new_value

def mutate_dim_size(original_value, ndims):
    """
    Apply targeted mutations for dim_size parameter
    Important: Prefer even values (divisible by 2) for split dimension requirement
    """
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Small even dimensions
        new_value = random.choice([2, 4, 6, 8, 10])
    
    elif mutation_type <= 4:
        # Small-medium even dimensions
        new_value = random.choice([16, 20, 32, 40, 50, 64])
    
    elif mutation_type <= 6:
        # Medium even dimensions (common powers of 2)
        new_value = random.choice([64, 128, 256, 512])
    
    elif mutation_type <= 8:
        # Larger even dimensions (be careful with ndims)
        if ndims <= 2:
            new_value = random.choice([512, 1000])
        elif ndims == 3:
            new_value = random.choice([100, 200])
        else:
            new_value = random.choice([10, 20, 50])
    
    elif mutation_type == 9:
        # Power of 2 values (always even)
        exp = random.randint(1, 10)  # 2 to 1024
        new_value = 2 ** exp
    
    else:
        # Add/subtract, ensuring result is even
        offset = random.choice([2, 4, 10, 20]) * 2
        new_value = max(2, original_value + random.choice([-offset, offset]))
    
    # Ensure positive and even
    new_value = max(2, new_value)
    if new_value % 2 != 0:
        new_value += 1  # Make it even
    
    # CRITICAL: Adjust for total elements constraint
    new_value = adjust_for_total_elements(ndims, new_value)
    
    # Ensure still even after adjustment
    if new_value % 2 != 0:
        new_value = max(2, new_value - 1)
    
    # Apply bounds
    new_value = min(new_value, MAX_SAFE_DIM_SIZE)
    
    # Final check: must be even
    if new_value % 2 != 0:
        new_value -= 1
    
    return max(2, new_value)

def mutate_repeat(original_value, ndims, dim_size):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)
    
    # Calculate workload
    total_elements = calculate_total_elements(ndims, dim_size)
    # The program runs 3*(ndims-1)+1 split dimension tests per execution
    num_iterations_per_run = 3 * (ndims - 1) + 1 if ndims > 1 else 1
    
    if mutation_type <= 5:
        # Most of the time: small repeat counts
        new_value = random.choice([1, 2, 3, 5, 10])
    
    elif mutation_type <= 7:
        # Sometimes: medium repeat counts
        new_value = random.choice([10, 20])
    
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
    
    # AGGRESSIVE timeout safety
    # Account for multiple iterations (3*(ndims-1)+1 per run)
    if total_elements > 5000000:
        new_value = min(new_value, 2)
    elif total_elements > 1000000:
        new_value = min(new_value, 5)
    elif total_elements > 500000:
        new_value = min(new_value, 10)
    
    # Additional factor: more dimensions = more split tests
    if ndims >= 5:
        new_value = min(new_value, 5)
    elif ndims >= 4:
        new_value = min(new_value, 10)
    
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
        numbers = [3, 10, 5]  # ndims=3, dim_size=10, repeat=5
    
    # Ensure we have at least 3 parameters
    while len(numbers) < 3:
        if len(numbers) == 0:
            numbers.append(3)    # ndims
        elif len(numbers) == 1:
            numbers.append(10)   # dim_size (even)
        else:
            numbers.append(5)    # repeat
    
    # Ensure all values are positive
    numbers = [max(1, n) for n in numbers]
    
    # Ensure dim_size is even
    if numbers[1] % 2 != 0:
        numbers[1] += 1
    
    # Adjust dim_size to keep total elements reasonable
    numbers[1] = adjust_for_total_elements(numbers[0], numbers[1])
    
    # Decide which parameter to mutate
    mutation_choice = random.random()
    
    if mutation_choice < 0.3:
        # Mutate ndims (30% probability)
        numbers[0] = mutate_ndims(numbers[0])
        # MUST adjust dim_size after changing ndims
        numbers[1] = adjust_for_total_elements(numbers[0], numbers[1])
        # Ensure still even
        if numbers[1] % 2 != 0:
            numbers[1] = max(2, numbers[1] - 1)
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
        numbers[0] = mutate_ndims(numbers[0])
        numbers[1] = mutate_dim_size(numbers[1], numbers[0])
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_NDIMS))
    numbers[1] = adjust_for_total_elements(numbers[0], numbers[1])
    numbers[1] = max(2, min(numbers[1], MAX_SAFE_DIM_SIZE))
    
    # CRITICAL: Ensure dim_size is even
    if numbers[1] % 2 != 0:
        numbers[1] = max(2, numbers[1] - 1)
    
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
