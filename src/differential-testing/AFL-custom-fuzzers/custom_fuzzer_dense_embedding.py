"""
Custom AFL++ fuzzer for dense embedding benchmark testing.

This fuzzer is designed to test the correctness of dense embedding calculations
by generating valid integer inputs for nrows, batch_size, and repeat parameters.

The program being tested: ./driver <nrows> <batch_size> <repeat>
- nrows: number of rows in embedding table (must be > batch_size * batch_size)
- batch_size: size of the batch (must be positive integer)
- repeat: number of benchmark repetitions (must be positive integer)

Key constraint: nrows > batch_size * batch_size

NOTE: The program runs 12 kernel launches per repeat (3 embedding dims × 4 block sizes),
making it very expensive. We use VERY conservative limits.

Mutation strategy focuses on:
1. Boundary values for batch_size (1, 2, small numbers)
2. Common batch sizes (8, 16, 32, 64)
3. Values that test the constraint nrows > batch_size²
4. Edge cases around block sizes (128, 256, 512, 1024)
5. Aggressive timeout protection (6000ms limit)

Setup:
1. Rename this file to: custom_fuzzer_embedding.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_embedding"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform
import math

# Configuration
POSITIVE_ONLY = True  # All parameters must be positive

# VERY conservative limits due to expensive kernels (12 launches per repeat)
# Timeout budget: 6000ms total
MAX_SAFE_NROWS = 50000         # 50K rows maximum
MAX_SAFE_BATCH_SIZE = 100      # 100 batch size maximum
MAX_SAFE_REPEAT = 5            # 5 repeats maximum (12 kernels × 5 = 60 launches)
MAX_COMBINED_LOAD = 100000     # nrows * batch_size cap

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer initialized on {node} with seed {seed}")
    print(f"Using conservative limits: nrows<={MAX_SAFE_NROWS}, batch_size<={MAX_SAFE_BATCH_SIZE}, repeat<={MAX_SAFE_REPEAT}")

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

def ensure_constraint(nrows, batch_size):
    """
    Ensure nrows > batch_size * batch_size
    Adjusts nrows if necessary
    """
    min_nrows = batch_size * batch_size + 1
    if nrows <= batch_size * batch_size:
        # Make nrows satisfy the constraint with small margin
        nrows = min_nrows + random.randint(1, 100)
    return nrows

def mutate_nrows(original_value, batch_size):
    """Apply targeted mutations for nrows parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 3:
        # Just above the constraint (batch_size² + small offset)
        offset = random.choice([1, 2, 5, 10, 20, 50])
        new_value = batch_size * batch_size + offset
    
    elif mutation_type <= 5:
        # Small multiples of batch_size² (testing boundary conditions)
        multiplier = random.choice([2, 3, 5])
        new_value = batch_size * batch_size * multiplier
    
    elif mutation_type <= 7:
        # Small to medium range values
        new_value = random.choice([
            100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000
        ])
        # Ensure constraint is met
        new_value = max(new_value, batch_size * batch_size + 1)
    
    elif mutation_type == 8:
        # Power of 2 values
        exp = random.randint(7, 15)  # 128 to 32K
        new_value = 2 ** exp
        new_value = max(new_value, batch_size * batch_size + 1)
    
    elif mutation_type == 9:
        # Add/subtract for boundary exploration
        offset = random.choice([1, 5, 10, 50, 100])
        new_value = original_value + random.choice([-offset, offset])
    
    else:
        # Multiply/divide by factors
        factor = random.choice([2, 5])
        if random.choice([True, False]) and original_value < MAX_SAFE_NROWS // factor:
            new_value = original_value * factor
        else:
            new_value = max(batch_size * batch_size + 1, original_value // factor)
    
    # Ensure constraint and bounds
    new_value = ensure_constraint(new_value, batch_size)
    new_value = min(new_value, MAX_SAFE_NROWS)
    new_value = max(batch_size * batch_size + 1, new_value)
    return new_value

def mutate_batch_size(original_value, nrows):
    """Apply targeted mutations for batch_size parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 3:
        # Very small batch sizes (edge cases)
        new_value = random.choice([1, 2, 3, 4, 5])
    
    elif mutation_type <= 6:
        # Small common batch sizes
        new_value = random.choice([8, 16, 24, 32, 48, 64])
    
    elif mutation_type <= 8:
        # Medium batch sizes (still safe)
        new_value = random.choice([10, 20, 30, 40, 50, 75, 100])
    
    elif mutation_type == 9:
        # Test the constraint: batch_size close to sqrt(nrows)
        max_batch = int(math.sqrt(nrows - 1))
        if max_batch > 1:
            new_value = max(1, max_batch - random.randint(0, 3))
        else:
            new_value = 1
    
    else:
        # Add/subtract small amounts
        offset = random.choice([1, 2, 5, 10])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    # Ensure positive
    new_value = max(1, new_value)
    
    # Ensure constraint: batch_size² < nrows
    max_allowed_batch = int(math.sqrt(nrows - 1))
    new_value = min(new_value, max_allowed_batch, MAX_SAFE_BATCH_SIZE)
    
    # Safety: if batch_size would be 0 or negative, set to 1
    new_value = max(1, new_value)
    
    return new_value

def mutate_repeat(original_value, nrows, batch_size):
    """Apply targeted mutations for repeat parameter - VERY conservative"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 7:
        # Almost always use minimal repeat counts
        new_value = random.choice([1, 1, 1, 2, 2, 3])
    
    elif mutation_type <= 9:
        # Rarely use slightly higher counts
        new_value = random.choice([4, 5])
    
    else:
        # Very rarely test higher
        new_value = random.choice([1, 2, 3, 5])
    
    # Ensure positive
    new_value = max(1, new_value)
    
    # AGGRESSIVE timeout safety
    # The program does 12 kernel launches per repeat
    workload = nrows * batch_size
    
    if workload > 50000:
        new_value = 1  # Force to 1 for large workloads
    elif workload > 20000:
        new_value = min(new_value, 2)
    elif workload > 10000:
        new_value = min(new_value, 3)
    elif workload > 5000:
        new_value = min(new_value, 4)
    
    # Overall safety check
    if nrows * batch_size > MAX_COMBINED_LOAD:
        new_value = 1
    
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
    
    # If no valid input, start with minimal safe defaults
    if not numbers:
        numbers = [100, 5, 1]  # nrows=100, batch_size=5, repeat=1
    
    # Ensure we have at least 3 parameters
    while len(numbers) < 3:
        if len(numbers) == 0:
            numbers.append(100)  # nrows
        elif len(numbers) == 1:
            numbers.append(5)  # batch_size
        else:
            numbers.append(1)  # repeat
    
    # Ensure constraint is initially satisfied
    numbers[0] = ensure_constraint(numbers[0], numbers[1])
    
    # Decide which parameter to mutate
    mutation_choice = random.random()
    
    if mutation_choice < 0.4:
        # Mutate nrows (40% probability)
        numbers[0] = mutate_nrows(numbers[0], numbers[1])
        # Always adjust repeat to stay within timeout
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    elif mutation_choice < 0.7:
        # Mutate batch_size (30% probability)
        old_batch_size = numbers[1]
        numbers[1] = mutate_batch_size(numbers[1], numbers[0])
        # If batch_size changed, ensure constraint still holds
        numbers[0] = ensure_constraint(numbers[0], numbers[1])
        # Always adjust repeat
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    else:
        # Mutate repeat (30% probability)
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    # Final constraint check
    numbers[0] = ensure_constraint(numbers[0], numbers[1])
    
    # Final safety checks
    numbers[0] = min(numbers[0], MAX_SAFE_NROWS)
    numbers[1] = min(numbers[1], MAX_SAFE_BATCH_SIZE)
    numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])  # Re-check repeat
    
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
