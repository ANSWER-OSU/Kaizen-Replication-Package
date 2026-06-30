"""
Custom AFL++ fuzzer for cross product benchmark testing.

This fuzzer is designed to test the correctness of cross product calculations
by generating valid integer inputs for nrows and repeat parameters.

The program being tested: ./driver <nrows> <repeat>
- nrows: number of rows in 2D tensor (must be positive integer)
- repeat: number of benchmark repetitions (must be positive integer)

Mutation strategy focuses on:
1. Boundary values (1, 2, small numbers)
2. Thread-related values (multiples of 256)
3. Power-of-2 values
4. Medium-range values for thorough testing
5. Timeout-aware mutations (avoids combinations that exceed 6000ms)

Setup:
1. Rename this file to: custom_fuzzer_cross.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_cross"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Configuration
POSITIVE_ONLY = True  # Both nrows and repeat must be positive

# Timeout budget: 6000ms total
# Rough estimate: each iteration takes some time, we want to stay well under timeout
MAX_SAFE_NROWS = 10000000      # 10M rows should be safe for correctness testing
MAX_SAFE_REPEAT = 100          # Keep repeat low for faster fuzzing iterations
MAX_COMBINED_LOAD = 50000000   # nrows * repeat should stay under this

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

def mutate_nrows(original_value):
    """Apply targeted mutations for nrows parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Edge cases: very small values
        new_value = random.choice([1, 2, 3, 5, 7, 10, 15, 31, 63, 127])
    
    elif mutation_type <= 4:
        # Thread-related boundaries (multiples/divisors of 256)
        # These may expose parallelization issues
        new_value = random.choice([64, 128, 192, 256, 384, 512, 768, 1024, 2048])
    
    elif mutation_type <= 6:
        # Power of 2 values (common edge cases in parallel code)
        exp = random.randint(4, 16)  # 16 to 65536
        new_value = 2 ** exp
    
    elif mutation_type <= 8:
        # Medium range values (good for thorough correctness testing)
        new_value = random.choice([
            100, 500, 1000, 5000, 10000,
            50000, 100000, 500000, 1000000
        ])
    
    elif mutation_type == 9:
        # Add/subtract for boundary exploration
        offset = random.choice([1, 2, 5, 10, 100, 1000])
        new_value = original_value + random.choice([-offset, offset])
    
    else:
        # Multiply/divide by small factors
        factor = random.choice([2, 3, 5, 10])
        if random.choice([True, False]) and original_value < MAX_SAFE_NROWS // factor:
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    # Ensure positive and within safe bounds
    new_value = max(1, min(new_value, MAX_SAFE_NROWS))
    return new_value

def mutate_repeat(original_value, nrows):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 3:
        # Very small repeat counts (minimal testing, fast)
        new_value = random.choice([1, 2, 3, 5])
    
    elif mutation_type <= 6:
        # Small to medium repeat counts (balance speed/thoroughness)
        new_value = random.choice([10, 20, 50, 100])
    
    elif mutation_type <= 8:
        # Medium repeat counts for more thorough testing
        new_value = random.choice([200, 500, 1000])
    
    elif mutation_type == 9:
        # Add/subtract small amounts
        offset = random.choice([1, 5, 10, 50])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Multiply/divide by factors
        factor = random.choice([2, 5, 10])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    # Ensure positive
    new_value = max(1, new_value)
    
    # Timeout safety: avoid combinations that might timeout
    # If nrows is large, keep repeat small
    if nrows > 1000000:
        new_value = min(new_value, 10)
    elif nrows > 100000:
        new_value = min(new_value, 50)
    
    # Overall safety check
    if nrows * new_value > MAX_COMBINED_LOAD:
        new_value = max(1, MAX_COMBINED_LOAD // nrows)
    
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
    
    # If no valid input, start with reasonable defaults
    if not numbers:
        numbers = [1000, 10]
    
    # Ensure we have at least 2 parameters (nrows and repeat)
    if len(numbers) < 2:
        numbers.append(10)  # Add default repeat value
    
    # Decide which parameter to mutate
    if random.random() < 0.5 and len(numbers) >= 2:
        # Mutate nrows (index 0)
        numbers[0] = mutate_nrows(numbers[0])
        # Possibly adjust repeat to stay within timeout
        numbers[1] = mutate_repeat(numbers[1], numbers[0])
    else:
        # Mutate repeat (index 1 if exists)
        if len(numbers) >= 2:
            numbers[1] = mutate_repeat(numbers[1], numbers[0])
    
    # Keep only first two parameters (nrows and repeat)
    numbers = numbers[:2]
    
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
