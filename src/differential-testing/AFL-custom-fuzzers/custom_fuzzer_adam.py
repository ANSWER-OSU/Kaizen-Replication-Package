"""
Custom AFL++ fuzzer for adam benchmark testing.

This fuzzer is designed to test the correctness of adam calculations
by generating valid integer inputs for vector_size, time_step, and repeat parameters.

The program being tested: ./main <vector_size> <time_step> <repeat>
- vector_size: number of elements in the vector (must be positive integer)
- time_step: number of time steps (must be positive integer)
- repeat: number of benchmark repetitions (must be positive integer)

Setup:
1. Rename this file to: custom_fuzzer_adam.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_adam"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Timeout budget
# We limit the max combinations to prevent timeouts or memory exhaustion in tests
MAX_SAFE_VECTOR_SIZE = 50000000  # Up to 50M elements
MAX_SAFE_TIME_STEP = 1000        # Time step iterations
MAX_SAFE_REPEAT = 100            # Number of benchmark repetitions
MAX_COMBINED_LOAD = 500000000    # bounds on vector_size * time_step * repeat

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

def mutate_vector_size(original_value):
    """Apply targeted mutations for vector_size parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Edge cases: very small values
        new_value = random.choice([1, 2, 3, 5, 7, 10, 15, 31, 63, 127])
    
    elif mutation_type <= 4:
        # Thread-related boundaries (multiples/divisors of 256)
        new_value = random.choice([64, 128, 192, 256, 384, 512, 768, 1024, 2048])
    
    elif mutation_type <= 6:
        # Power of 2 values
        exp = random.randint(4, 16)  # 16 to 65536
        new_value = 2 ** exp
    
    elif mutation_type <= 8:
        # Medium range values
        new_value = random.choice([
            100, 500, 1000, 5000, 10000,
            50000, 100000, 500000, 1000000
        ])
    
    elif mutation_type == 9:
        # Add/subtract for boundary exploration
        offset = random.choice([1, 2, 5, 10, 100, 1000])
        new_value = original_value + random.choice([-offset, offset])
    
    else:
        # Multiply/divide by factors
        factor = random.choice([2, 3, 5, 10])
        if random.choice([True, False]) and original_value < MAX_SAFE_VECTOR_SIZE // factor:
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    new_value = max(1, min(new_value, MAX_SAFE_VECTOR_SIZE))
    return new_value

def mutate_time_step(original_value, vector_size):
    """Apply targeted mutations for time_step parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 3:
        new_value = random.choice([1, 2, 3, 5])
    elif mutation_type <= 6:
        new_value = random.choice([10, 20, 50, 100])
    elif mutation_type <= 8:
        new_value = random.choice([200, 500, 1000])
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
    new_value = min(new_value, MAX_SAFE_TIME_STEP)
    return new_value

def mutate_repeat(original_value, vector_size, time_step):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 3:
        new_value = random.choice([1, 2, 3, 5])
    elif mutation_type <= 6:
        new_value = random.choice([10, 20, 50, 100])
    elif mutation_type <= 8:
        new_value = random.choice([200, 500, 1000])
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
    
    # Restrict repeat heavily if the operations are already intense
    if vector_size * time_step > 1000000:
        new_value = min(new_value, 10)
    elif vector_size * time_step > 100000:
        new_value = min(new_value, 50)
        
    if vector_size * time_step * new_value > MAX_COMBINED_LOAD:
        new_value = max(1, MAX_COMBINED_LOAD // max(1, vector_size * time_step))
        
    new_value = min(new_value, MAX_SAFE_REPEAT)
    return new_value

def fuzz(buf, add_buf, max_size):
    """
    Main fuzzing function called by AFL++
    """
    try:
        buf_str = buf.decode('utf-8', errors='ignore').strip()
    except:
        buf_str = str(buf).strip()
    
    numbers = parse_input(buf_str)
    
    # If no valid input, start with reasonable defaults matching typical invocation
    if not numbers:
        numbers = [10000, 200, 100]
    
    while len(numbers) < 3:
        if len(numbers) == 1:
            numbers.append(200)
        elif len(numbers) == 2:
            numbers.append(100)
    
    # Decide which parameter to mutate
    choice = random.randint(0, 2)
    if choice == 0:
        numbers[0] = mutate_vector_size(numbers[0])
        numbers[1] = mutate_time_step(numbers[1], numbers[0])
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    elif choice == 1:
        numbers[1] = mutate_time_step(numbers[1], numbers[0])
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    else:
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    # Safety fallback to prevent any timeouts
    while numbers[0] * numbers[1] * numbers[2] > MAX_COMBINED_LOAD:
        if numbers[2] > 1:
            numbers[2] = max(1, numbers[2] // 2)
        elif numbers[1] > 1:
            numbers[1] = max(1, numbers[1] // 2)
        else:
            numbers[0] = max(1, numbers[0] // 2)
            
    # Pack up the first 3 parameters
    numbers = numbers[:3]
    
    mutated_string = " ".join(str(num) for num in numbers)
    
    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]
    
    return bytearray(mutated_string.encode('utf-8'))

def fuzz_count(buf):
    """
    Return number of mutations to perform
    """
    return 1
