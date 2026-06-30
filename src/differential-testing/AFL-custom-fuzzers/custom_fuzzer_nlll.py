"""
Custom AFL++ fuzzer for NLL (Negative Log Likelihood) Loss kernel testing.

This fuzzer is designed to test the correctness of NLL loss calculations
by generating valid integer inputs for nframe, n_classes, and repeat parameters.

The program being tested: ./driver <nframe> <n_classes> <repeat>
- nframe: minibatch size / number of frames (must be positive)
- n_classes: number of classification classes (must be positive)
- repeat: number of kernel repetitions (must be positive)

Key insights:
- Total elements: input_size = nframe * n_classes
- Memory allocation: input (nframe*n_classes), weights (nframe), target (nframe)
- Computation complexity: O(nframe * n_classes) per iteration
- Tests 5 thread block sizes: 64, 128, 256, 512, 1024
- Total work: nframe * n_classes * repeat * 5 (for 5 block sizes)

Mutation strategy focuses on:
1. Common minibatch sizes (powers of 2: 16, 32, 64, 128, 256)
2. Typical number of classes (10, 100, 1000, 10000 - classification tasks)
3. Small repeat counts due to 5 block size tests per run
4. Timeout protection based on total work

Setup:
1. Rename this file to: custom_fuzzer_nll.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_nll"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Configuration
POSITIVE_ONLY = True  # All parameters must be positive

# Conservative limits for 6000ms timeout per execution
# The program runs 5 thread block sizes per execution (64, 128, 256, 512, 1024)
MAX_SAFE_NFRAME = 100000        # 100K frames
MAX_SAFE_N_CLASSES = 100000     # 100K classes
MAX_SAFE_REPEAT = 100           # 100 repeats
MAX_TOTAL_WORK = 500000000      # 500M operations (nframe * n_classes * repeat * 5)

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer initialized on {node} with seed {seed}")
    print(f"Using limits: nframe<={MAX_SAFE_NFRAME}, n_classes<={MAX_SAFE_N_CLASSES}, repeat<={MAX_SAFE_REPEAT}")
    print(f"Max total work: {MAX_TOTAL_WORK} (nframe * n_classes * repeat * 5 block sizes)")

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

def calculate_total_work(nframe, n_classes, repeat):
    """
    Calculate total work: nframe * n_classes * repeat * 5
    (5 because program tests 5 different thread block sizes)
    """
    try:
        work_per_repeat = nframe * n_classes
        total_work = work_per_repeat * repeat * 5  # 5 block sizes
        return total_work
    except:
        return MAX_TOTAL_WORK + 1

def mutate_nframe(original_value):
    """Apply targeted mutations for nframe (minibatch size) parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Very small minibatch sizes (edge cases)
        new_value = random.choice([1, 2, 4, 8, 16])
    
    elif mutation_type <= 4:
        # Small typical minibatch sizes
        new_value = random.choice([32, 64, 128, 256])
    
    elif mutation_type <= 6:
        # Medium minibatch sizes (common in deep learning)
        new_value = random.choice([512, 1024, 2048])
    
    elif mutation_type <= 8:
        # Larger minibatch sizes
        new_value = random.choice([4096, 8192, 16384])
    
    elif mutation_type == 9:
        # Power of 2 values
        exp = random.randint(4, 16)  # 16 to 65536
        new_value = 2 ** exp
    
    else:
        # Add/subtract
        offset = random.choice([1, 10, 100, 1000])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    # Ensure positive and within bounds
    new_value = max(1, min(new_value, MAX_SAFE_NFRAME))
    return new_value

def mutate_n_classes(original_value):
    """Apply targeted mutations for n_classes parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Very small number of classes (edge cases)
        new_value = random.choice([2, 5, 10])
    
    elif mutation_type <= 4:
        # Small classification tasks (MNIST, CIFAR-10)
        new_value = random.choice([10, 20, 50, 100])
    
    elif mutation_type <= 6:
        # Medium classification tasks (CIFAR-100, ImageNet-subset)
        new_value = random.choice([100, 200, 500, 1000])
    
    elif mutation_type <= 8:
        # Large classification tasks (ImageNet, large vocabulary)
        new_value = random.choice([1000, 5000, 10000, 50000])
    
    elif mutation_type == 9:
        # Power of 2 values
        exp = random.randint(4, 16)  # 16 to 65536
        new_value = 2 ** exp
    
    else:
        # Add/subtract
        offset = random.choice([1, 10, 100, 1000])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    # Ensure positive and within bounds
    new_value = max(1, min(new_value, MAX_SAFE_N_CLASSES))
    return new_value

def mutate_repeat(original_value, nframe, n_classes):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)
    
    # Calculate workload per repeat (including 5 block size tests)
    work_per_repeat = nframe * n_classes * 5
    
    if mutation_type <= 5:
        # Small repeat counts (program tests 5 block sizes, so keep this low)
        new_value = random.choice([1, 2, 3, 5, 10])
    
    elif mutation_type <= 7:
        # Medium repeat counts
        new_value = random.choice([10, 20, 50])
    
    elif mutation_type <= 8:
        # Larger repeat counts
        new_value = random.choice([50, 100])
    
    elif mutation_type == 9:
        # Add/subtract
        offset = random.choice([1, 5, 10])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Multiply/divide
        factor = random.choice([2, 5, 10])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    # Ensure positive
    new_value = max(1, new_value)
    
    # AGGRESSIVE timeout safety based on total work
    total_work = calculate_total_work(nframe, n_classes, new_value)
    
    if total_work > MAX_TOTAL_WORK:
        # Calculate safe repeat count
        safe_repeat = max(1, MAX_TOTAL_WORK // work_per_repeat)
        new_value = min(new_value, safe_repeat)
    
    # Additional workload-based throttling
    if work_per_repeat > 50000000:  # > 50M ops per repeat
        new_value = min(new_value, 5)
    elif work_per_repeat > 10000000:  # > 10M ops per repeat
        new_value = min(new_value, 10)
    elif work_per_repeat > 5000000:   # > 5M ops per repeat
        new_value = min(new_value, 20)
    elif work_per_repeat > 1000000:   # > 1M ops per repeat
        new_value = min(new_value, 50)
    
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
        numbers = [128, 100, 10]  # nframe=128, n_classes=100, repeat=10
    
    # Ensure we have at least 3 parameters
    while len(numbers) < 3:
        if len(numbers) == 0:
            numbers.append(128)   # nframe (minibatch size)
        elif len(numbers) == 1:
            numbers.append(100)   # n_classes
        else:
            numbers.append(10)    # repeat
    
    # Ensure all values are positive
    numbers = [max(1, n) for n in numbers]
    
    # Decide which parameter to mutate
    mutation_choice = random.random()
    
    if mutation_choice < 0.35:
        # Mutate nframe (35% probability)
        numbers[0] = mutate_nframe(numbers[0])
        # Adjust repeat to keep total work reasonable
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    elif mutation_choice < 0.65:
        # Mutate n_classes (30% probability)
        numbers[1] = mutate_n_classes(numbers[1])
        # Adjust repeat to keep total work reasonable
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    else:
        # Mutate repeat (35% probability)
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    # Sometimes mutate multiple parameters (20% chance)
    if random.random() < 0.2:
        numbers[0] = mutate_nframe(numbers[0])
        numbers[1] = mutate_n_classes(numbers[1])
        numbers[2] = mutate_repeat(numbers[2], numbers[0], numbers[1])
    
    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_NFRAME))
    numbers[1] = max(1, min(numbers[1], MAX_SAFE_N_CLASSES))
    
    # Re-check repeat with final values
    total_work = calculate_total_work(numbers[0], numbers[1], numbers[2])
    if total_work > MAX_TOTAL_WORK:
        work_per_repeat = numbers[0] * numbers[1] * 5
        safe_repeat = max(1, MAX_TOTAL_WORK // work_per_repeat)
        numbers[2] = min(numbers[2], safe_repeat)
    
    numbers[2] = max(1, min(numbers[2], MAX_SAFE_REPEAT))
    
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
