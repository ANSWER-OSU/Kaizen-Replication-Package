"""
Custom AFL++ fuzzer for LIF (Leaky Integrate-and-Fire) neuron simulation testing.

This fuzzer is designed to test the correctness of neural simulation calculations
by generating valid integer inputs for neurons_per_item, num_items, and num_steps parameters.

The program being tested: ./driver <neurons_per_item> <num_items> <num_steps>
- neurons_per_item: number of neurons per item (must be positive)
- num_items: number of items to process (must be positive)
- num_steps: number of simulation time steps (must be positive)

Key insights:
- Total neurons: num_neurons = neurons_per_item * num_items
- Memory allocation proportional to num_neurons and num_items
- Computation: num_neurons * num_steps operations
- Each step processes all neurons sequentially

Mutation strategy focuses on:
1. Various neurons_per_item values (common: 10-1000)
2. Different num_items values (batch size)
3. Small to medium num_steps (simulation steps)
4. Timeout protection: total work = num_neurons * num_steps

Setup:
1. Rename this file to: custom_fuzzer_lif.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_lif"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Configuration
POSITIVE_ONLY = True  # All parameters must be positive

# Conservative limits for 6000ms timeout
# Total work = neurons_per_item * num_items * num_steps
MAX_SAFE_NEURONS_PER_ITEM = 10000   # 10K neurons per item
MAX_SAFE_NUM_ITEMS = 10000          # 10K items
MAX_SAFE_NUM_STEPS = 1000           # 1K simulation steps
MAX_TOTAL_WORK = 100000000          # 100M operations (neurons * steps)

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer initialized on {node} with seed {seed}")
    print(f"Using limits: neurons_per_item<={MAX_SAFE_NEURONS_PER_ITEM}, num_items<={MAX_SAFE_NUM_ITEMS}, num_steps<={MAX_SAFE_NUM_STEPS}")
    print(f"Max total work: {MAX_TOTAL_WORK} (neurons_per_item * num_items * num_steps)")

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

def calculate_total_work(neurons_per_item, num_items, num_steps):
    """Calculate total work: neurons_per_item * num_items * num_steps"""
    try:
        total_neurons = neurons_per_item * num_items
        total_work = total_neurons * num_steps
        return total_work
    except:
        return MAX_TOTAL_WORK + 1

def mutate_neurons_per_item(original_value):
    """Apply targeted mutations for neurons_per_item parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Very small values (edge cases)
        new_value = random.choice([1, 2, 5, 10])
    
    elif mutation_type <= 4:
        # Small typical values (10-100)
        new_value = random.choice([10, 20, 50, 100])
    
    elif mutation_type <= 6:
        # Medium values (100-1000) - common in neural simulations
        new_value = random.choice([100, 200, 500, 1000])
    
    elif mutation_type <= 8:
        # Larger values
        new_value = random.choice([1000, 2000, 5000, 10000])
    
    elif mutation_type == 9:
        # Power of 2 values
        exp = random.randint(4, 13)  # 16 to 8192
        new_value = 2 ** exp
    
    else:
        # Add/subtract
        offset = random.choice([1, 10, 100, 1000])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    # Ensure positive and within bounds
    new_value = max(1, min(new_value, MAX_SAFE_NEURONS_PER_ITEM))
    return new_value

def mutate_num_items(original_value):
    """Apply targeted mutations for num_items parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Very small batch sizes
        new_value = random.choice([1, 2, 5, 10])
    
    elif mutation_type <= 4:
        # Small batch sizes
        new_value = random.choice([16, 32, 64, 100, 128])
    
    elif mutation_type <= 6:
        # Medium batch sizes (common in ML)
        new_value = random.choice([256, 512, 1000, 1024])
    
    elif mutation_type <= 8:
        # Larger batch sizes
        new_value = random.choice([2000, 5000, 10000])
    
    elif mutation_type == 9:
        # Power of 2 values
        exp = random.randint(4, 13)  # 16 to 8192
        new_value = 2 ** exp
    
    else:
        # Add/subtract
        offset = random.choice([1, 10, 100, 1000])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    # Ensure positive and within bounds
    new_value = max(1, min(new_value, MAX_SAFE_NUM_ITEMS))
    return new_value

def mutate_num_steps(original_value, neurons_per_item, num_items):
    """Apply targeted mutations for num_steps parameter"""
    mutation_type = random.randint(1, 10)
    
    # Calculate current workload
    total_neurons = neurons_per_item * num_items
    
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
        # Power of 2 values
        exp = random.randint(4, 10)  # 16 to 1024
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
    
    # Ensure positive
    new_value = max(1, new_value)
    
    # AGGRESSIVE timeout safety based on total work
    total_work = calculate_total_work(neurons_per_item, num_items, new_value)
    
    if total_work > MAX_TOTAL_WORK:
        # Calculate safe num_steps
        safe_steps = max(1, MAX_TOTAL_WORK // total_neurons)
        new_value = min(new_value, safe_steps)
    
    # Additional workload-based throttling
    if total_neurons > 5000000:
        new_value = min(new_value, 10)
    elif total_neurons > 1000000:
        new_value = min(new_value, 50)
    elif total_neurons > 500000:
        new_value = min(new_value, 100)
    elif total_neurons > 100000:
        new_value = min(new_value, 500)
    
    # Apply bounds
    new_value = min(new_value, MAX_SAFE_NUM_STEPS)
    
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
        numbers = [100, 100, 10]  # neurons_per_item=100, num_items=100, num_steps=10
    
    # Ensure we have at least 3 parameters
    while len(numbers) < 3:
        if len(numbers) == 0:
            numbers.append(100)   # neurons_per_item
        elif len(numbers) == 1:
            numbers.append(100)   # num_items
        else:
            numbers.append(10)    # num_steps
    
    # Ensure all values are positive
    numbers = [max(1, n) for n in numbers]
    
    # Decide which parameter to mutate
    mutation_choice = random.random()
    
    if mutation_choice < 0.35:
        # Mutate neurons_per_item (35% probability)
        numbers[0] = mutate_neurons_per_item(numbers[0])
        # Adjust num_steps to keep total work reasonable
        numbers[2] = mutate_num_steps(numbers[2], numbers[0], numbers[1])
    
    elif mutation_choice < 0.65:
        # Mutate num_items (30% probability)
        numbers[1] = mutate_num_items(numbers[1])
        # Adjust num_steps to keep total work reasonable
        numbers[2] = mutate_num_steps(numbers[2], numbers[0], numbers[1])
    
    else:
        # Mutate num_steps (35% probability)
        numbers[2] = mutate_num_steps(numbers[2], numbers[0], numbers[1])
    
    # Sometimes mutate multiple parameters (20% chance)
    if random.random() < 0.2:
        numbers[0] = mutate_neurons_per_item(numbers[0])
        numbers[1] = mutate_num_items(numbers[1])
        numbers[2] = mutate_num_steps(numbers[2], numbers[0], numbers[1])
    
    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_NEURONS_PER_ITEM))
    numbers[1] = max(1, min(numbers[1], MAX_SAFE_NUM_ITEMS))
    
    # Re-check num_steps with final values
    total_work = calculate_total_work(numbers[0], numbers[1], numbers[2])
    if total_work > MAX_TOTAL_WORK:
        total_neurons = numbers[0] * numbers[1]
        safe_steps = max(1, MAX_TOTAL_WORK // total_neurons)
        numbers[2] = min(numbers[2], safe_steps)
    
    numbers[2] = max(1, min(numbers[2], MAX_SAFE_NUM_STEPS))
    
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
