"""
Custom AFL++ fuzzer for background subtraction (BGS) kernel testing.

This fuzzer is designed to test the correctness of background subtraction algorithms
by generating valid integer inputs for width, height, merged, and repeat parameters.

The program being tested: ./driver <width> <height> <merged> <repeat>
- width: image width in pixels (must be positive)
- height: image height in pixels (must be positive)
- merged: use merged kernel (0) or separate kernels (1) - binary flag
- repeat: number of frame processing repetitions (must be positive)

Key insights:
- Total pixels: imgSize = width * height
- Memory allocation: 8 arrays of size imgSize (Img, Img1, Img2, Bn, Bn_ref, Mp, Tn, Tn_ref)
- Timing starts from frame 3 (i >= 2), so effective iterations = repeat - 2
- Linear complexity: O(imgSize * repeat)
- Tests both merged and separate kernel implementations

Mutation strategy focuses on:
1. Common image resolutions (small to large)
2. Power-of-2 dimensions and standard aspect ratios
3. Binary merged flag (0 or 1)
4. Various repeat counts
5. Timeout protection based on total pixels and repeat count

Setup:
1. Rename this file to: custom_fuzzer_bgs.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_bgs"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# Configuration
POSITIVE_ONLY = True  # width, height, and repeat must be positive

# Conservative limits for 6000ms timeout
# Total work = width * height * repeat
MAX_SAFE_WIDTH = 10000           # 10K pixels wide
MAX_SAFE_HEIGHT = 10000          # 10K pixels tall
MAX_SAFE_REPEAT = 1000           # 1K frames
MAX_TOTAL_PIXELS = 100000000     # 100M pixels (width * height * repeat)

# Common image resolutions for reference
COMMON_RESOLUTIONS = [
    (320, 240),    # QVGA
    (640, 480),    # VGA
    (800, 600),    # SVGA
    (1024, 768),   # XGA
    (1280, 720),   # HD 720p
    (1920, 1080),  # Full HD 1080p
    (3840, 2160),  # 4K UHD
]

# Global state
node = ""

def init(seed):
    """Initialize the fuzzer with a random seed"""
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom fuzzer initialized on {node} with seed {seed}")
    print(f"Using limits: width<={MAX_SAFE_WIDTH}, height<={MAX_SAFE_HEIGHT}, repeat<={MAX_SAFE_REPEAT}")
    print(f"Max total pixels: {MAX_TOTAL_PIXELS} (width * height * repeat)")

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

def calculate_total_pixels(width, height, repeat):
    """Calculate total pixels processed: width * height * repeat"""
    try:
        imgSize = width * height
        total = imgSize * repeat
        return total
    except:
        return MAX_TOTAL_PIXELS + 1

def mutate_width(original_value):
    """Apply targeted mutations for width parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Very small widths (edge cases)
        new_value = random.choice([1, 10, 50, 100])
    
    elif mutation_type <= 4:
        # Small widths (100-1000)
        new_value = random.choice([100, 320, 640, 800, 1024])
    
    elif mutation_type <= 6:
        # Medium widths (HD range)
        new_value = random.choice([1280, 1920, 2048])
    
    elif mutation_type <= 7:
        # Use common resolutions
        width, _ = random.choice(COMMON_RESOLUTIONS)
        new_value = width
    
    elif mutation_type == 8:
        # Power of 2 values
        exp = random.randint(4, 12)  # 16 to 4096
        new_value = 2 ** exp
    
    elif mutation_type == 9:
        # Add/subtract
        offset = random.choice([1, 10, 100, 1000])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Multiply/divide by factors
        factor = random.choice([2, 4])
        if random.choice([True, False]) and original_value < MAX_SAFE_WIDTH // factor:
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    # Ensure positive and within bounds
    new_value = max(1, min(new_value, MAX_SAFE_WIDTH))
    return new_value

def mutate_height(original_value):
    """Apply targeted mutations for height parameter"""
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 2:
        # Very small heights (edge cases)
        new_value = random.choice([1, 10, 50, 100])
    
    elif mutation_type <= 4:
        # Small heights (100-1000)
        new_value = random.choice([100, 240, 480, 600, 768])
    
    elif mutation_type <= 6:
        # Medium heights (HD range)
        new_value = random.choice([720, 1080, 1200])
    
    elif mutation_type <= 7:
        # Use common resolutions
        _, height = random.choice(COMMON_RESOLUTIONS)
        new_value = height
    
    elif mutation_type == 8:
        # Power of 2 values
        exp = random.randint(4, 12)  # 16 to 4096
        new_value = 2 ** exp
    
    elif mutation_type == 9:
        # Add/subtract
        offset = random.choice([1, 10, 100, 1000])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Multiply/divide by factors
        factor = random.choice([2, 4])
        if random.choice([True, False]) and original_value < MAX_SAFE_HEIGHT // factor:
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    # Ensure positive and within bounds
    new_value = max(1, min(new_value, MAX_SAFE_HEIGHT))
    return new_value

def mutate_merged(original_value):
    """
    Apply targeted mutations for merged parameter (0 or 1)
    0 = use separate kernels (findMovingPixels, updateBackground, updateThreshold)
    1 = use merged kernel
    """
    mutation_type = random.randint(1, 10)
    
    if mutation_type <= 5:
        # 50% of the time: test merged kernel
        new_value = 1
    elif mutation_type <= 9:
        # 40% of the time: test separate kernels
        new_value = 0
    else:
        # 10% of the time: flip current value
        new_value = 1 - original_value
    
    # Ensure it's 0 or 1
    new_value = 1 if new_value != 0 else 0
    return new_value

def mutate_repeat(original_value, width, height):
    """Apply targeted mutations for repeat parameter"""
    mutation_type = random.randint(1, 10)
    
    # Calculate image size
    imgSize = width * height
    
    if mutation_type <= 4:
        # Small repeat counts (note: timing starts at frame 3)
        new_value = random.choice([3, 5, 10, 20, 50])
    
    elif mutation_type <= 6:
        # Medium repeat counts
        new_value = random.choice([100, 200, 500])
    
    elif mutation_type <= 8:
        # Larger repeat counts
        new_value = random.choice([500, 1000])
    
    elif mutation_type == 9:
        # Add/subtract
        offset = random.choice([1, 5, 10, 50, 100])
        new_value = max(1, original_value + random.choice([-offset, offset]))
    
    else:
        # Multiply/divide by factors
        factor = random.choice([2, 5, 10])
        if random.choice([True, False]):
            new_value = original_value * factor
        else:
            new_value = max(1, original_value // factor)
    
    # Ensure positive (minimum 3 for timing to work properly)
    new_value = max(3, new_value)
    
    # AGGRESSIVE timeout safety based on total pixels
    total_pixels = calculate_total_pixels(width, height, new_value)
    
    if total_pixels > MAX_TOTAL_PIXELS:
        # Calculate safe repeat count
        safe_repeat = max(3, MAX_TOTAL_PIXELS // imgSize)
        new_value = min(new_value, safe_repeat)
    
    # Additional workload-based throttling
    if imgSize > 5000000:  # > 5M pixels per frame
        new_value = min(new_value, 10)
    elif imgSize > 2000000:  # > 2M pixels per frame
        new_value = min(new_value, 50)
    elif imgSize > 1000000:  # > 1M pixels per frame
        new_value = min(new_value, 100)
    elif imgSize > 500000:   # > 500K pixels per frame
        new_value = min(new_value, 200)
    
    # Apply bounds
    new_value = min(new_value, MAX_SAFE_REPEAT)
    
    # Ensure minimum of 3 for timing
    new_value = max(3, new_value)
    
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
        numbers = [640, 480, 1, 10]  # VGA resolution, merged, 10 frames
    
    # Ensure we have at least 4 parameters
    while len(numbers) < 4:
        if len(numbers) == 0:
            numbers.append(640)   # width
        elif len(numbers) == 1:
            numbers.append(480)   # height
        elif len(numbers) == 2:
            numbers.append(1)     # merged
        else:
            numbers.append(10)    # repeat
    
    # Ensure width, height, repeat are positive; merged is 0 or 1
    numbers[0] = max(1, numbers[0])
    numbers[1] = max(1, numbers[1])
    numbers[2] = 1 if numbers[2] != 0 else 0
    numbers[3] = max(3, numbers[3])  # Minimum 3 for timing
    
    # Decide which parameter to mutate
    mutation_choice = random.random()
    
    if mutation_choice < 0.3:
        # Mutate width (30% probability)
        numbers[0] = mutate_width(numbers[0])
        # Adjust repeat to keep total pixels reasonable
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])
    
    elif mutation_choice < 0.55:
        # Mutate height (25% probability)
        numbers[1] = mutate_height(numbers[1])
        # Adjust repeat to keep total pixels reasonable
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])
    
    elif mutation_choice < 0.70:
        # Mutate merged flag (15% probability)
        numbers[2] = mutate_merged(numbers[2])
    
    else:
        # Mutate repeat (30% probability)
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])
    
    # Sometimes mutate multiple parameters (20% chance)
    if random.random() < 0.2:
        numbers[0] = mutate_width(numbers[0])
        numbers[1] = mutate_height(numbers[1])
        numbers[2] = mutate_merged(numbers[2])
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])
    
    # Sometimes use common resolutions (10% chance)
    if random.random() < 0.1:
        width, height = random.choice(COMMON_RESOLUTIONS)
        numbers[0] = width
        numbers[1] = height
        numbers[3] = mutate_repeat(numbers[3], numbers[0], numbers[1])
    
    # Final safety checks
    numbers[0] = max(1, min(numbers[0], MAX_SAFE_WIDTH))
    numbers[1] = max(1, min(numbers[1], MAX_SAFE_HEIGHT))
    numbers[2] = 1 if numbers[2] != 0 else 0
    
    # Re-check repeat with final width and height
    total_pixels = calculate_total_pixels(numbers[0], numbers[1], numbers[3])
    if total_pixels > MAX_TOTAL_PIXELS:
        imgSize = numbers[0] * numbers[1]
        safe_repeat = max(3, MAX_TOTAL_PIXELS // imgSize)
        numbers[3] = min(numbers[3], safe_repeat)
    
    numbers[3] = max(3, min(numbers[3], MAX_SAFE_REPEAT))
    
    # Keep only first four parameters
    numbers = numbers[:4]
    
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
