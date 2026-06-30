"""
This python file provides functions to perform custom mutation of AFL++ to fuzz space-separated number inputs.

This fuzzer is designed to work with inputs that are space-separated lists of numbers like:
- "5 30 12"
- "7 1" 
- "10"

The fuzzer applies three types of mutations:
1. Off by N (1-5): adds or subtracts 1,2,3,4, or 5 from a number
2. Flip sign: makes negative if positive and positive if negative  
3. Min/Max: sets to integer min/max (C++ int limits)

Configuration:
- POSITIVE_ONLY: When True, keeps all numbers >= 0 during mutation

Before running this file, set up:
1. export PYTHONPATH=$(dirname /path/to/custom-fuzzer.py)
2. export AFL_PYTHON_MODULE=custom-fuzzer
3. export AFL_CUSTOM_MUTATOR_ONLY=1
4. export AFL_DISABLE_TRIM=1
5. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform
import sys
import os
import re
import math

# Default C++ int limits (typically 32-bit signed integers)
DEFAULT_INT_MIN = -2147483648
DEFAULT_INT_MAX = 2147483647

# Global variables for min/max values (will be loaded from files)
INT_MIN = DEFAULT_INT_MIN
INT_MAX = DEFAULT_INT_MAX

# Configuration constants
POSITIVE_ONLY = False  # Set to True to keep all numbers >= 0 during mutation

# Global variables
node = ""
lines_initialized = False
min_max_loaded = False

def load_min_max_values():
    """Load min and max values from custom-fuzzer-setup files"""
    global INT_MIN, INT_MAX, min_max_loaded
    
    if min_max_loaded:
        return  # Already loaded
    
    # Look for custom-fuzzer-setup directory relative to current working directory
    setup_dir = "custom-fuzzer-setup"
    min_file = os.path.join(setup_dir, "min.txt")
    max_file = os.path.join(setup_dir, "max.txt")
    
    # Extract min values
    try:
        if os.path.exists(min_file):
            with open(min_file, 'r') as f:
                min_params = f.read().strip()
            # Parse the first number from the min parameters
            if min_params:
                min_numbers = [int(x) for x in min_params.split() if x.lstrip('-').isdigit()]
                if min_numbers:
                    INT_MIN = min(min_numbers)
                    print(f"Custom fuzzer: Loaded INT_MIN = {INT_MIN} from {min_file}")
    except Exception as e:
        print(f"Custom fuzzer: Failed to load min values from {min_file}: {e}")
        print(f"Custom fuzzer: Using default INT_MIN = {DEFAULT_INT_MIN}")
    
    # Extract max values
    try:
        if os.path.exists(max_file):
            with open(max_file, 'r') as f:
                max_params = f.read().strip()
            # Parse the first number from the max parameters  
            if max_params:
                max_numbers = [int(x) for x in max_params.split() if x.lstrip('-').isdigit()]
                if max_numbers:
                    INT_MAX = max(max_numbers)
                    print(f"Custom fuzzer: Loaded INT_MAX = {INT_MAX} from {max_file}")
    except Exception as e:
        print(f"Custom fuzzer: Failed to load max values from {max_file}: {e}")
        print(f"Custom fuzzer: Using default INT_MAX = {DEFAULT_INT_MAX}")
    
    min_max_loaded = True
    print(f"Custom fuzzer: Using INT_MIN = {INT_MIN}, INT_MAX = {INT_MAX}")

class NumberInput:
    def __init__(self, input_string, min_list=None, max_list=None):
        self.original_string = input_string.strip()
        self.numbers = self.parse_numbers(input_string)
        # Optional per-position bounds
        self.min_list = min_list or []
        self.max_list = max_list or []
    
    def parse_numbers(self, input_string):
        """Parse space-separated whole numbers from input string"""
        try:
            parts = input_string.strip().split()
            numbers = []
            for part in parts:
                # Only accept whole numbers (integers)
                try:
                    # Check if it's a valid integer
                    num = int(part)
                    numbers.append(num)
                except ValueError:
                    # If any part is not a whole number, return empty list
                    return []
            return numbers
        except:
            return []
    
    def mutate(self):
        """Apply one of three mutation strategies to a random whole number"""
        # If no valid whole numbers found, fail mutation
        if not self.numbers:
            return self.original_string
        
        # Choose a random number from the list to mutate
        mutate_index = random.randint(0, len(self.numbers) - 1)
        original_value = self.numbers[mutate_index]
        
        # All numbers should be integers at this point due to parse_numbers
        # Choose mutation strategy randomly
        mutation_type = random.randint(1, 9)
        
        if mutation_type < 10:
            # Off by N 5% rounded outwards
            offset = math.ceil(original_value * 0.05)
            if random.choice([True, False]):  # randomly choose add or subtract
                new_value = original_value + offset
            else:
                new_value = original_value - offset
            
            # Ensure non-negative if POSITIVE_ONLY is enabled
            if POSITIVE_ONLY and new_value < 0:
                new_value = 0
        elif mutation_type == 10:
            # Min/Max: sets to integer min or max
            # Prefer per-position bounds if available
            if mutate_index < len(self.min_list) and mutate_index < len(self.max_list):
                lo = self.min_list[mutate_index]
                hi = self.max_list[mutate_index]
                if lo > hi:
                    lo, hi = hi, lo
                new_value = random.choice([lo, hi])
            else:
                if POSITIVE_ONLY:
                    # Use 0 and INT_MAX instead of INT_MIN and INT_MAX
                    new_value = random.choice([0, INT_MAX])
                else:
                    new_value = random.choice([INT_MIN, INT_MAX])
        
        # Clamp to per-position bounds if provided
        if mutate_index < len(self.min_list) and mutate_index < len(self.max_list):
            lo = self.min_list[mutate_index]
            hi = self.max_list[mutate_index]
            if lo > hi:
                lo, hi = hi, lo
            if new_value < lo:
                new_value = lo
            elif new_value > hi:
                new_value = hi
        
        # Create new numbers list with the mutated value
        new_numbers = self.numbers.copy()
        new_numbers[mutate_index] = new_value
        
        # Convert back to string format (all integers)
        result_parts = [str(num) for num in new_numbers]
        
        return " ".join(result_parts)



'''
AFL++ will call this function once it starts up
We set the seed and initialize basic state
'''
def init(seed):
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    # Load min/max values from files
    load_min_max_values()

'''We need this function defined, but there's nothing to do on fuzzer shutdown for now'''
def deinit():
    pass

'''
Runs a single iteration of the fuzzer.
Takes input buffer, applies number-based mutations, returns mutated input
'''
def fuzz(buf, add_buf, max_size):
    global lines_initialized
    
    # Decode input buffer to string
    try:
        buf_str = buf.decode('utf-8', errors='ignore').strip()
    except:
        buf_str = str(buf).strip()
    
    # If empty input, return something basic
    if not buf_str:
        buf_str = "1"

    # Try to parse combined seed/min/max format: '<seed> && : <min> && : <max>'
    seed_list = None
    min_list = None
    max_list = None
    parts = re.split(r"\s*&&\s*:\s*", buf_str)
    if len(parts) == 3:
        seed_str, min_str, max_str = parts
        def to_ints(s):
            out = []
            for tok in s.strip().split():
                try:
                    out.append(int(tok))
                except ValueError:
                    # ignore non-integers in the sequence
                    pass
            return out
        seed_list = to_ints(seed_str)
        min_list = to_ints(min_str)
        max_list = to_ints(max_str)
        # Fall back to global defaults if min/max lengths don't match
        if not seed_list:
            seed_list = [1]
        # Build a NumberInput with per-position bounds when available
        ni_input_str = " ".join(str(x) for x in seed_list)
        number_input = NumberInput(ni_input_str, min_list=min_list, max_list=max_list)
        mutated_string = number_input.mutate()
        # Re-emit in the same combined format with original min/max preserved
        mutated_string = f"{mutated_string} && : {' '.join(map(str, min_list))} && : {' '.join(map(str, max_list))}"
    else:
        # Standard behavior (no embedded min/max)
        number_input = NumberInput(buf_str)
        mutated_string = number_input.mutate()
    
    # Ensure we don't exceed max_size
    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]
    
    # Return as bytearray
    return bytearray(mutated_string.encode('utf-8'))
