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

# C++ int limits (typically 32-bit signed integers)
INT_MIN = -2147483648
INT_MAX = 2147483647

# Configuration constants
POSITIVE_ONLY = True  # Set to True to keep all numbers >= 0 during mutation

# Global variables
node = ""
lines_initialized = False

class NumberInput:
    def __init__(self, input_string):
        self.original_string = input_string.strip()
        self.numbers = self.parse_numbers(input_string)
    
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
        mutation_type = random.randint(1, 3)
        
        if mutation_type == 1:
            # Off by N (1-5): adds or subtracts 1,2,3,4, or 5
            offset = random.randint(1, 5)
            if random.choice([True, False]):  # randomly choose add or subtract
                new_value = original_value + offset
            else:
                new_value = original_value - offset
            
            # Ensure non-negative if POSITIVE_ONLY is enabled
            if POSITIVE_ONLY and new_value < 0:
                new_value = 0
                
        elif mutation_type == 2:
            # Flip sign: makes negative if positive and positive if negative
            new_value = -original_value
            
            # Ensure non-negative if POSITIVE_ONLY is enabled
            if POSITIVE_ONLY and new_value < 0:
                new_value = abs(original_value)  # Use absolute value instead
            
        elif mutation_type == 3:
            # Min/Max: sets to integer min or max
            if POSITIVE_ONLY:
                # Use 0 and INT_MAX instead of INT_MIN and INT_MAX
                new_value = random.choice([0, INT_MAX])
            else:
                new_value = random.choice([INT_MIN, INT_MAX])
        
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
    
    # Create NumberInput object and mutate
    number_input = NumberInput(buf_str)
    mutated_string = number_input.mutate()
    
    # Ensure we don't exceed max_size
    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]
    
    # Return as bytearray
    return bytearray(mutated_string.encode('utf-8'))

# Optional: Add some helper functions for testing
def test_mutations():
    """Test function to verify mutations work correctly"""
    test_inputs = [
        "5 30 12",      # Valid: all whole numbers
        "7 1",          # Valid: all whole numbers  
        "10",           # Valid: single whole number
        "-5 0 100",     # Valid: all whole numbers (including negative and zero)
        "10 abc 4",     # Invalid: contains non-number
        "1.5 2 3",      # Invalid: contains decimal
        "10 20 30.0",   # Invalid: contains decimal (even if .0)
        "abc def",      # Invalid: no numbers
        ""              # Invalid: empty
    ]
    
    random.seed(42)  # Fixed seed for reproducible testing
    
    print(f"Testing mutations (POSITIVE_ONLY = {POSITIVE_ONLY}):")
    for test_input in test_inputs:
        number_input = NumberInput(test_input)
        print(f"Input: '{test_input}'")
        print(f"  Parsed numbers: {number_input.numbers}")
        if number_input.numbers:
            for i in range(3):  # Test 3 mutations per valid input
                mutated = number_input.mutate()
                print(f"  Mutation {i+1}: {test_input} -> {mutated}")
        else:
            print(f"  Mutation: {test_input} -> {test_input} (no mutation - invalid input)")
        print()

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_mutations()