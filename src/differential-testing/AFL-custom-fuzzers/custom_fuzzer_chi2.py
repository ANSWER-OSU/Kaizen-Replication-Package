"""
Custom AFL++ fuzzer for SNP Chi-Square Association Test (gwas/snp-omp) testing.

This fuzzer is designed to test the correctness of SNP (Single Nucleotide Polymorphism)
chi-square association calculations by generating valid integer inputs for all 6 parameters.

The program being tested: ./driver <rows> <cols> <cases> <controls> <threads> <repeat>
- rows:     total number of individuals (cases + controls combined, must be >= cases + controls)
- cols:     number of SNPs to analyze (must be positive)
- cases:    number of case individuals (must be positive, < rows)
- controls: number of control individuals (must be positive, cases + controls <= rows)
- threads:  number of OpenMP threads (must be positive)
- repeat:   number of kernel repetitions for timing (must be positive)

Key insights:
- Data matrix is rows * cols bytes: rows=individuals, cols=SNPs
- Memory allocation: rows * cols (data) + 2 * cols * floats (results)
- cases + controls must NOT exceed rows (would cause out-of-bounds reads)
- Each SNP column is processed independently: O(rows) per SNP, O(rows * cols) total
- Chi-square computed per SNP: 3 genotype bins (0,1,2) x 2 groups (case/control)
- threads maps to OMP thread_limit; valid range 1-1024 typically
- repeat only affects timing loop, not correctness; keep low (1-10) to avoid timeouts

Mutation strategy focuses on:
1. Correct constraint: cases + controls <= rows (enforced on every mutation)
2. Various SNP counts (cols): small to large
3. Case/control balance: balanced, skewed, extreme minority ratios
4. Thread counts: 1, 32, 64, 128, 256, 512
5. Timeout protection: total work = rows * cols * repeat

Setup:
1. Rename this file to: custom_fuzzer_snp.py (underscores, not hyphens)
2. export PYTHONPATH="/path/to/directory/containing/this/file"
3. export AFL_CUSTOM_MUTATOR_LIBRARY="/path/to/AFLplusplus/custom_mutators/python/python_mutator.so"
4. export AFL_PYTHON_MODULE="custom_fuzzer_chi2"
5. export AFL_CUSTOM_MUTATOR_ONLY=1
6. export AFL_DISABLE_TRIM=1
7. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
"""

import random
import platform

# ---------------------------------------------------------------------------
# Safety / timeout limits
# Total work ≈ rows * cols * repeat (data bytes processed * repeats)
# ---------------------------------------------------------------------------
MAX_ROWS        = 100000    # max total individuals
MAX_COLS        = 100000    # max SNPs
MAX_CASES       = 50000
MAX_CONTROLS    = 50000
MAX_THREADS     = 1024
MAX_REPEAT      = 10        # keep low; repeat only affects timing
MAX_TOTAL_WORK  = 500000000 # 500M byte-operations before throttle

# Global state
node = ""

# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

def init(seed):
    global node
    random.seed(seed)
    node = platform.node().split(".")[0]
    print(f"Custom SNP fuzzer initialized on {node} with seed {seed}")
    print(f"Limits: rows<={MAX_ROWS}, cols<={MAX_COLS}, cases<={MAX_CASES}, "
          f"controls<={MAX_CONTROLS}, threads<={MAX_THREADS}, repeat<={MAX_REPEAT}")
    print(f"Max total work (rows*cols*repeat): {MAX_TOTAL_WORK}")

def deinit():
    pass

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_input(buf_str):
    """Parse up to 6 space-separated integers from input string."""
    try:
        parts = buf_str.strip().split()
        numbers = []
        for part in parts:
            try:
                numbers.append(int(part))
            except ValueError:
                pass
        return numbers
    except Exception:
        return []

def calculate_total_work(rows, cols, repeat):
    """Approximate total work: data size * repeat."""
    try:
        return rows * cols * repeat
    except Exception:
        return MAX_TOTAL_WORK + 1

def safe_repeat(rows, cols, desired_repeat):
    """Cap repeat so total work stays under MAX_TOTAL_WORK."""
    data_size = rows * cols
    if data_size == 0:
        return 1
    cap = max(1, MAX_TOTAL_WORK // data_size)
    return min(desired_repeat, cap, MAX_REPEAT)

# ---------------------------------------------------------------------------
# Per-parameter mutators
# ---------------------------------------------------------------------------

COMMON_THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def mutate_rows(original):
    t = random.randint(1, 10)
    if t <= 2:
        v = random.choice([2, 4, 8, 10, 16])
    elif t <= 4:
        v = random.choice([32, 64, 100, 128, 256])
    elif t <= 6:
        v = random.choice([512, 1000, 1024, 2048, 5000])
    elif t <= 8:
        v = random.choice([10000, 20000, 50000, 100000])
    elif t == 9:
        exp = random.randint(1, 16)
        v = 2 ** exp
    else:
        offset = random.choice([1, 10, 100, 1000])
        v = max(2, original + random.choice([-offset, offset]))
    return max(2, min(v, MAX_ROWS))


def mutate_cols(original):
    t = random.randint(1, 10)
    if t <= 2:
        v = random.choice([1, 2, 4, 8, 16])
    elif t <= 4:
        v = random.choice([32, 64, 100, 128, 256])
    elif t <= 6:
        v = random.choice([512, 1000, 1024, 5000, 10000])
    elif t <= 8:
        v = random.choice([20000, 50000, 100000])
    elif t == 9:
        exp = random.randint(0, 16)
        v = 2 ** exp
    else:
        offset = random.choice([1, 10, 100, 1000])
        v = max(1, original + random.choice([-offset, offset]))
    return max(1, min(v, MAX_COLS))


def mutate_cases(original, rows):
    """Mutate cases, keeping it < rows and leaving room for at least 1 control."""
    t = random.randint(1, 10)
    max_cases = rows - 1  # must leave room for at least 1 control
    if t <= 2:
        v = random.choice([1, 2, 5, 10])
    elif t <= 4:
        v = random.choice([16, 32, 50, 64, 100])
    elif t <= 6:
        # roughly half of rows (balanced design)
        v = max(1, rows // 2 + random.randint(-rows // 10, rows // 10))
    elif t <= 8:
        # skewed: minority cases
        v = max(1, rows // 10)
    elif t == 9:
        # skewed: majority cases
        v = max(1, int(rows * 0.9))
    else:
        offset = random.choice([1, 10, 100])
        v = max(1, original + random.choice([-offset, offset]))
    return max(1, min(v, max_cases, MAX_CASES))


def mutate_controls(original, rows, cases):
    """Mutate controls so that cases + controls <= rows."""
    max_controls = rows - cases
    if max_controls < 1:
        return 1
    t = random.randint(1, 10)
    if t <= 2:
        v = random.choice([1, 2, 5, 10])
    elif t <= 4:
        v = random.choice([16, 32, 50, 64, 100])
    elif t <= 6:
        # balanced with cases
        v = cases + random.randint(-cases // 2, cases // 2)
    elif t <= 8:
        # use nearly all remaining rows
        v = max_controls - random.randint(0, max(0, max_controls // 10))
    elif t == 9:
        # very few controls
        v = random.randint(1, max(1, max_controls // 10))
    else:
        offset = random.choice([1, 10, 100])
        v = max(1, original + random.choice([-offset, offset]))
    return max(1, min(v, max_controls, MAX_CONTROLS))


def mutate_threads(original):
    t = random.randint(1, 5)
    if t <= 2:
        v = random.choice(COMMON_THREAD_COUNTS)
    elif t <= 4:
        exp = random.randint(0, 10)
        v = 2 ** exp
    else:
        offset = random.choice([1, 2, 4, 8, 16])
        v = max(1, original + random.choice([-offset, offset]))
    return max(1, min(v, MAX_THREADS))


def mutate_repeat(original, rows, cols):
    t = random.randint(1, 5)
    if t <= 2:
        v = random.choice([1, 2, 3, 5])
    elif t <= 4:
        v = random.choice([1, 5, 10])
    else:
        offset = random.choice([1, 2])
        v = max(1, original + random.choice([-offset, offset]))
    v = max(1, v)
    return safe_repeat(rows, cols, v)

# ---------------------------------------------------------------------------
# Enforce constraints and produce a valid 6-tuple
# ---------------------------------------------------------------------------

def enforce_constraints(nums):
    """
    Given a list of up to 6 ints, clamp and fix all cross-parameter constraints.
    Returns [rows, cols, cases, controls, threads, repeat].
    """
    while len(nums) < 6:
        defaults = [1000, 512, 400, 500, 64, 1]
        nums.append(defaults[len(nums)])

    rows, cols, cases, controls, threads, repeat = nums[:6]

    # Clamp to global bounds first
    rows     = max(2,  min(rows,     MAX_ROWS))
    cols     = max(1,  min(cols,     MAX_COLS))
    cases    = max(1,  min(cases,    MAX_CASES,    rows - 1))
    controls = max(1,  min(controls, MAX_CONTROLS, rows - cases))
    threads  = max(1,  min(threads,  MAX_THREADS))
    repeat   = max(1,  min(repeat,   MAX_REPEAT))

    # Throttle repeat based on total work
    repeat = safe_repeat(rows, cols, repeat)

    return [rows, cols, cases, controls, threads, repeat]

# ---------------------------------------------------------------------------
# Main AFL++ fuzz entry point
# ---------------------------------------------------------------------------

def fuzz(buf, add_buf, max_size):
    """
    Main fuzzing function called by AFL++.

    Args:
        buf:      input buffer (bytes)
        add_buf:  additional buffer for splicing (unused here)
        max_size: maximum output size

    Returns:
        bytearray: mutated input
    """
    try:
        buf_str = buf.decode('utf-8', errors='ignore').strip()
    except Exception:
        buf_str = ""

    numbers = parse_input(buf_str)

    # Fall back to safe defaults if parsing failed
    if len(numbers) < 6:
        numbers = [1000, 512, 400, 500, 64, 1]

    numbers = enforce_constraints(numbers)
    rows, cols, cases, controls, threads, repeat = numbers

    # Choose a mutation strategy
    strategy = random.random()

    if strategy < 0.20:
        # Mutate rows; re-derive cases/controls to stay valid
        rows     = mutate_rows(rows)
        cases    = mutate_cases(cases, rows)
        controls = mutate_controls(controls, rows, cases)

    elif strategy < 0.35:
        # Mutate cols only (no cross-constraints with other params)
        cols = mutate_cols(cols)

    elif strategy < 0.55:
        # Mutate cases (controls must stay valid)
        cases    = mutate_cases(cases, rows)
        controls = mutate_controls(controls, rows, cases)

    elif strategy < 0.70:
        # Mutate controls only
        controls = mutate_controls(controls, rows, cases)

    elif strategy < 0.82:
        # Mutate thread count
        threads = mutate_threads(threads)

    elif strategy < 0.90:
        # Mutate repeat
        repeat = mutate_repeat(repeat, rows, cols)

    else:
        # Multi-parameter mutation (20% of time via this branch + earlier 20% overlap)
        rows     = mutate_rows(rows)
        cols     = mutate_cols(cols)
        cases    = mutate_cases(cases, rows)
        controls = mutate_controls(controls, rows, cases)
        threads  = mutate_threads(threads)
        repeat   = mutate_repeat(repeat, rows, cols)

    # Occasionally splice interesting corner cases (5% chance)
    if random.random() < 0.05:
        edge_cases = [
            [2,   1,    1,    1,    1,   1],   # absolute minimum
            [10,  10,   5,    5,    1,   1],   # tiny balanced
            [100, 100,  50,   50,   32,  1],   # small balanced, typical threads
            [1000, 1000, 500, 499, 64,  1],    # cases+controls = rows-1
            [1000, 1000, 999,  1,   64,  1],   # extreme case skew
            [1000, 1000,   1, 999,  64,  1],   # extreme control skew
            [100,  500, 50,   50,  128,  3],   # more SNPs than individuals
            [50000, 100, 25000, 24999, 256, 1], # large individuals, few SNPs
        ]
        numbers = random.choice(edge_cases)
        rows, cols, cases, controls, threads, repeat = enforce_constraints(numbers)

    # Re-enforce all constraints after all mutations
    numbers = enforce_constraints([rows, cols, cases, controls, threads, repeat])

    # Serialize
    mutated_string = " ".join(str(n) for n in numbers)

    # Respect max_size
    if len(mutated_string) > max_size:
        mutated_string = mutated_string[:max_size]

    return bytearray(mutated_string.encode('utf-8'))


def fuzz_count(buf):
    """Return number of mutations AFL++ should request per input. Keep at 1."""
    return 1
