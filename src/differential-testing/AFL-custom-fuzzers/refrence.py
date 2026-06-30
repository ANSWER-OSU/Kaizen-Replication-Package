"""
This python file provides functions to perform custom mutation of AFL++ to fuzz the LAMMPS input file.

Before run this file, you need to set up:
1. Put the cmd_dict.json and custom.py in the same folder
2. export PYTHONPATH=$(dirname /path/to/custom.py)
3. export AFL_PYTHON_MODULE=custom
4. export AFL_CUSTOM_MUTATOR_ONLY=1
5. export AFL_DISABLE_TRIM=1
6. export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1

An example of how to run this file:
/nfs/hpc/share/motwanim/software/AFLplusplus/afl-fuzz -d -t 30000 -i input_cases -o output_cases -- /nfs/hpc/share/motwanim/software/HPCTest-manish/HPCTest/src/differential-driver/build_matthew/run_subsystem_compare_logs -pk kokkos newton on neigh half -k on t 1 -sf kk -in @@ -l log.lammps.openmp

"""

import json
import os
import re
import random
import platform

import combinedeventanalyzer

# Configuration (set environment variable DIRECTOR_ENABLED=true)
DIRECTOR_ENABLED = os.getenv('DIRECTOR_ENABLED') is not None and os.getenv('DIRECTOR_ENABLED').lower() == 'true'

command_dict = {} # load json file into dictionary
input_lines = [] # The lines of the input file
editable_lines = [] # The lines which contain editable commands
previous_director_metrics = None
previous_metric_product = None
previous_signal = None
previously_mutated_line_index = None
lines_initialized = False
node = ""

class EditableLine:
    def __init__(self, i, cmd):
       self.index = i
       self.command = cmd
    
    def getIndex(self):
        return self.index
    def getCommand(self):
        return self.command
    def getProbability(self):
        return self.probability
    def setProbability(self, prob):
        self.probability = prob
    
    def mutate(self):
        global input_lines
        parts = input_lines[self.index].split()
        new_parts = process_command_config(command_dict[self.command], parts, random.random() < 0.5)
        return new_parts
    
    def adjust_probability(self, metric_score):
        self.probability += metric_score * self.probability
        
    def __str__(self):
        return f"Editable Command: [{self.getCommand()}] has weight {self.getProbability()} and is line {self.getIndex()}"
    def __repr__(self):
        return self.__str__()
    
'''
AFL++ will call this function once it starts up
We set the seed and initialize the command dictionary
'''
def init(seed):
    global command_dict
    global node
    global previous_director_metrics
    random.seed(seed)
    with open('/nfs/hpc/share/motwanim/software/HPCTest-manish/HPCTest/src/AFL_fuzz/cmd_dict.json', 'r') as file:
        command_dict = json.load(file)
    node = platform.node().split(".")[0]

'''We need this function defined, but there's nothing to do on fuzzer shutdown for now'''
def deinit():
    pass

def get_director_report():
    global node
    metric_values=[]
   
    events_files = [fileName for fileName in os.listdir('.') if fileName.endswith(".combined_events")]

    if len(events_files) == 0:
        print("Director failed to generate/analyze events")
        return previous_director_metrics

    metric_values = combinedeventanalyzer.getMetrics(events_files[0])

    for fileName in events_files:
        os.remove('./' + fileName)

    return metric_values


''' 
the method to mutate the existing cmd values using
the cmd_config dict. This works recursively.
ASSUMPTIONS: 
    - we only mutate values of style_args and key_values and not the arguments/keys
'''
def fuzz_value(config, originalValue):  
    if(len(config)==0):
        return ""
    elif "No change" in config:
        return originalValue
    elif (config[0]=="float") or (config==float):
        return str(round(random.uniform(0, 10), 2))
    elif (config[0]=="integer") or (config=="integer"):
        return str(random.randint(0, 10))
    else:   
        return str(random.choice(config))

def process_command_config(command_config, parts, isRandom = True):
    # updated_parts = [parts[0]] // resulted in IndexError, cause the there was empty list given to this by the mutate function
    try: 
        updated_parts = [parts[0]]
    except IndexError:
        return ""
    i=1
    if("key-values" in command_config and len(command_config.keys())==1):   
        return parts
    if "style_arg" in command_config:
        style_keys = list(command_config["style_arg"].keys())
        if(isRandom):
            i=1
            while(i< len(parts)):
                if("arg_"+str(i) in command_config):
                    updated_parts.append(parts[i])
                i+=1
            chosen_style = random.choice(style_keys)
        else:
            i = 1
            while(i < len(parts)):
                if("arg_"+str(i) in command_config):
                    updated_parts.append(parts[i])
                else:
                    chosen_style = parts[i]
                    break
                i+=1
            
        updated_parts.append(chosen_style)
        
        style_args = None

        try:
            style_args = command_config[chosen_style]["style_arg"]
        except KeyError:
                try:
                    style_args = command_config["style_arg"][chosen_style]
                except KeyError:
                    pass
        
        if isinstance(style_args, dict):
            for key, value in style_args.items():
                if isinstance(value, list):
                    updated_parts.append(fuzz_value(value, "1.0"))
        elif isinstance(style_args, list):
            updated_parts.append(fuzz_value(style_args, "1.0"))
    else:
        
        for i in range(1, min(len(parts), 4)): 
            arg_key = f"arg_{i}"
            if arg_key in command_config:
                
                updated_parts.append(fuzz_value(command_config[arg_key], parts[i]))

    return updated_parts

'''Merge commands which fall on multiple lines into a single line'''
def merge_multiline_input(lines):
    merged_lines = []
    current_command = []
    for line in lines:
        line = line.strip()
        if line.endswith('&'):
            # Remove '&' and add to the current command
            current_command.append(line[:-1].strip())
        else:
            # Add the current line to the command and reset
            current_command.append(line)
            merged_lines.append(' '.join(current_command))
            current_command = []
    # Handle case where the last line ends with '&'
    if current_command:
        merged_lines.append(' '.join(current_command))
    return merged_lines

def initialize_lines(lines):
    global editable_lines
    # Parse and check lines:
    # If they match commands we can edit, add them to the editable_lines list
    f = open("FuzzingLog.txt", "a")
    f.write("--------------------------Initialization--------------------------\n")
    for i in range(len(lines)):
        line = lines[i]
        if not line.startswith('#') and not line.startswith('thermo') and not line.startswith('run')  and len(line.strip()) > 1: # If not a comment
            if '#' in line:
                lines[i] = lines[i].split('#')[0] #Trim off inline comment
                line = lines[i]
            parts = line.split()
            if parts[0] in command_dict.keys():
                
                editableLine = EditableLine(i, parts[0])
                editable_lines.append(editableLine)
                f.write(f"{lines[i]} : EDITABLE\n")
                continue
        f.write(f"{lines[i]} : NOT EDITABLE\n")
    f.close()
    # Set all of the initial probabilities to 1/n
    for editable_line in editable_lines:
        editable_line.setProbability(1/len(editable_lines))

    # Delete dry run events files
    if DIRECTOR_ENABLED:
        events_files = [fileName for fileName in os.listdir('.') if fileName.endswith(".combined_events")]
        for fileName in events_files:
            os.remove('./' + fileName)

def dump_iteration_debug_info(metrics,diff, before, after):
    sep = '\n\t'
    with open('FuzzingLog.txt','a') as f:
        f.write("-"*20+"\n")
        if DIRECTOR_ENABLED:
            f.write("Metrics from Previous Iteration:\n")
            f.write(f"{metrics}\n\n")
            f.write(f"Signal Difference: {diff}\n\n")
        
        f.write(f"Edit Made This Iteration:\n{before} -> {after}\n\n")
        f.write("Editable Lines:\n\t")
        f.write(f"{sep.join([str(line) for line in editable_lines])}\n\n\n")

        


'''
Runs a single iteration of the fuzzer.
Calculates the director scores and adjust probabilities of commands (if enabled)
Then generate a new input and save state for next iteration
'''
def fuzz(buf, add_buf, max_size):
    global editable_lines
    global previously_mutated_line_index
    global previous_director_metrics
    global previous_metric_product
    global input_lines
    global lines_initialized
 
    buf_str = buf.decode('utf-8', errors='ignore')
    lines = merge_multiline_input(buf_str.split('\n'))
    input_lines = lines
    
    if not lines_initialized:
        initialize_lines(lines)
        lines_initialized = True

    signal_difference = None
    if DIRECTOR_ENABLED:
        director_metrics = get_director_report()
        if director_metrics is not None:

            #Calculate metric product
            metric_product = 1.0
            epsilon_adjustment = 0.00001 # makes it so we do not multiply by 0. (all values are positive)
            for metric in director_metrics:
                metric_product *= epsilon_adjustment + director_metrics[metric]

            if previous_director_metrics is not None:
                # Calculating the signal difference:                
                signal_difference = metric_product - previous_metric_product
                
                # Calculating the signal sign:
                max_values_across_seed_inputs={
                    'Mem Leak Discrepancies': 0.2631578947368421, 
                    'Num Parallel For': 87786, 
                    'Num Parallel Scan': 206, 
                    'Num Parallel Reduce': 12722, 
                    'Num Fences': 611640, 
                    'Num Deep Copies': 13755
                }
                adjusted_difference_sum = 0.0
                for metric in director_metrics:
                    if metric != 'Mem Leak Discrepancies':
                        adjusted_difference_sum += (director_metrics[metric] - previous_director_metrics[metric]) / max_values_across_seed_inputs[metric]
                if adjusted_difference_sum < 0:
                    signal_difference *= -1

                # FIXME: I'm not so sure about this.
                signal_difference /= (metric_product + previous_metric_product)

                editable_lines[previously_mutated_line_index].adjust_probability(signal_difference)
                
                # Normalize probabilities to reflect change
                total = sum([line.getProbability() for line in editable_lines])
                for line in editable_lines:
                    line.setProbability(line.getProbability() / total)
            
            previous_director_metrics = director_metrics
            previous_metric_product = metric_product
        

    indices, line_mutate = zip(*[(i, line) for i, line in enumerate(editable_lines)])
    
    line_to_mutate_index = random.choices(indices, weights=[command.getProbability() for command in editable_lines])[0]
    
    line_to_mutate = editable_lines[line_to_mutate_index]
    
    debug_before_print_line = input_lines[line_to_mutate.index]

    new_parts = line_to_mutate.mutate()
    input_lines[line_to_mutate.index] = " ".join(new_parts)

    debug_after_print_line = input_lines[line_to_mutate.index]

    mutated_input = '\n'.join(input_lines)
    previously_mutated_line_index = line_to_mutate_index

    # Debug Info Printouts
    dump_iteration_debug_info(director_metrics, signal_difference,debug_before_print_line, debug_after_print_line)


    if len(mutated_input) > max_size:
        mutated_input = mutated_input[:max_size]

    return bytearray(mutated_input, 'utf-8')