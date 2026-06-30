#====================================================================
# Author: Zheming Jin (jinz@ornl.gov). Edited
#====================================================================

import os
import re
import shutil

def replace_file(src, dst):
    if os.path.exists(dst):
        print(f"File restore from {dst} to {src}")
        shutil.copy(dst, src) 
    else:
        print(f"File copy from {src} to {dst}")
        shutil.copy(src, dst) 

def replace_function(file_path, function_name, function_code):
    """
    Replaces a function in a file

    Args:
        file_path (str): Path to the file.
        function_name (str): Name of the function to replace.
        function_code (str): The function code under test.
    """
    # make a copy for restore 
    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        print(f"Creating backup: {backup_path}")
        shutil.copy(file_path, backup_path)

    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except Exception as e:
        print(f"Failed to read the file: {file_path}: {e}")
        return False

    pattern = "// begin of {0}(.*?)// end of {0}".format(function_name)
    match = re.search(pattern, content, re.M | re.S)

    print("============= pattern match ===================")
    print(match)
    print("============= pattern match ===================")
    if match:
         # return the string obtained by replacing the leftmost non-overlapping
         # occurrences of pattern in string by the function_code

         # =================== START OF CHANGE ===================
         # The original re.sub call would re-process escape sequences in `function_code`,
         # causing issues with strings inside the code (e.g., turning '\\n' into a newline).
         # new_content = re.sub(pattern, r"\n" + function_code + r"\n", content, 1, re.DOTALL)

         # By using a lambda for the replacement, re.sub inserts the text literally
         # without processing backslash escapes within the 'function_code' string.
         new_content = re.sub(pattern, lambda m: f"\n{function_code}\n", content, 1, re.DOTALL)
         # =================== END OF CHANGE ===================

         try:
             with open(file_path, 'w') as file:
                 file.write(new_content)
             print(f"Function '{function_name}' replaced successfully in '{file_path}'.")
             return True
         except Exception as e:
             print(f"Error writing to {file_path}: {e}")
             return False
    else:
        print(f"Function '{function_name}' not found in '{file_path}'.")
        return False
