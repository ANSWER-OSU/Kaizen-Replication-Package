#include "RunBear.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>
#include <memory>
#include <array>

/*
* Executes a command and captures its output.
* Returns the output as a string, or empty string on failure.
*/
std::string executeCommand(const std::string &command) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    
    if (!pipe) {
        return "";
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    
    return result;
}

/*
* Parses make verbose output to extract compilation commands.
* Returns a vector of compilation command strings.
*/
std::vector<std::string> parseMakeOutput(const std::string &make_output, const std::string &build_dir) {
    std::vector<std::string> commands;
    std::istringstream stream(make_output);
    std::string line;
    
    // Regex patterns to match compilation commands
    std::regex nvcc_pattern(R"(^\s*(nvcc|/usr/local/cuda/bin/nvcc)\s+.*\.(cu|cpp|c)\s*)");
    std::regex gcc_pattern(R"(^\s*(gcc|g\+\+|clang|clang\+\+)\s+.*\.(cu|cpp|c|cc|cxx)\s*)");
    
    while (std::getline(stream, line)) {
        // Skip empty lines and common make output
        if (line.empty() || line.find("make[") != std::string::npos || 
            line.find("Entering directory") != std::string::npos ||
            line.find("Leaving directory") != std::string::npos) {
            continue;
        }
        
        // Check if line contains a compilation command
        if (std::regex_search(line, nvcc_pattern) || std::regex_search(line, gcc_pattern)) {
            // Clean up the line (remove leading/trailing whitespace)
            size_t start = line.find_first_not_of(" \t");
            size_t end = line.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                std::string clean_command = line.substr(start, end - start + 1);
                commands.push_back(clean_command);
                llvm::outs() << "Found compilation command: " << clean_command << "\n";
            }
        }
    }
    
    return commands;
}

/*
* Generates compile_commands.json from a vector of compilation commands.
* Returns true on success, else false.
*/
bool writeCompileCommandsJson(const std::vector<std::string> &commands, const std::string &build_dir) {
    std::string json_path = build_dir + "/compile_commands.json";
    std::ofstream json_file(json_path);
    
    if (!json_file.is_open()) {
        llvm::errs() << "Error: Failed to create compile_commands.json at " << json_path << "\n";
        return false;
    }
    
    json_file << "[\n";
    
    for (size_t i = 0; i < commands.size(); ++i) {
        const std::string &command = commands[i];
        
        // Parse the command to extract file and directory
        std::istringstream cmd_stream(command);
        std::string token;
        std::string source_file;
        std::string compiler;
        
        // Get the compiler (first token)
        if (cmd_stream >> compiler) {
            // Look for source file (ends with .cu, .cpp, .c, etc.)
            while (cmd_stream >> token) {
                if (token.find(".cu") != std::string::npos || 
                    token.find(".cpp") != std::string::npos ||
                    token.find(".c") != std::string::npos ||
                    token.find(".cc") != std::string::npos ||
                    token.find(".cxx") != std::string::npos) {
                    source_file = token;
                    break;
                }
            }
        }
        
        // If we couldn't find a source file, skip this command
        if (source_file.empty()) {
            continue;
        }
        
        // Convert relative path to absolute if needed
        std::string abs_source_file;
        if (source_file[0] != '/') {
            abs_source_file = build_dir + "/" + source_file;
        } else {
            abs_source_file = source_file;
        }
        
        // Write JSON entry
        json_file << "  {\n";
        json_file << "    \"directory\": \"" << build_dir << "\",\n";
        json_file << "    \"command\": \"" << command << "\",\n";
        json_file << "    \"file\": \"" << abs_source_file << "\"\n";
        json_file << "  }";
        
        if (i < commands.size() - 1) {
            json_file << ",";
        }
        json_file << "\n";
    }
    
    json_file << "]\n";
    json_file.close();
    
    llvm::outs() << "Generated compile_commands.json with " << commands.size() << " entries.\n";
    return true;
}

/*
* Runs 'make' with verbose flags in the given directory to generate compile_commands.json.
* Returns true on success, else false. Exits in same directory as start.
*/
bool generateCompileCommands(const std::string &build_dir) {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == 0) {
        llvm::errs() << "Error: Failed to get current working directory: " << std::strerror(errno) << "\n";
        return false;
    }
    
    if (chdir(build_dir.c_str()) != 0) {
        llvm::errs() << "Error: Failed to change directory to \"" << build_dir << "\": " << std::strerror(errno) << "\n";
        return false;
    }
    
    llvm::outs() << "Running \"make clean && make VERBOSE=1\" (" << build_dir << ")\n";
    
    // First clean the build
    int clean_ret = system("make clean >/dev/null 2>&1");
    if (clean_ret != 0) {
        llvm::outs() << "Warning: make clean failed, continuing anyway.\n";
    }
    
    // Run make with verbose output and capture it
    std::string make_output = executeCommand("make VERBOSE=1 2>&1");
    
    // Change back to original directory
    chdir(cwd);
    
    if (make_output.empty()) {
        llvm::errs() << "Error: Failed to capture make output or make failed.\n";
        return false;
    }
    
    // Parse the make output to extract compilation commands
    std::vector<std::string> commands = parseMakeOutput(make_output, build_dir);
    
    if (commands.empty()) {
        llvm::errs() << "Error: No compilation commands found in make output.\n";
        return false;
    }
    
    // Generate compile_commands.json
    return writeCompileCommandsJson(commands, build_dir);
}
