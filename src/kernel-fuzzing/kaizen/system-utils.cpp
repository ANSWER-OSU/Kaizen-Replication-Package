#include "system-utils.h"
#include "environment-config.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <random>
#include <iomanip>
#include <cstdio>

std::string SystemUtils::generateUUID() {
    // Simple UUID v4 generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    const char* chars = "0123456789abcdef";
    std::string uuid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";
    
    for (auto& c : uuid) {
        if (c == 'x') {
            c = chars[dis(gen)];
        } else if (c == 'y') {
            c = chars[(dis(gen) & 0x3) | 0x8];
        }
    }
    
    return uuid;
}

int SystemUtils::runCommand(const std::string& command, bool captureOutput, 
                           std::string* output, std::string* errorOutput) {
    if (!captureOutput) {
        return system(command.c_str());
    }
    
    // For captured output, we'll use popen
    std::string fullCommand = command;
    if (errorOutput) {
        fullCommand += " 2>&1";  // Redirect stderr to stdout
    }
    
    FILE* pipe = popen(fullCommand.c_str(), "r");
    if (!pipe) {
        return -1;
    }
    
    char buffer[1024];
    std::string result;
    
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    
    int exitCode = pclose(pipe);
    
    if (output) {
        *output = result;
    }
    
    return exitCode;
}

int SystemUtils::runCommandInDir(const std::string& command, const std::string& workingDir,
                                bool captureOutput, std::string* output) {
    std::string originalDir = getCurrentWorkingDirectory();
    
    if (!changeWorkingDirectory(workingDir)) {
        std::cerr << "Error: Could not change to directory: " << workingDir << std::endl;
        return -1;
    }
    
    int result = runCommand(command, captureOutput, output);
    
    // Restore original directory
    changeWorkingDirectory(originalDir);
    
    return result;
}

bool SystemUtils::createDirectories(const std::string& path) {
    try {
        return std::filesystem::create_directories(path);
    } catch (const std::exception& e) {
        std::cerr << "Error creating directories: " << e.what() << std::endl;
        return false;
    }
}

bool SystemUtils::copyDirectory(const std::string& source, const std::string& destination) {
    try {
        std::filesystem::copy(source, destination, 
                             std::filesystem::copy_options::recursive |
                             std::filesystem::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error copying directory: " << e.what() << std::endl;
        return false;
    }
}

bool SystemUtils::copyFile(const std::string& source, const std::string& destination) {
    try {
        std::filesystem::copy_file(source, destination,
                                  std::filesystem::copy_options::overwrite_existing);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error copying file: " << e.what() << std::endl;
        return false;
    }
}

bool SystemUtils::exists(const std::string& path) {
    return std::filesystem::exists(path);
}

std::string SystemUtils::getAbsolutePath(const std::string& path) {
    try {
        return std::filesystem::absolute(path);
    } catch (const std::exception& e) {
        std::cerr << "Error getting absolute path: " << e.what() << std::endl;
        return path;
    }
}

std::string SystemUtils::getCurrentWorkingDirectory() {
    char* cwd = getcwd(nullptr, 0);
    if (cwd) {
        std::string result(cwd);
        free(cwd);
        return result;
    }
    return "";
}

bool SystemUtils::changeWorkingDirectory(const std::string& path) {
    return chdir(path.c_str()) == 0;
}

int SystemUtils::runPythonScript(const std::string& scriptPath, 
                                const std::vector<std::string>& args,
                                const std::string& workingDir,
                                bool captureOutput,
                                std::string* output) {
    // Use the virtual environment Python if available, otherwise fall back to system Python
    std::string pythonCmd = "python3";
    if (exists("/opt/kaizen_venv/bin/python3")) {
        pythonCmd = "/opt/kaizen_venv/bin/python3";
    }
    
    std::string command = pythonCmd + " " + escapeShellArg(scriptPath);
    
    for (const auto& arg : args) {
        command += " " + escapeShellArg(arg);
    }
    
    if (workingDir.empty()) {
        return runCommand(command, captureOutput, output);
    } else {
        return runCommandInDir(command, workingDir, captureOutput, output);
    }
}

std::string SystemUtils::escapeShellArg(const std::string& arg) {
    // Simple shell argument escaping - wrap in single quotes and escape any single quotes
    std::string escaped = "'";
    for (char c : arg) {
        if (c == '\'') {
            escaped += "'\"'\"'";
        } else {
            escaped += c;
        }
    }
    escaped += "'";
    return escaped;
}
