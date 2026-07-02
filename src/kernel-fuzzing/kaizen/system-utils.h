#ifndef SYSTEM_UTILS_H
#define SYSTEM_UTILS_H

#include <string>
#include <vector>

class SystemUtils {
public:
    // Generate a UUID string
    static std::string generateUUID();
    
    // Run a system command and return exit code
    static int runCommand(const std::string& command, bool captureOutput = false, 
                         std::string* output = nullptr, std::string* errorOutput = nullptr);
    
    // Run a command in a specific working directory
    static int runCommandInDir(const std::string& command, const std::string& workingDir,
                              bool captureOutput = false, std::string* output = nullptr);
    
    // Create directory recursively (like mkdir -p)
    static bool createDirectories(const std::string& path);
    
    // Copy directory recursively
    static bool copyDirectory(const std::string& source, const std::string& destination);
    
    // Copy a single file
    static bool copyFile(const std::string& source, const std::string& destination);
    
    // Check if file or directory exists
    static bool exists(const std::string& path);
    
    // Get absolute path
    static std::string getAbsolutePath(const std::string& path);
    
    // Get current working directory
    static std::string getCurrentWorkingDirectory();
    
    // Change working directory
    static bool changeWorkingDirectory(const std::string& path);
    
    // Execute Python script with arguments
    static int runPythonScript(const std::string& scriptPath, 
                              const std::vector<std::string>& args = {},
                              const std::string& workingDir = "",
                              bool captureOutput = false,
                              std::string* output = nullptr);

private:
    static std::string escapeShellArg(const std::string& arg);
};

#endif // SYSTEM_UTILS_H
