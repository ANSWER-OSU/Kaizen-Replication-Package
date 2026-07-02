#include "environment-config.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <unistd.h>

std::map<std::string, std::string> EnvironmentConfig::envVars;

bool EnvironmentConfig::loadFromFile(const std::string& envFilePath) {
    std::ifstream file(envFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open .env file: " << envFilePath << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Find the equals sign
        size_t equalsPos = line.find('=');
        if (equalsPos == std::string::npos) {
            continue;
        }
        
        std::string key = trim(line.substr(0, equalsPos));
        std::string value = trim(line.substr(equalsPos + 1));
        
        // Remove quotes if present
        if (value.length() >= 2) {
            if ((value.front() == '"' && value.back() == '"') ||
                (value.front() == '\'' && value.back() == '\'')) {
                value = value.substr(1, value.length() - 2);
            }
        }
        
        // Expand variables in the value
        value = expandVariables(value);
        
        // Store in our map and set in environment
        envVars[key] = value;
        setenv(key.c_str(), value.c_str(), 1);
    }
    
    file.close();
    return true;
}

std::string EnvironmentConfig::getEnv(const std::string& key, const std::string& defaultValue) {
    // First check system environment (prioritize container environment variables)
    const char* value = std::getenv(key.c_str());
    if (value) {
        return std::string(value);
    }
    
    // Then check our loaded vars from .env file
    auto it = envVars.find(key);
    if (it != envVars.end()) {
        return it->second;
    }
    
    return defaultValue;
}

std::string EnvironmentConfig::getRequiredEnv(const std::string& key) {
    std::string value = getEnv(key);
    if (value.empty()) {
        throw std::runtime_error("Required environment variable not set: " + key);
    }
    return value;
}

bool EnvironmentConfig::hasEnv(const std::string& key) {
    return envVars.find(key) != envVars.end() || std::getenv(key.c_str()) != nullptr;
}

const std::map<std::string, std::string>& EnvironmentConfig::getAllEnvVars() {
    return envVars;
}

std::string EnvironmentConfig::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::string EnvironmentConfig::expandVariables(const std::string& value) {
    std::string result = value;
    
    // Handle $(pwd) special case
    std::regex pwdRegex(R"(\$\(pwd\))");
    if (std::regex_search(result, pwdRegex)) {
        char* cwd = getcwd(nullptr, 0);
        if (cwd) {
            result = std::regex_replace(result, pwdRegex, std::string(cwd));
            free(cwd);
        }
    }
    
    // Handle ${VAR} and $VAR patterns
    std::regex varRegex(R"(\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*))");
    std::smatch match;
    
    while (std::regex_search(result, match, varRegex)) {
        std::string varName = match[1].str();
        if (varName.empty()) {
            varName = match[2].str();
        }
        
        // Get the variable value
        std::string varValue = getEnv(varName);
        
        // Replace the match with the variable value
        result.replace(match.position(), match.length(), varValue);
    }
    
    return result;
}
