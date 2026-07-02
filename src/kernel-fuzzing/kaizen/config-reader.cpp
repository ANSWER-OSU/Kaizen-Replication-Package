#include "config-reader.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <sstream>
#include <cstdlib>

std::string ConfigReader::getKaizenConfigPath() {
    const char* configPath = std::getenv("KAIZEN_CONFIG_FILE");
    if (configPath) {
        return std::string(configPath);
    }
    llvm::errs() << "Error: KAIZEN_CONFIG_FILE environment variable not set\n";
    return "";
}

std::ifstream ConfigReader::openConfigFile(const std::string& configFile) {
    std::ifstream file(configFile);
    return file;
}

std::string ConfigReader::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::string ConfigReader::readConfigValue(const std::string& configFile, 
                                        const std::string& section, 
                                        const std::string& key) {
    std::ifstream file = openConfigFile(configFile);
    
    if (!file.is_open()) {
        llvm::errs() << "Error: Could not read config file: " << configFile << "\n";
        return "";
    }
    
    std::string line;
    bool inSection = false;
    std::string targetSection = "[" + section + "]";
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Check if we're entering the target section
        if (line == targetSection) {
            inSection = true;
            continue;
        }
        
        // Check if we're entering a different section
        if (line.length() > 0 && line[0] == '[') {
            inSection = false;
            continue;
        }
        
        // If we're in the target section, look for the key
        if (inSection && line.find('=') != std::string::npos) {
            size_t equalPos = line.find('=');
            std::string configKey = trim(line.substr(0, equalPos));
            std::string configValue = trim(line.substr(equalPos + 1));
            
            if (configKey == key) {
                file.close();
                return configValue;
            }
        }
    }
    
    file.close();
    return "";
}

double ConfigReader::readConfigDouble(const std::string& configFile,
                                     const std::string& section,
                                     const std::string& key,
                                     double defaultValue) {
    std::string valueStr = readConfigValue(configFile, section, key);
    if (valueStr.empty()) {
        return defaultValue;
    }
    
    try {
        return std::stod(valueStr);
    } catch (const std::exception& e) {
        llvm::errs() << "Warning: Could not parse double value '" << valueStr 
                     << "' for " << section << "." << key << ", using default " 
                     << defaultValue << "\n";
        return defaultValue;
    }
}

std::map<std::string, std::string> ConfigReader::readConfigSection(const std::string& configFile,
                                                                  const std::string& section) {
    std::map<std::string, std::string> result;
    std::ifstream file = openConfigFile(configFile);
    
    if (!file.is_open()) {
        llvm::errs() << "Error: Could not read config file: " << configFile << "\n";
        return result;
    }
    
    std::string line;
    bool inSection = false;
    std::string targetSection = "[" + section + "]";
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Check if we're entering the target section
        if (line == targetSection) {
            inSection = true;
            continue;
        }
        
        // Check if we're entering a different section
        if (line.length() > 0 && line[0] == '[') {
            if (inSection) break; // We've finished reading our section
            inSection = false;
            continue;
        }
        
        // If we're in the target section, collect all key-value pairs
        if (inSection && line.find('=') != std::string::npos) {
            size_t equalPos = line.find('=');
            std::string configKey = trim(line.substr(0, equalPos));
            std::string configValue = trim(line.substr(equalPos + 1));
            
            // Handle multi-line values (like types = { ... })
            if (configValue.find('{') != std::string::npos && configValue.find('}') == std::string::npos) {
                // This is a multi-line value starting with {
                std::string multiLineValue = configValue;
                std::string nextLine;
                
                while (std::getline(file, nextLine)) {
                    nextLine = trim(nextLine);
                    if (nextLine.empty() || nextLine[0] == '#') continue;
                    
                    multiLineValue += " " + nextLine;
                    
                    // Check if we found the closing brace
                    if (nextLine.find('}') != std::string::npos) {
                        break;
                    }
                }
                
                result[configKey] = multiLineValue;
            } else {
                result[configKey] = configValue;
            }
        }
    }
    
    file.close();
    return result;
}

std::map<std::string, int> ConfigReader::parseTypesString(const std::string& typesStr) {
    std::map<std::string, int> result;
    
    // Find the opening and closing braces
    size_t openBrace = typesStr.find('{');
    size_t closeBrace = typesStr.find('}');
    
    if (openBrace == std::string::npos || closeBrace == std::string::npos) {
        return result; // Empty map if malformed
    }
    
    // Extract content between braces
    std::string content = typesStr.substr(openBrace + 1, closeBrace - openBrace - 1);
    content = trim(content);
    
    if (content.empty()) {
        return result;
    }
    
    // Parse key=value pairs separated by commas
    size_t pos = 0;
    while (pos < content.length()) {
        // Find the next equals sign
        size_t equalPos = content.find('=', pos);
        if (equalPos == std::string::npos) break;
        
        // Extract key
        std::string key = trim(content.substr(pos, equalPos - pos));
        
        // Find the end of the value (comma or end of string)
        size_t valueStart = equalPos + 1;
        size_t commaPos = content.find(',', valueStart);
        size_t valueEnd = (commaPos == std::string::npos) ? content.length() : commaPos;
        
        // Extract value
        std::string valueStr = trim(content.substr(valueStart, valueEnd - valueStart));
        
        if (!key.empty() && !valueStr.empty()) {
            try {
                result[key] = std::stoi(valueStr);
            } catch (const std::exception&) {
                // Skip invalid values
            }
        }
        
        pos = (commaPos == std::string::npos) ? content.length() : commaPos + 1;
    }
    
    return result;
}

int ConfigReader::getTypeWeight(const std::string& configFile, 
                              const std::string& section, 
                              const std::string& typeName) {
    auto sectionData = readConfigSection(configFile, section);
    std::string typesStr = sectionData["types"];
    
    llvm::outs() << "Debug: Reading types from section '" << section << "': '" << typesStr << "'\n";
    
    if (typesStr.empty()) {
        llvm::outs() << "Debug: Types string is empty\n";
        return -1;
    }
    
    auto typesMap = parseTypesString(typesStr);
    
    llvm::outs() << "Debug: Parsed " << typesMap.size() << " types:\n";
    for (const auto& pair : typesMap) {
        llvm::outs() << "  " << pair.first << " = " << pair.second << "\n";
    }
    
    auto it = typesMap.find(typeName);
    int result = (it != typesMap.end()) ? it->second : -1;
    
    llvm::outs() << "Debug: Looking for '" << typeName << "', result: " << result << "\n";
    
    return result;
}

bool ConfigReader::isMutationCategoryEnabled(const std::string& configFile, 
                                           const std::string& section) {
    std::string probabilityStr = readConfigValue(configFile, section, "probability");
    
    if (probabilityStr.empty()) {
        return false;
    }
    
    try {
        return std::stoi(probabilityStr) > 0;
    } catch (const std::exception&) {
        return false;
    }
}
