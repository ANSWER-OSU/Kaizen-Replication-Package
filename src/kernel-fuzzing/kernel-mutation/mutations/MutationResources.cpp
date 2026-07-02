#include "MutationResources.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <sstream>

std::vector<std::string> MutationResources::loadLinesFromFile(const std::string& configKey) const {
    std::vector<std::string> lines;
    
    // Get kaizen.cfg path from environment
    std::string kaizenConfigPath = ConfigReader::getKaizenConfigPath();
    if (kaizenConfigPath.empty()) {
        llvm::errs() << "Error: Cannot read config path for " << configKey << "\n";
        return lines;
    }
    
    // Read file path from config
    std::string filePath = ConfigReader::readConfigValue(kaizenConfigPath, "mutation", configKey);
    if (filePath.empty()) {
        llvm::errs() << "Error: " << configKey << " not found in kaizen.cfg\n";
        return lines;
    }
    
    std::ifstream file(filePath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (!line.empty() && line[0] != '#' && line.find_first_not_of(" \t\r\n") != std::string::npos) {
                lines.push_back(line);
            }
        }
        file.close();
    } else {
        llvm::errs() << "Error: Could not read file: " << filePath << "\n";
    }
    
    return lines;
}

std::vector<std::string> MutationResources::loadTemplatesFromFile(const std::string& configKey) const {
    std::vector<std::string> templates;
    
    // Get kaizen.cfg path from environment
    std::string kaizenConfigPath = ConfigReader::getKaizenConfigPath();
    if (kaizenConfigPath.empty()) {
        llvm::errs() << "Error: Cannot read config path for " << configKey << "\n";
        return templates;
    }
    
    // Read file path from config
    std::string filePath = ConfigReader::readConfigValue(kaizenConfigPath, "mutation", configKey);
    if (filePath.empty()) {
        llvm::errs() << "Error: " << configKey << " not found in kaizen.cfg\n";
        return templates;
    }
    
    std::ifstream file(filePath);
    if (file.is_open()) {
        std::string line;
        std::string currentTemplate;
        
        while (std::getline(file, line)) {
            // Skip comment lines
            if (!line.empty() && line[0] == '#') {
                continue;
            }
            
            // Check if this is a blank line (template separator)
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
                // If we have accumulated a template, save it
                if (!currentTemplate.empty()) {
                    templates.push_back(currentTemplate);
                    currentTemplate.clear();
                }
            } else {
                // Add line to current template
                if (!currentTemplate.empty()) {
                    currentTemplate += "\n";
                }
                currentTemplate += line;
            }
        }
        
        // Don't forget the last template if file doesn't end with blank line
        if (!currentTemplate.empty()) {
            templates.push_back(currentTemplate);
        }
        
        file.close();
    } else {
        llvm::errs() << "Error: Could not read file: " << filePath << "\n";
    }
    
    return templates;
}

std::vector<std::string> MutationResources::loadFalseConditions() const {
    return loadLinesFromFile("false_conditions_file");
}

std::vector<std::string> MutationResources::loadDeadCodeStatements() const {
    return loadLinesFromFile("dead_code_statements_file");
}

std::vector<std::string> MutationResources::loadDeadForLoops() const {
    return loadTemplatesFromFile("dead_for_loops_file");
}

std::vector<std::string> MutationResources::loadDeadWhileLoops() const {
    return loadTemplatesFromFile("dead_while_loops_file");
}

std::vector<std::string> MutationResources::loadVariableTypes() const {
    return loadLinesFromFile("variable_types_file");
}

std::map<std::string, std::vector<std::string>> MutationResources::loadVariableValues() const {
    std::map<std::string, std::vector<std::string>> valueMap;
    auto lines = loadLinesFromFile("variable_values_file");
    
    for (const auto& line : lines) {
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            std::string type = line.substr(0, colonPos);
            std::string valuesStr = line.substr(colonPos + 1);
            
            std::vector<std::string> values;
            std::stringstream ss(valuesStr);
            std::string value;
            
            while (std::getline(ss, value, ',')) {
                values.push_back(value);
            }
            
            valueMap[type] = values;
        }
    }
    
    return valueMap;
}

std::map<std::string, std::vector<std::string>> MutationResources::loadVariableNameComponents() const {
    std::map<std::string, std::vector<std::string>> componentMap;
    
    std::string kaizenConfigPath = ConfigReader::getKaizenConfigPath();
    if (kaizenConfigPath.empty()) {
        llvm::errs() << "Error: Cannot read config path for variable_names_file\n";
        return componentMap;
    }
    
    std::string filePath = ConfigReader::readConfigValue(kaizenConfigPath, "mutation", "variable_names_file");
    if (filePath.empty()) {
        llvm::errs() << "Error: variable_names_file not found in kaizen.cfg\n";
        return componentMap;
    }
    
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        llvm::errs() << "Warning: Could not open variable names file: " << filePath << "\n";
        return componentMap;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Parse type:value1,value2,value3 format
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            std::string type = line.substr(0, colonPos);
            std::string valuesStr = line.substr(colonPos + 1);
            
            std::vector<std::string> values;
            std::stringstream ss(valuesStr);
            std::string value;
            
            while (std::getline(ss, value, ',')) {
                values.push_back(value);
            }
            
            componentMap[type] = values;
        }
    }
    
    return componentMap;
}

std::string MutationResources::generateRandomVariableName() const {
    // Try to load from file first
    auto components = loadVariableNameComponents();
    
    std::vector<std::string> prefixes;
    std::vector<std::string> suffixes;
    
    if (!components.empty() && components.count("prefixes") > 0 && components.count("suffixes") > 0) {
        prefixes = components.at("prefixes");
        suffixes = components.at("suffixes");
    } else {
        // error
        llvm::errs() << "Error: Could not load variable name components from file\n";
        return "";
    }
    
    std::uniform_int_distribution<size_t> prefixDist(0, prefixes.size() - 1);
    std::uniform_int_distribution<size_t> suffixDist(0, suffixes.size() - 1);
    std::uniform_int_distribution<int> numDist(1, 999);
    
    return prefixes[prefixDist(rng_)] + 
           suffixes[suffixDist(rng_)] + 
           std::to_string(numDist(rng_));
}

std::string MutationResources::replacePlaceholders(const std::string& template_str, 
                                                   const std::map<std::string, std::string>& replacements) const {
    std::string result = template_str;
    
    for (const auto& pair : replacements) {
        std::string placeholder = "{" + pair.first + "}";
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), pair.second);
            pos += pair.second.length();
        }
    }
    
    return result;
}
