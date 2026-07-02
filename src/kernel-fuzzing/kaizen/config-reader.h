#ifndef KAIZEN_CONFIG_READER_H
#define KAIZEN_CONFIG_READER_H

#include <string>
#include <map>
#include <fstream>

class ConfigReader {
public:
    // Get the path to kaizen.cfg from environment variable
    static std::string getKaizenConfigPath();
    
    // Read a configuration value from a config file
    static std::string readConfigValue(const std::string& configFile, 
                                     const std::string& section, 
                                     const std::string& key);
    
    // Read a double configuration value from a config file
    static double readConfigDouble(const std::string& configFile,
                                 const std::string& section,
                                 const std::string& key,
                                 double defaultValue = 0.0);
    
    // Read all key-value pairs from a section
    static std::map<std::string, std::string> readConfigSection(const std::string& configFile,
                                                               const std::string& section);
    
    // Parse types configuration and get the weight for a specific type
    // Returns -1 if type not found or disabled, otherwise returns the weight
    static int getTypeWeight(const std::string& configFile, 
                           const std::string& section, 
                           const std::string& typeName);
    
    // Check if a mutation category is enabled (probability > 0)
    static bool isMutationCategoryEnabled(const std::string& configFile, 
                                        const std::string& section);

private:
    // Helper method to open config file (no guessing, exact path only)
    static std::ifstream openConfigFile(const std::string& configFile);
    
    // Helper method to trim whitespace
    static std::string trim(const std::string& str);
    
    // Helper method to parse types string format "{ key = value, key2 = value2 }"
    static std::map<std::string, int> parseTypesString(const std::string& typesStr);
};

#endif // KAIZEN_CONFIG_READER_H
