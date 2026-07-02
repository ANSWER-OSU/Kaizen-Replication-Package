#ifndef ENVIRONMENT_CONFIG_H
#define ENVIRONMENT_CONFIG_H

#include <string>
#include <map>

class EnvironmentConfig {
public:
    // Static method to load environment from .env file
    static bool loadFromFile(const std::string& envFilePath);
    
    // Get environment variable with optional default value
    static std::string getEnv(const std::string& key, const std::string& defaultValue = "");
    
    // Get required environment variable (throws if not found)
    static std::string getRequiredEnv(const std::string& key);
    
    // Check if environment variable exists
    static bool hasEnv(const std::string& key);
    
    // Get all loaded environment variables
    static const std::map<std::string, std::string>& getAllEnvVars();

private:
    static std::map<std::string, std::string> envVars;
    static std::string trim(const std::string& str);
    static std::string expandVariables(const std::string& value);
};

#endif // ENVIRONMENT_CONFIG_H
