#ifndef MUTATION_RESOURCES_H
#define MUTATION_RESOURCES_H

#include "../../kaizen/config-reader.h"
#include <string>
#include <vector>
#include <map>
#include <random>

class MutationResources {
public:
    MutationResources(std::mt19937& rng) : rng_(rng) {}
    
    // Load false conditions from file
    std::vector<std::string> loadFalseConditions() const;
    
    // Load dead code statements from file
    std::vector<std::string> loadDeadCodeStatements() const;
    
    // Load dead for loop patterns from file
    std::vector<std::string> loadDeadForLoops() const;
    
    // Load dead while loop patterns from file
    std::vector<std::string> loadDeadWhileLoops() const;
    
    // Load variable types from file
    std::vector<std::string> loadVariableTypes() const;
    
    // Load variable values by type from file
    std::map<std::string, std::vector<std::string>> loadVariableValues() const;
    
    // Load variable name components from file
    std::map<std::string, std::vector<std::string>> loadVariableNameComponents() const;
    
    // Generate random variable name
    std::string generateRandomVariableName() const;
    
    // Get random element from vector
    template<typename T>
    T getRandomElement(const std::vector<T>& vec) const {
        if (vec.empty()) return T{};
        std::uniform_int_distribution<size_t> dist(0, vec.size() - 1);
        return vec[dist(rng_)];
    }
    
    // Replace placeholders in template strings
    std::string replacePlaceholders(const std::string& template_str, 
                                   const std::map<std::string, std::string>& replacements) const;

private:
    std::mt19937& rng_;
    
    // Helper method to load lines from a file
    std::vector<std::string> loadLinesFromFile(const std::string& configKey) const;
    
    // Helper method to load multi-line templates separated by blank lines
    std::vector<std::string> loadTemplatesFromFile(const std::string& configKey) const;
};

#endif // MUTATION_RESOURCES_H
