#ifndef KAIZEN_FILE_UTILS_H
#define KAIZEN_FILE_UTILS_H

#include <string>

// Get session directory from environment variable
std::string getSessionDirFromEnv();

// Copy directory structure including all files
void copyDirectoryStructure(const std::string& sourceDir, const std::string& destDir, const std::string& excludeFile = "");

#endif // KAIZEN_FILE_UTILS_H
