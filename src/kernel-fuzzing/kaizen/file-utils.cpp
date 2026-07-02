#include "file-utils.h"
#include <filesystem>
#include <cstdlib>
#include <iostream>

// Get session directory from environment variable
std::string getSessionDirFromEnv() {
    const char* envSessionDir = std::getenv("KAIZEN_SESSION_DIR");
    if (envSessionDir) {
        return std::string(envSessionDir);
    }
    return "";
}

// Copy directory structure including all files
void copyDirectoryStructure(const std::string& sourceDir, const std::string& destDir, const std::string& excludeFile) {
    if (!std::filesystem::exists(sourceDir)) {
        std::cerr << "Source directory does not exist: " << sourceDir << std::endl;
        return;
    }
    std::filesystem::create_directories(destDir);

    for (const auto& entry : std::filesystem::recursive_directory_iterator(sourceDir)) {
        const auto& path = entry.path();
        auto relative_path = std::filesystem::relative(path, sourceDir);
        auto dest_path = std::filesystem::path(destDir) / relative_path;

        if (entry.is_directory()) {
            std::filesystem::create_directories(dest_path);
        } else if (entry.is_regular_file()) {
            if (path.filename() != excludeFile) {
                std::filesystem::copy(path, dest_path, std::filesystem::copy_options::overwrite_existing);
            }
        }
    }
}
