#include "JSONOutput.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <sstream>
#include <regex>

std::string removeTripleSlashComments(const std::string &content) {
    std::istringstream stream(content);
    std::string line;
    std::string result;
    
    while (std::getline(stream, line)) {
        // Check if line contains triple slash comment (with optional leading whitespace)
        std::regex tripleSlashRegex(R"(^\s*///.*$)");
        if (!std::regex_match(line, tripleSlashRegex)) {
            // Keep the line if it's not a triple slash comment
            if (!result.empty()) {
                result += "\n";
            }
            result += line;
        }
    }
    
    return result;
}

std::string escapeJSON(llvm::StringRef S) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    for (const char C : S) {
        switch (C) {
        case '"': OS << "\\\""; break;
        case '\\': OS << "\\\\"; break;
        case '\b': OS << "\\b"; break;
        case '\f': OS << "\\f"; break;
        case '\n': OS << "\\n"; break;
        case '\r': OS << "\\r"; break;
        case '\t': OS << "\\t"; break;
        default:
            if (C >= 0 && C < 32) { // Control characters
                OS << "\\u00" << llvm::hexdigit((C >> 4) & 0xF) << llvm::hexdigit(C & 0xF);
            } else {
                OS << C;
            }
        }
    }
    return OS.str();
}

bool ensureOutputDirectory(const std::string &outputPath) {
    llvm::SmallString<256> dirPath(outputPath);
    llvm::sys::path::remove_filename(dirPath);

    if (dirPath.empty()) {
        return true; // Current directory, no need to create
    }

    std::error_code EC = llvm::sys::fs::create_directories(dirPath);
    if (EC) {
        llvm::errs() << "Error creating output directory '" << dirPath << "': " << EC.message() << "\n";
        return false;
    }

    return true;
}

void printResultsAsJSON(const std::vector<FileResult> &results, llvm::raw_ostream &OS) {
    OS << "[\n";
    bool firstFile = true;
    for (const auto& result : results) {
        if (result.sections.empty()) continue;

        if (!firstFile) OS << ",\n";
        OS << "  {\n";
        OS << "    \"source_file\": \"" << escapeJSON(result.sourceFile) << "\",\n";
        OS << "    \"compile_command\": \"" << escapeJSON(result.compileCommand) << "\",\n";
        OS << "    \"sections\": [\n";

        bool firstSection = true;
        for (const auto &section : result.sections) {
            if (!firstSection) OS << ",\n";
            OS << "      {\n";
            OS << "        \"type\": \"" << section.type << "\",\n";
            if (!section.name.empty()) {
                OS << "        \"name\": \"" << escapeJSON(section.name) << "\",\n";
            }
            // Filter out triple slash comments before escaping JSON
            std::string filteredContent = removeTripleSlashComments(section.content);
            OS << "        \"content\": \"" << escapeJSON(filteredContent) << "\"\n";
            OS << "      }";
            firstSection = false;
        }

        OS << "\n    ]\n";
        OS << "  }";
        firstFile = false;
    }
    OS << "\n]\n";
    OS.flush();
}
