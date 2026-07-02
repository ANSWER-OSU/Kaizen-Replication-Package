#ifndef COMPILATION_COMMAND_GENERATOR_H
#define COMPILATION_COMMAND_GENERATOR_H

#include "clang/Tooling/CompilationDatabase.h"
#include <string>
#include <vector>
#include <memory>

/**
 * Filters and transforms an NVCC compilation command to a Clang-compatible version.
 */
std::vector<std::string> filterCompileCommand(const std::vector<std::string>& command);

/**
 * A wrapper around CompilationDatabase that filters all commands through filterCompileCommand.
 */
class FilteredCompilationDatabase : public clang::tooling::CompilationDatabase {
public:
    FilteredCompilationDatabase(std::unique_ptr<clang::tooling::CompilationDatabase> Compilations);

    std::vector<clang::tooling::CompileCommand> getCompileCommands(llvm::StringRef FilePath) const override;
    std::vector<std::string> getAllFiles() const override;
    std::vector<clang::tooling::CompileCommand> getAllCompileCommands() const override;

private:
    std::unique_ptr<clang::tooling::CompilationDatabase> Compilations;
};

#endif // COMPILATION_COMMAND_GENERATOR_H
