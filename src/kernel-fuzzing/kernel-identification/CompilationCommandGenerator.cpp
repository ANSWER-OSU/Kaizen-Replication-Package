#include "CompilationCommandGenerator.h"
#include "llvm/Support/raw_ostream.h"


/**
 * Filters and transforms an NVCC compilation command to a Clang-compatible version.
 */
std::vector<std::string> filterCompileCommand(const std::vector<std::string>& command) {
    std::vector<std::string> new_command;
    if (command.empty()) {
        return new_command;
    }

    // Replace the compiler with clang++
    new_command.push_back("/usr/bin/clang++-18");

    new_command.push_back("-stdlib=libstdc++");

    // new_command.push_back("-D_LIBCPP_VERSION");

    // new_command.push_back("-D_VSTD=std");

    for (size_t i = 1; i < command.size(); ++i) {
        const std::string &arg = command[i];

        // Replace NVCC-specific flags with Clang equivalents
        if (arg.find("-arch=") != std::string::npos) {
            new_command.push_back("--cuda-gpu-arch=" + arg.substr(arg.find("=") + 1));
        } else if (arg == "-Xcompiler") {
            // Skip NVCC's -Xcompiler flag, as Clang does not need it
            continue;
        } else if (arg == "-cubin" || arg == "-ptx") {
            // Skip NVCC-specific output formats
            continue;
        } else {
            // Preserve other flags
            new_command.push_back(arg);
        }
    }

    //  Clang-specific CUDA flags
    new_command.push_back("--cuda-path=/usr/local/cuda");
    new_command.push_back("--cuda-host-only"); // Compile host code only

    // Prevent CUDA headers from defining __noinline__
    new_command.push_back("-D__NO_CUDA__NOINLINE__");

    // Debug print the filtered command
    llvm::errs() << "Filtered command: ";
    for (const auto &arg : new_command) {
        llvm::errs() << arg << " ";
    }
    llvm::errs() << "\n";

    return new_command;
}

/**
 * A wrapper around CompilationDatabase that filters all commands through filterCompileCommand.
 */
FilteredCompilationDatabase::FilteredCompilationDatabase(
    std::unique_ptr<clang::tooling::CompilationDatabase> Compilations)
    : Compilations(std::move(Compilations)) {}

std::vector<clang::tooling::CompileCommand> 
FilteredCompilationDatabase::getCompileCommands(llvm::StringRef FilePath) const {
    auto commands = Compilations->getCompileCommands(FilePath);
    for (auto &command : commands) {
        command.CommandLine = filterCompileCommand(command.CommandLine);
    }
    return commands;
}

std::vector<std::string> FilteredCompilationDatabase::getAllFiles() const {
    return Compilations->getAllFiles();
}

std::vector<clang::tooling::CompileCommand> 
FilteredCompilationDatabase::getAllCompileCommands() const {
    auto commands = Compilations->getAllCompileCommands();
    for (auto &command : commands) {
        command.CommandLine = filterCompileCommand(command.CommandLine);
    }
    return commands;
}
