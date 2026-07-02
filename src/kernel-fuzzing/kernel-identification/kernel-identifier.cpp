#include "kernel-identifier.h"
#include "../kaizen/file-utils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <random>
#include <iomanip>

#include "../kernel-identification/KernelExtractor.h"
#include "../kernel-identification/RunBear.h"
#include "../kernel-identification/CompilationCommandGenerator.h"
#include "../kernel-translation/JSONOutput.h"
#include "../kernel-mutation/KernelMutator.h"
#include "../kernel-identification/kernel-identifier.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

static cl::opt<std::string> BuildPath("p", cl::desc("Path to the build directory containing a Makefile"));
static cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"), cl::value_desc("filename"));
static cl::opt<bool> MutateKernels("mutate", cl::desc("Enable kernel mutation"), cl::init(false));

static cl::OptionCategory MyToolCategory("kernel-extractor options");

// Store the output filename for later retrieval
static std::string last_output_filename;

class MyFrontendAction : public ASTFrontendAction {
public:
    MyFrontendAction(MatchFinder *Finder, KernelExtractor *Extractor, std::vector<FileResult> *Results, const std::string &CompileCommand)
        : Finder(Finder), Extractor(Extractor), Results(Results), CompileCommand(CompileCommand) {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        Extractor->setCompileCommand(CompileCommand);
        return Finder->newASTConsumer();
    }

    void EndSourceFileAction() override {
        Results->push_back(Extractor->takeResult());
    }

private:
    MatchFinder *Finder;
    KernelExtractor *Extractor;
    std::vector<FileResult> *Results;
    std::string CompileCommand;
};

class MyFrontendActionFactory : public FrontendActionFactory {
public:
    MyFrontendActionFactory(MatchFinder &Finder, KernelExtractor &Extractor, std::vector<FileResult> &Results, const FilteredCompilationDatabase &CompDB)
        : Finder(Finder), Extractor(Extractor), Results(Results), CompDB(CompDB) {}

    std::unique_ptr<FrontendAction> create() override {
        // Get the compile command for the current file being processed
        std::string compileCommandStr;
        auto commands = CompDB.getAllCompileCommands();
        if (!commands.empty()) {
            // Join the command line arguments into a single string
            for (const auto &arg : commands[0].CommandLine) {
                if (!compileCommandStr.empty()) compileCommandStr += " ";
                compileCommandStr += arg;
            }
        }
        return std::make_unique<MyFrontendAction>(&Finder, &Extractor, &Results, compileCommandStr);
    }

private:
    MatchFinder &Finder;
    KernelExtractor &Extractor;
    std::vector<FileResult> &Results;
    const FilteredCompilationDatabase &CompDB;
};


int run_kernel_extractor(int argc, const char **argv) {
    cl::HideUnrelatedOptions(MyToolCategory);
    cl::ParseCommandLineOptions(argc, argv);

    // Store the output filename
    last_output_filename = OutputFilename;
    
    // Validate that BuildPath is provided for kernel extraction
    if (BuildPath.empty()) {
        llvm::errs() << "Error: -p option (build path) is required for kernel extraction.\n";
        return 1;
    }

    // Get absolute path for the build directory
    llvm::SmallString<256> absBuildPath(BuildPath);
    if (std::error_code ec = llvm::sys::fs::make_absolute(absBuildPath)) {
        llvm::errs() << "Error getting absolute path for build directory: " << ec.message() << "\n";
        return 1;
    }
    
    // Generate compilation database (bear) inside folder
    std::string compilation_database_path;
    if (!BuildPath.empty()) {
        if (!generateCompileCommands(BuildPath)) {
            llvm::errs() << "Failed to generate compilation database.\n";
            return 1;
        }
        compilation_database_path = BuildPath;
    } else {
        compilation_database_path = ".";
    }

    std::string ErrorMessage;
    auto Compilations = JSONCompilationDatabase::loadFromFile(
        compilation_database_path + "/compile_commands.json", ErrorMessage,
        JSONCommandLineSyntax::Gnu);

    if (!Compilations) {
        llvm::errs() << "Error reading compilation database: " << ErrorMessage << "\n";
        return 1;
    }

    FilteredCompilationDatabase FilteredCompilations(std::move(Compilations));
    ClangTool Tool(FilteredCompilations, FilteredCompilations.getAllFiles());

    // Create the extractor with mutation enabled or disabled
    KernelMutatorFn mutatorFn = nullptr;
    if (MutateKernels) {
        mutatorFn = static_cast<void(*)(clang::FunctionDecl*, clang::Rewriter&)>(mutateKernel);
        llvm::outs() << "Kernel mutation enabled\n";
    }
    KernelExtractor Extractor(mutatorFn);
    Extractor.setProjectPath(std::string(absBuildPath.c_str()));
    
    MatchFinder Finder;
    Finder.addMatcher(functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("cudaKernel"), &Extractor);

    std::vector<FileResult> all_results;
    MyFrontendActionFactory factory(Finder, Extractor, all_results, FilteredCompilations);

    Tool.run(&factory);

    // Save mutated files if mutation was enabled and we have rewritten content
    if (MutateKernels) {
        // Get session directory from environment
        std::string sessionDir = getSessionDirFromEnv();
        
        if (!sessionDir.empty()) {
            for (auto& result : all_results) {
                if (result.needsReprocessing && !result.rewrittenFileContent.empty()) {
                    // Create mutated directory within session directory
                    std::string mutatedDir = sessionDir + "/mutated";
                    std::filesystem::create_directories(mutatedDir);
                    
                    // Get the original file info
                    std::filesystem::path originalPath(result.sourceFile);
                    std::string originalDir = originalPath.parent_path().string();
                    std::string originalFilename = originalPath.filename().string();
                    
                    // Copy all files from original directory
                    copyDirectoryStructure(originalDir, mutatedDir);
                    
                    // Full path for the mutated file in mutated directory (keep original name)
                    std::filesystem::path mutatedFilePath = std::filesystem::path(mutatedDir) / originalFilename;
                    
                    // Write the mutated file
                    std::ofstream mutatedFile(mutatedFilePath);
                    if (mutatedFile.is_open()) {
                        mutatedFile << result.rewrittenFileContent;
                        mutatedFile.close();
                        llvm::outs() << "Saved mutated file to: " << mutatedFilePath.string() << "\n";
                    } else {
                        llvm::errs() << "Failed to create mutated file: " << mutatedFilePath.string() << "\n";
                    }
                }
            }
        }
    }

    std::error_code EC;
    std::unique_ptr<raw_fd_ostream> OutFile;
    raw_ostream *OS = &outs();

    if (!OutputFilename.empty()) {
        OutFile = std::make_unique<raw_fd_ostream>(OutputFilename, EC);
        if (EC) {
            errs() << "Error opening output file " << OutputFilename << ": " << EC.message() << "\n";
            return 1;
        }
        OS = OutFile.get();
    }

    printResultsAsJSON(all_results, *OS);

    return 0;
}

int extract_kernels_from_path(const std::string& buildPath, const std::string& outputFilename, bool mutate, KernelMutatorFn mutatorFn, const std::string& mutated_file_path) {
    // Get absolute path for the build directory
    llvm::SmallString<256> absBuildPath(buildPath);
    if (std::error_code ec = llvm::sys::fs::make_absolute(absBuildPath)) {
        llvm::errs() << "Error getting absolute path for build directory: " << ec.message() << "\n";
        return 1;
    }
    
    // Generate compilation database (bear) inside folder
    std::string compilation_database_path;
    if (!buildPath.empty()) {
        if (!generateCompileCommands(buildPath)) {
            llvm::errs() << "Failed to generate compilation database.\n";
            return 1;
        }
        compilation_database_path = buildPath;
    } else {
        compilation_database_path = ".";
    }

    std::string ErrorMessage;
    auto Compilations = JSONCompilationDatabase::loadFromFile(
        compilation_database_path + "/compile_commands.json", ErrorMessage,
        JSONCommandLineSyntax::Gnu);

    if (!Compilations) {
        llvm::errs() << "Error reading compilation database: " << ErrorMessage << "\n";
        return 1;
    }

    FilteredCompilationDatabase FilteredCompilations(std::move(Compilations));
    ClangTool Tool(FilteredCompilations, FilteredCompilations.getAllFiles());

    KernelExtractor Extractor(mutatorFn);
    Extractor.setProjectPath(std::string(absBuildPath.c_str()));
    
    MatchFinder Finder;
    Finder.addMatcher(functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("cudaKernel"), &Extractor);

    std::vector<FileResult> all_results;
    MyFrontendActionFactory factory(Finder, Extractor, all_results, FilteredCompilations);

    Tool.run(&factory);

    // Save mutated files if mutation was enabled and we have rewritten content
    if (mutate) {
        // Get session directory from environment
        std::string sessionDir = getSessionDirFromEnv();
        
        if (!sessionDir.empty()) {
            for (auto& result : all_results) {
                if (result.needsReprocessing && !result.rewrittenFileContent.empty()) {
                    // Create mutated directory within session directory
                    std::string mutatedDir = sessionDir + "/mutated";
                    std::filesystem::create_directories(mutatedDir);
                    
                    // Get the original file info
                    std::filesystem::path originalPath(result.sourceFile);
                    std::string originalDir = originalPath.parent_path().string();
                    std::string originalFilename = originalPath.filename().string();
                    
                    // Copy all files from original directory
                    copyDirectoryStructure(originalDir, mutatedDir);
                    
                    // Full path for the mutated file in mutated directory (keep original name)
                    std::filesystem::path mutatedFilePath = std::filesystem::path(mutatedDir) / originalFilename;
                    
                    // Write the mutated file
                    std::ofstream mutatedFile(mutatedFilePath);
                    if (mutatedFile.is_open()) {
                        mutatedFile << result.rewrittenFileContent;
                        mutatedFile.close();
                        llvm::outs() << "Saved mutated file to: " << mutatedFilePath.string() << "\n";
                    } else {
                        llvm::errs() << "Failed to create mutated file: " << mutatedFilePath.string() << "\n";
                    }
                }
            }
        }
    }

    std::error_code EC;
    std::unique_ptr<raw_fd_ostream> OutFile;
    raw_ostream *OS = &outs();

    if (!outputFilename.empty()) {
        OutFile = std::make_unique<raw_fd_ostream>(outputFilename, EC);
        if (EC) {
            errs() << "Error opening output file " << outputFilename << ": " << EC.message() << "\n";
            return 1;
        }
        OS = OutFile.get();
    }

    printResultsAsJSON(all_results, *OS);

    return 0;
}

std::string get_output_filename() {
    return last_output_filename;
}