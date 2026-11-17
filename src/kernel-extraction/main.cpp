#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <regex>

// Frontend action for kernel extraction
class KernelExtractionAction : public clang::ASTFrontendAction {
public:
    KernelExtractionAction(clang::ast_matchers::MatchFinder* finder) : finder(finder) {}
    
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) override {
        return finder->newASTConsumer();
    }
    
private:
    clang::ast_matchers::MatchFinder* finder;
};

// Factory for creating frontend actions
class KernelExtractionActionFactory : public clang::tooling::FrontendActionFactory {
public:
    KernelExtractionActionFactory(clang::ast_matchers::MatchFinder* finder) : finder(finder) {}
    
    std::unique_ptr<clang::FrontendAction> create() override {
        return std::make_unique<KernelExtractionAction>(finder);
    }
    
private:
    clang::ast_matchers::MatchFinder* finder;
};

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

// Structure to hold kernel information
struct KernelInfo {
    std::string name;
    std::string content;
    int number;
    SourceLocation sourceLocation;
};

// Kernel extractor callback class
class SimpleKernelExtractor : public MatchFinder::MatchCallback {
public:
    SimpleKernelExtractor(const std::string& projectPath) : kernelCounter(0), projectPath(projectPath) {}
    
    void run(const MatchFinder::MatchResult &Result) override {
        llvm::outs() << "DEBUG: MatchFinder::run() called\n";
        
        const auto *KernelFunc = Result.Nodes.getNodeAs<FunctionDecl>("cudaKernel");
        if (!KernelFunc) {
            llvm::outs() << "DEBUG: No kernel function found in match result\n";
            return;
        }
        
        llvm::outs() << "DEBUG: Found function: " << KernelFunc->getNameAsString() << "\n";
        
        // Check if the kernel is from a valid location
        SourceLocation loc = KernelFunc->getBeginLoc();
        if (loc.isInvalid()) {
            llvm::outs() << "DEBUG: Invalid source location for kernel\n";
            return;
        }
        
        llvm::outs() << "DEBUG: Kernel location: " << loc.printToString(*Result.SourceManager) << "\n";
        
        // Skip system headers
        if (Result.SourceManager->isInSystemHeader(loc)) {
            llvm::outs() << "DEBUG: Skipping kernel in system header\n";
            return;
        }
        
        // Get the file path for this kernel
        std::string filePath;
        
        // First try the presumed location (handles macro expansions)
        clang::PresumedLoc presumedLoc = Result.SourceManager->getPresumedLoc(loc);
        if (presumedLoc.isValid()) {
            filePath = presumedLoc.getFilename();
        } else {
            // Fallback to expansion location
            clang::SourceLocation expLoc = Result.SourceManager->getExpansionLoc(loc);
            const clang::FileEntry* fileEntry = Result.SourceManager->getFileEntryForID(Result.SourceManager->getFileID(expLoc));
            if (fileEntry) {
                filePath = fileEntry->getName().str();
            } else {
                filePath = Result.SourceManager->getFilename(loc).str();
            }
        }
        llvm::outs() << "DEBUG: Kernel file path: " << filePath << "\n";
        llvm::outs() << "DEBUG: Project path: " << projectPath << "\n";
        
        // Ensure the kernel is from within our project directory
        if (!isWithinProjectDirectory(filePath)) {
            llvm::outs() << "DEBUG: Skipping kernel from outside project: " << filePath << "\n";
            return;
        }
        
        // Extract kernel information
        std::string kernelName = KernelFunc->getNameAsString();
        SourceRange kernelRange = KernelFunc->getSourceRange();
        
        // Check if this function has a template declaration and extend range to include it
        if (FunctionTemplateDecl *FuncTemplate = KernelFunc->getDescribedFunctionTemplate()) {
            SourceLocation templateBegin = FuncTemplate->getBeginLoc();
            if (templateBegin.isValid()) {
                llvm::outs() << "DEBUG: Found template declaration, extending range to include it\n";
                kernelRange.setBegin(templateBegin);
            }
        } else if (const FunctionDecl *TemplatedFunc = KernelFunc->getTemplateInstantiationPattern()) {
            // Handle template specializations
            if (FunctionTemplateDecl *FuncTemplate = TemplatedFunc->getDescribedFunctionTemplate()) {
                SourceLocation templateBegin = FuncTemplate->getBeginLoc();
                if (templateBegin.isValid()) {
                    llvm::outs() << "DEBUG: Found template declaration for specialization, extending range to include it\n";
                    kernelRange.setBegin(templateBegin);
                }
            }
        }
        
        llvm::outs() << "DEBUG: Kernel name: " << kernelName << "\n";
        llvm::outs() << "DEBUG: Kernel range: " << kernelRange.getBegin().printToString(*Result.SourceManager) 
                     << " to " << kernelRange.getEnd().printToString(*Result.SourceManager) << "\n";
        
        // Get the kernel source text
        std::string kernelText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(kernelRange), 
            *Result.SourceManager, 
            Result.Context->getLangOpts()).str();
        
        llvm::outs() << "DEBUG: Extracted kernel text length: " << kernelText.length() << "\n";
        llvm::outs() << "DEBUG: First 100 chars of kernel: " << kernelText.substr(0, 100) << "\n";
        
        // Check for duplicates - avoid adding the same kernel multiple times
        if (isDuplicateKernel(kernelName, kernelText)) {
            llvm::outs() << "DEBUG: Skipping duplicate kernel: " << kernelName << "\n";
            return;
        }
        
        // Store kernel info
        KernelInfo info;
        info.name = kernelName;
        info.content = kernelText;
        info.number = ++kernelCounter;
        info.sourceLocation = loc;
        
        kernels.push_back(info);
        
        llvm::outs() << "SUCCESS: Found project kernel #" << kernelCounter << ": " << kernelName 
                     << " at " << loc.printToString(*Result.SourceManager) << "\n";
    }
    
    const std::vector<KernelInfo>& getKernels() const {
        return kernels;
    }
    
private:
    std::vector<KernelInfo> kernels;
    int kernelCounter;
    std::string projectPath;
    
    // Check if a file path is within the project directory
    bool isWithinProjectDirectory(const std::string& filePath) {
        llvm::outs() << "DEBUG: Checking if file is within project directory\n";
        llvm::outs() << "DEBUG: File path: '" << filePath << "'\n";
        llvm::outs() << "DEBUG: Project path: '" << projectPath << "'\n";
        
        if (filePath.empty() || projectPath.empty()) {
            llvm::outs() << "DEBUG: Empty file path or project path\n";
            return false;
        }
        
        // Convert both paths to absolute paths for comparison
        std::error_code ec;
        std::string absFilePath = std::filesystem::canonical(filePath, ec);
        if (ec) {
            llvm::outs() << "DEBUG: canonical() failed for file path, trying absolute(): " << ec.message() << "\n";
            // If canonical fails, try absolute
            absFilePath = std::filesystem::absolute(filePath);
        }
        
        std::string absProjectPath = std::filesystem::canonical(projectPath, ec);
        if (ec) {
            llvm::outs() << "DEBUG: canonical() failed for project path, trying absolute(): " << ec.message() << "\n";
            absProjectPath = std::filesystem::absolute(projectPath);
        }
        
        llvm::outs() << "DEBUG: Absolute file path: '" << absFilePath << "'\n";
        llvm::outs() << "DEBUG: Absolute project path: '" << absProjectPath << "'\n";
        
        // Check if the file path starts with the project path
        bool result = absFilePath.find(absProjectPath) == 0;
        llvm::outs() << "DEBUG: isWithinProjectDirectory result: " << (result ? "true" : "false") << "\n";
        return result;
    }
    
    // Check for duplicate kernels based on name and content
    bool isDuplicateKernel(const std::string& kernelName, const std::string& kernelContent) {
        for (const auto& existingKernel : kernels) {
            if (existingKernel.name == kernelName && 
                existingKernel.content == kernelContent) {
                return true;
            }
        }
        return false;
    }
};

// Function to run make -n and extract compilation command
std::string extractCompilationCommand(const std::string& inputPath) {
    llvm::outs() << "DEBUG: Extracting compilation command from: " << inputPath << "\n";
    std::string command = "cd '" + inputPath + "' && make clean && make -n 2>&1";
    llvm::outs() << "DEBUG: Running command: " << command << "\n";
    
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        llvm::errs() << "ERROR: Failed to run make -n\n";
        return "";
    }
    
    std::string result;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    pclose(pipe);
    
    llvm::outs() << "DEBUG: Make output length: " << result.length() << "\n";
    llvm::outs() << "DEBUG: Make output (first 500 chars):\n" << result.substr(0, 500) << "\n";
    
    // Find the compilation command (look for lines with nvcc or gcc/g++/clang)
    std::istringstream iss(result);
    std::string line;
    std::string compilationCmd;
    int lineNum = 0;
    
    while (std::getline(iss, line)) {
        lineNum++;
        llvm::outs() << "DEBUG: Line " << lineNum << ": " << line << "\n";
        
        // Look for compilation commands (contains compiler and .cu/.cpp files)
        if ((line.find("nvcc") != std::string::npos || 
             line.find("gcc") != std::string::npos || 
             line.find("g++") != std::string::npos ||
             line.find("clang") != std::string::npos) &&
            (line.find(".cu") != std::string::npos || 
             line.find(".cpp") != std::string::npos ||
             line.find(".c ") != std::string::npos)) {
            llvm::outs() << "DEBUG: Found compilation command at line " << lineNum << ": " << line << "\n";
            compilationCmd = line;
            break;
        }
    }
    
    if (compilationCmd.empty()) {
        llvm::outs() << "DEBUG: No compilation command found in make output\n";
    } else {
        llvm::outs() << "DEBUG: Selected compilation command: " << compilationCmd << "\n";
    }
    
    return compilationCmd;
}

// Function to modify compilation command for clang analysis
std::string modifyCompilationCommand(const std::string& originalCmd, const std::string& inputPath) {
    std::string modifiedCmd = originalCmd;
    
    // Replace compiler with clang++-18
    std::regex nvccRegex(R"(\bnvcc\b)");
    std::regex gccRegex(R"(\bg\+\+\b)");
    std::regex clangRegex(R"(\bclang\+\+\b)");
    
    modifiedCmd = std::regex_replace(modifiedCmd, nvccRegex, "/usr/bin/clang++-18");
    modifiedCmd = std::regex_replace(modifiedCmd, gccRegex, "/usr/bin/clang++-18");
    modifiedCmd = std::regex_replace(modifiedCmd, clangRegex, "/usr/bin/clang++-18");
    
    // Remove CUDA-specific flags that clang might not understand
    std::vector<std::string> flagsToRemove = {
        "-arch=sm_[0-9]+", "-gencode", "arch=compute_[0-9]+,code=sm_[0-9]+",
        "-rdc=true", "-rdc=false", "--relocatable-device-code=true", "--relocatable-device-code=false",
        "-lcudart", "-lcuda", "-lcublas", "-lcurand", "-lcufft"
    };
    
    for (const auto& flag : flagsToRemove) {
        std::regex flagRegex(flag);
        modifiedCmd = std::regex_replace(modifiedCmd, flagRegex, "");
    }
    
    // Add required flags
    modifiedCmd += " -stdlib=libstdc++";
    modifiedCmd += " --cuda-path=/usr/local/cuda";
    modifiedCmd += " --offload-host-only";
    modifiedCmd += " -D__NO_CUDA__NOINLINE__";
    
    // Remove output specification and linking flags for AST generation
    std::regex outputRegex(R"(-o\s+\S+)");
    modifiedCmd = std::regex_replace(modifiedCmd, outputRegex, "");
    
    // Remove linking flags
    std::regex linkRegex(R"(-l\w+)");
    modifiedCmd = std::regex_replace(modifiedCmd, linkRegex, "");
    
    // Add syntax-only flag
    modifiedCmd += " -fsyntax-only";
    
    return modifiedCmd;
}

// Function to find source files in the input directory
std::vector<std::string> findSourceFiles(const std::string& inputPath) {
    std::vector<std::string> sourceFiles;
    
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(inputPath)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".cu" || ext == ".cpp" || ext == ".c" || ext == ".cxx") {
                    sourceFiles.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        llvm::errs() << "Error scanning directory: " << e.what() << "\n";
    }
    
    return sourceFiles;
}

// Function to write kernels to output file
void writeKernelsToFile(const std::vector<KernelInfo>& kernels, const std::string& outputPath) {
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        llvm::errs() << "Error: Could not open output file: " << outputPath << "\n";
        return;
    }
    
    for (const auto& kernel : kernels) {
        outFile << "============" << kernel.number << ":" << kernel.name << "============\n";
        outFile << kernel.content << "\n\n";
    }
    
    outFile.close();
    llvm::outs() << "Kernels written to: " << outputPath << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        llvm::errs() << "Usage: " << argv[0] << " <input-folder> <output-file>\n";
        llvm::errs() << "  input-folder: Path to the folder containing source code with Makefile\n";
        llvm::errs() << "  output-file:  Path where extracted kernels will be saved\n";
        return 1;
    }
    
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    
    // Check if input directory exists
    if (!std::filesystem::exists(inputPath) || !std::filesystem::is_directory(inputPath)) {
        llvm::errs() << "Error: Input directory does not exist: " << inputPath << "\n";
        return 1;
    }
    
    // Create output directory if needed
    std::filesystem::path outDir = std::filesystem::path(outputPath).parent_path();
    if (!outDir.empty()) {
        std::filesystem::create_directories(outDir);
    }
    
    llvm::outs() << "Extracting kernels from: " << inputPath << "\n";
    llvm::outs() << "Output file: " << outputPath << "\n";
    
    // Step 1: Extract compilation command
    llvm::outs() << "Step 1: Running make -n to extract compilation command...\n";
    std::string originalCmd = extractCompilationCommand(inputPath);
    if (originalCmd.empty()) {
        llvm::errs() << "Error: Could not extract compilation command from make -n\n";
        return 1;
    }
    
    llvm::outs() << "Original command: " << originalCmd << "\n";
    
    // Step 2: Modify compilation command
    llvm::outs() << "Step 2: Modifying compilation command for clang analysis...\n";
    std::string modifiedCmd = modifyCompilationCommand(originalCmd, inputPath);
    llvm::outs() << "Modified command: " << modifiedCmd << "\n";
    
    // Step 3: Find source files
    llvm::outs() << "Step 3: Finding source files...\n";
    std::vector<std::string> sourceFiles = findSourceFiles(inputPath);
    if (sourceFiles.empty()) {
        llvm::errs() << "Error: No source files found in: " << inputPath << "\n";
        return 1;
    }
    
    llvm::outs() << "Found " << sourceFiles.size() << " source files\n";
    
    // Step 4: Set up AST analysis
    llvm::outs() << "Step 4: Setting up AST analysis...\n";
    
    // Get absolute path for project directory
    std::string absInputPath = std::filesystem::absolute(inputPath);
    llvm::outs() << "DEBUG: Absolute input path: " << absInputPath << "\n";
    
    // Process each source file
    std::vector<KernelInfo> allKernels;
    int totalKernelCounter = 0;
    
    llvm::outs() << "DEBUG: Creating SimpleKernelExtractor with project path: " << absInputPath << "\n";
    SimpleKernelExtractor extractor(absInputPath);
    
    llvm::outs() << "DEBUG: Creating MatchFinder\n";
    MatchFinder finder;
    
    llvm::outs() << "DEBUG: Adding matcher for __global__ functions\n";
    auto matcher = functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("cudaKernel");
    finder.addMatcher(matcher, &extractor);
    llvm::outs() << "DEBUG: Matcher added successfully\n";
    
    for (const auto& sourceFile : sourceFiles) {
        llvm::outs() << "DEBUG: ===== Processing source file: " << sourceFile << " =====\n";
        
        // Create a simple compilation database for this file
        std::vector<std::string> args = {"/usr/bin/clang++-18"};
        
        // Add basic flags
        args.push_back("-x");
        args.push_back("cuda");
        args.push_back("-fsyntax-only");
        args.push_back("--cuda-path=/usr/local/cuda");
        args.push_back("-D__NO_CUDA__NOINLINE__");
        args.push_back("-nocudalib");
        args.push_back("-I/usr/local/cuda/include");
        
        llvm::outs() << "DEBUG: Basic args added\n";
        
        // Add include paths and other flags from the original command
        std::istringstream iss(modifiedCmd);
        std::string arg;
        int argCount = 0;
        while (iss >> arg) {
            argCount++;
            if (arg.find("-I") == 0 || arg.find("-D") == 0 || arg.find("-std=") == 0) {
                llvm::outs() << "DEBUG: Adding arg from modified command: " << arg << "\n";
                args.push_back(arg);
            }
        }
        llvm::outs() << "DEBUG: Processed " << argCount << " arguments from modified command\n";
        
        // Add the source file
        args.push_back(sourceFile);
        
        llvm::outs() << "DEBUG: Final compilation arguments:\n";
        for (size_t i = 0; i < args.size(); i++) {
            llvm::outs() << "  [" << i << "] " << args[i] << "\n";
        }
        
        // Create compilation database using the correct constructor
        llvm::outs() << "DEBUG: Creating FixedCompilationDatabase\n";
        FixedCompilationDatabase compilationDB(".", args);
        
        // Create ClangTool
        llvm::outs() << "DEBUG: Creating ClangTool\n";
        std::vector<std::string> sourceFilesVec = {sourceFile};
        ClangTool tool(compilationDB, sourceFilesVec);
        
        // Create frontend action factory
        llvm::outs() << "DEBUG: Creating KernelExtractionActionFactory\n";
        KernelExtractionActionFactory factory(&finder);
        
        // Run the tool
        llvm::outs() << "DEBUG: Running ClangTool on file: " << sourceFile << "\n";
        int result = tool.run(&factory);
        llvm::outs() << "DEBUG: ClangTool finished with exit code: " << result << "\n";
        
        if (result != 0) {
            llvm::errs() << "WARNING: Failed to process file: " << sourceFile << " (exit code: " << result << ")\n";
        } else {
            llvm::outs() << "DEBUG: Successfully processed file: " << sourceFile << "\n";
        }
        
        llvm::outs() << "DEBUG: Current kernel count: " << extractor.getKernels().size() << "\n";
    }
    
    // Get all kernels found and renumber them
    const auto& foundKernels = extractor.getKernels();
    for (auto kernel : foundKernels) {
        kernel.number = ++totalKernelCounter;
        allKernels.push_back(kernel);
    }
    
    // Step 5: Write results
    llvm::outs() << "Step 5: Writing results...\n";
    llvm::outs() << "Found " << allKernels.size() << " total kernels\n";
    
    if (allKernels.empty()) {
        llvm::outs() << "No CUDA kernels found in the source files.\n";
        // Still create the output file
        std::ofstream outFile(outputPath);
        outFile << "No CUDA kernels found.\n";
        outFile.close();
    } else {
        writeKernelsToFile(allKernels, outputPath);
    }
    
    llvm::outs() << "Kernel extraction completed successfully!\n";
    return 0;
}
