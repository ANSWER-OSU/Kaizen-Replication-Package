#ifndef KERNEL_EXTRACTOR_H
#define KERNEL_EXTRACTOR_H

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <functional>
#include <string>
#include <vector>

using namespace std;

// Forward declaration
class KernelExtractor;

// Function pointer type for kernel mutation
using KernelMutatorFn = std::function<void(clang::FunctionDecl*, clang::Rewriter&)>;

/*
 * a section of code in the source file, either kernel or non-kernel
 */
struct Section {
    string type;
    string name;
    string content;
};

/*
 * a result of the kernel extraction process, contains source file location,
 * compile command, code sections
 */
struct FileResult {
    string sourceFile;
    string compileCommand;
    vector<Section> sections;
    string rewrittenFileContent; // For mutated files that need reprocessing
    bool needsReprocessing = false; // Flag to indicate if this result needs reprocessing
};

/*
 * called at the end of translation file processing, generates FileResult currentResult from kernels
 */
class KernelExtractor : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    // Constructor with optional mutator function
    KernelExtractor(KernelMutatorFn mutator = nullptr) : mutator(mutator) {}
    
    // Override to initialize the extractor using MatchFinder
    void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
    // 
    void onEndOfTranslationUnit() override;

    // Set the compile command for the current file
    void setCompileCommand(const std::string &command);
    
    // Set the project path for filtering kernels
    void setProjectPath(const std::string &path);
    
    // Set the mutator function
    void setMutator(KernelMutatorFn fn) { mutator = fn; }

    FileResult takeResult();

private:
    // Helper function to extract kernel text from rewritten content
    std::string extractKernelFromRewrittenContent(const std::string& rewrittenContent, 
                                                  const std::string& kernelName, 
                                                  const clang::SourceRange& originalRange);

    struct KernelInfo {
        string Name;
        clang::SourceRange Range;
        clang::FunctionDecl* Decl; // Store the actual FunctionDecl for AST access
        bool isMutated = false; // Flag to indicate if this kernel was mutated
        string mutatedText; // Store the mutated kernel text
        
        bool operator<(const KernelInfo &Other) const {
            return Range.getBegin() < Other.Range.getBegin();
        }
    };

    vector<KernelInfo> Kernels;
    clang::SourceManager *sourceManager = nullptr;
    const clang::LangOptions *langOptions = nullptr;
    clang::FileID mainFileID;
    clang::ASTContext *astContext = nullptr;
    std::string projectPath;
    
    FileResult currentResult;
    KernelMutatorFn mutator;
};

#endif
