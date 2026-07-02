#include "KernelExtractor.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <random>
#include <ctime>

using namespace clang;
using namespace clang::ast_matchers;

/*
 * matches "cudaKernel" functions, defined with the CUDAGlobal attribute, extracts name and range
 */
void KernelExtractor::run(const MatchFinder::MatchResult &Result) {
    if (!sourceManager) {
        sourceManager = Result.SourceManager;
        langOptions = &Result.Context->getLangOpts();
        astContext = Result.Context;
        mainFileID = sourceManager->getMainFileID();
        const FileEntry *MainFileEntry = sourceManager->getFileEntryForID(mainFileID);
        if (MainFileEntry) currentResult.sourceFile = MainFileEntry->tryGetRealPathName().str();
    }

    const auto *KernelFunc = Result.Nodes.getNodeAs<FunctionDecl>("cudaKernel");
    if (KernelFunc) {
        // Check if the kernel is from a locatable file
        SourceLocation loc = KernelFunc->getBeginLoc();
        if (loc.isInvalid()) return;

        // Use Clang's source manager to check if the kernel is in a system header.
        // This is more reliable than parsing file paths.
        if (sourceManager->isInSystemHeader(loc)) {
            return; // Skip kernels from system headers (e.g., /usr/include, CUDA includes)
        }

        // Debug print for accepted kernels
        llvm::outs() << "Found project kernel: " << KernelFunc->getNameAsString() 
                     << " at " << loc.printToString(*sourceManager) << "\n";

        std::string kernelName = KernelFunc->getNameAsString();
        SourceLocation beginLoc = KernelFunc->getBeginLoc();
        
        // Check for duplicates - avoid adding the same kernel multiple times
        bool isDuplicate = false;
        for (const auto& existingKernel : Kernels) {
            if (existingKernel.Name == kernelName && 
                existingKernel.Range.getBegin() == beginLoc) {
                isDuplicate = true;
                break;
            }
        }
        
        if (!isDuplicate) {
            // create a KernelInfo struct and add it to Kernels
            KernelInfo info;
            info.Name = kernelName;
            info.Range = KernelFunc->getSourceRange();
            info.Decl = const_cast<FunctionDecl*>(KernelFunc);
            Kernels.push_back(info);
        }
    }
}

/*
 * called at the end of translation file processing, generates FileResult currentResult from kernels
 */
void KernelExtractor::onEndOfTranslationUnit() {
    if (!sourceManager || Kernels.empty()) return;

    currentResult.sections.clear();
    std::sort(Kernels.begin(), Kernels.end());

    Rewriter Rewrite;
    Rewrite.setSourceMgr(*sourceManager, *langOptions);

    int kernelToMutateIndex = -1;
    if (mutator && !Kernels.empty()) {
        static std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::uniform_int_distribution<size_t> dist(0, Kernels.size() - 1);
        kernelToMutateIndex = dist(rng);
    }

    // Extract original kernel text for all kernels first
    for (size_t i = 0; i < Kernels.size(); i++) {
        auto& kernel = Kernels[i];
        SourceRange kernelRange(kernel.Range.getBegin(), kernel.Range.getEnd());
        std::string kernelText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(kernelRange), *sourceManager, *langOptions).str();
        
        // Store original text (will be overwritten for mutated kernel)
        kernel.mutatedText = kernelText;
    }

    // Apply mutation to selected kernel if needed
    if (mutator && kernelToMutateIndex >= 0) {
        auto& kernelToMutate = Kernels[kernelToMutateIndex];
        
        // Apply the mutation
        mutator(kernelToMutate.Decl, Rewrite);
        kernelToMutate.isMutated = true;
        
        // Extract the mutated kernel text from the rewriter
        const RewriteBuffer *RewriteBuf = Rewrite.getRewriteBufferFor(mainFileID);
        if (RewriteBuf) {
            // Get the complete rewritten file content
            std::string rewrittenContent;
            llvm::raw_string_ostream stream(rewrittenContent);
            RewriteBuf->write(stream);
            stream.flush();
            
            // Extract the mutated kernel function from the rewritten content
            std::string mutatedKernelText = extractKernelFromRewrittenContent(
                rewrittenContent, kernelToMutate.Name, kernelToMutate.Range);
            
            if (!mutatedKernelText.empty()) {
                kernelToMutate.mutatedText = mutatedKernelText;
            }
            
            // Store rewritten content for potential reprocessing
            currentResult.rewrittenFileContent = rewrittenContent;
            currentResult.needsReprocessing = true;
        }
    }

    // Add all kernels to the result using the appropriate text (original or mutated)
    for (const auto& kernel : Kernels) {
        currentResult.sections.push_back({"kernel", kernel.Name, kernel.mutatedText});
    }

    Kernels.clear();
    sourceManager = nullptr;
    langOptions = nullptr;
    astContext = nullptr;
}

void KernelExtractor::setCompileCommand(const std::string &command) {
    currentResult.compileCommand = command;
}

void KernelExtractor::setProjectPath(const std::string &path) {
    projectPath = path;
}

FileResult KernelExtractor::takeResult() {
    FileResult result = std::move(currentResult);
    currentResult = {}; // reset for next use (in next file)
    return result;
}

std::string KernelExtractor::extractKernelFromRewrittenContent(const std::string& rewrittenContent, 
                                                               const std::string& kernelName, 
                                                               const clang::SourceRange& originalRange) {
    // Look for the kernel function in the rewritten content
    // Start by finding __global__ declarations
    size_t searchPos = 0;
    while (searchPos < rewrittenContent.size()) {
        size_t globalPos = rewrittenContent.find("__global__", searchPos);
        if (globalPos == std::string::npos) break;
        
        // Look for the kernel name after __global__
        size_t namePos = rewrittenContent.find(kernelName, globalPos);
        if (namePos == std::string::npos || namePos > globalPos + 200) { // reasonable distance
            searchPos = globalPos + 10;
            continue;
        }
        
        // Check if this is actually the function name (not inside a comment or string)
        // Simple check: ensure it's followed by '('
        size_t parenPos = rewrittenContent.find('(', namePos + kernelName.length());
        size_t nextSpace = rewrittenContent.find(' ', namePos + kernelName.length());
        if (parenPos == std::string::npos || (nextSpace != std::string::npos && nextSpace < parenPos)) {
            searchPos = namePos + kernelName.length();
            continue;
        }
        
        // Find the opening brace of the function body
        size_t bracePos = rewrittenContent.find('{', parenPos);
        if (bracePos == std::string::npos) {
            searchPos = namePos + kernelName.length();
            continue;
        }
        
        // Find the matching closing brace
        int braceCount = 1;
        size_t endPos = bracePos + 1;
        while (endPos < rewrittenContent.size() && braceCount > 0) {
            if (rewrittenContent[endPos] == '{') braceCount++;
            else if (rewrittenContent[endPos] == '}') braceCount--;
            endPos++;
        }
        
        if (braceCount == 0) {
            // Find the start of the function declaration
            size_t startPos = globalPos;
            // Go back to find the beginning of the line or previous function
            while (startPos > 0 && rewrittenContent[startPos-1] != '\n') {
                startPos--;
            }
            
            std::string extractedKernel = rewrittenContent.substr(startPos, endPos - startPos);
            return extractedKernel;
        }
        
        searchPos = namePos + kernelName.length();
    }
    
    // If we couldn't find the kernel, return empty string
    return "";
}