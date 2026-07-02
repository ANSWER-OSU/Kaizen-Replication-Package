#include "DeleteMarkedMutation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <regex>

void DeleteMarkedMutation::apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;
    
    // Find all mutation blocks of the target type
    std::vector<MutationBlock> mutationBlocks = findTargetMutationBlocks(KernelDecl, Rewrite);
    
    if (mutationBlocks.empty()) {
        llvm::outs() << "No " << targetMutationType_ << " mutation blocks found to delete\n";
        return;
    }
    
    // Select a random mutation block to delete
    std::uniform_int_distribution<size_t> blockDist(0, mutationBlocks.size() - 1);
    const auto& selectedBlock = mutationBlocks[blockDist(rng_)];
    
    // Remove the entire block (from begin marker to end marker)
    Rewrite.RemoveText(clang::SourceRange(selectedBlock.startLoc, selectedBlock.endLoc));
    
    llvm::outs() << "Deleted " << selectedBlock.mutationType 
                 << " mutation block from iteration " << selectedBlock.iterationNumber 
                 << " (lines " << selectedBlock.startLine << " to " << selectedBlock.endLine << ")\n";
}

std::vector<DeleteMarkedMutation::MutationBlock> DeleteMarkedMutation::findTargetMutationBlocks(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    std::vector<MutationBlock> blocks;
    
    if (!KernelDecl->hasBody()) return blocks;
    
    // Get the source manager and function body
    const clang::SourceManager& SM = Rewrite.getSourceMgr();
    clang::CompoundStmt* body = clang::dyn_cast<clang::CompoundStmt>(KernelDecl->getBody());
    if (!body) return blocks;
    
    // Get the file ID and buffer for the source file
    clang::SourceLocation bodyStart = body->getBeginLoc();
    clang::FileID fileID = SM.getFileID(bodyStart);
    
    llvm::StringRef fileBuffer = SM.getBufferData(fileID);
    std::string fileContent = fileBuffer.str();
    
    // Regex to match begin markers with specific mutation type
    std::string beginPattern = R"(^(\s*///\s*Begin of Iteration\s+(\d+)\s+mutation:\s+)" + 
                               targetMutationType_ + R"(\s*\n))";
    std::regex beginMarkerRegex(beginPattern);
    
    std::sregex_iterator iter(fileContent.begin(), fileContent.end(), beginMarkerRegex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        const std::smatch& match = *iter;
        
        int iteration = std::stoi(match[2].str());
        
        // Find the position of this begin comment in the file
        size_t beginPos = match.position();
        
        // Look for the corresponding end marker
        std::string endMarkerPattern = R"(^(\s*///\s*End of Iteration\s+)" + std::to_string(iteration) + 
                                      R"(\s+mutation:\s+)" + targetMutationType_ + R"(\s*\n)";
        std::regex endMarkerRegex(endMarkerPattern);
        
        // Search for end marker after the begin marker
        std::string remainingContent = fileContent.substr(beginPos + match.length());
        std::smatch endMatch;
        
        if (std::regex_search(remainingContent, endMatch, endMarkerRegex)) {
            // Calculate the end position
            size_t endPos = beginPos + match.length() + endMatch.position() + endMatch.length();
            
            // Convert file positions to source locations
            clang::SourceLocation startLoc = SM.getLocForStartOfFile(fileID).getLocWithOffset(beginPos);
            clang::SourceLocation endLoc = SM.getLocForStartOfFile(fileID).getLocWithOffset(endPos);
            
            // Validate that these locations are within the function body
            clang::SourceRange bodyRange = body->getSourceRange();
            if (SM.isBeforeInTranslationUnit(startLoc, bodyRange.getBegin()) ||
                SM.isBeforeInTranslationUnit(bodyRange.getEnd(), endLoc)) {
                continue; // Skip if outside function body
            }
            
            MutationBlock block;
            block.startLoc = startLoc;
            block.endLoc = endLoc;
            block.iterationNumber = iteration;
            block.mutationType = targetMutationType_;
            block.startLine = SM.getSpellingLineNumber(startLoc);
            block.endLine = SM.getSpellingLineNumber(endLoc);
            
            blocks.push_back(block);
        }
    }
    
    return blocks;
}
