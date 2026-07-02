#include "AddCommentMutation.h"
#include "../../kaizen/config-reader.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <fstream>
#include <sstream>

std::vector<std::string> AddCommentMutation::getCommentList() const {
    std::vector<std::string> comments;
    
    // Get kaizen.cfg path from environment
    std::string kaizenConfigPath = ConfigReader::getKaizenConfigPath();
    if (kaizenConfigPath.empty()) {
        llvm::errs() << "Error: Cannot read comment options without kaizen.cfg path\n";
        return comments;
    }
    
    // Read comment file path from config
    std::string commentFilePath = ConfigReader::readConfigValue(kaizenConfigPath, "mutation", "comment_options_file");
    if (commentFilePath.empty()) {
        llvm::errs() << "Error: comment_options_file not found in kaizen.cfg\n";
        return comments;
    }
    
    std::ifstream file(commentFilePath);
    
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines and trim whitespace
            if (!line.empty() && line.find_first_not_of(" \t\r\n") != std::string::npos) {
                comments.push_back(line);
            }
        }
        file.close();
    } else {
        llvm::errs() << "Error: Could not read comment options file: " << commentFilePath << "\n";
    }
    
    return comments;
}

void AddCommentMutation::apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;
    
    InsertionPointFinder Finder;
    Finder.TraverseDecl(KernelDecl);
    
    // Get list of available comments
    auto comments = getCommentList();
    if (comments.empty()) return;
    
    // Select a random comment
    std::uniform_int_distribution<size_t> commentDist(0, comments.size() - 1);
    std::string selectedComment = comments[commentDist(rng_)];
    
    // Use the improved insertion point finding
    if (Finder.insertionPoints.empty()) {
        llvm::errs() << "No valid insertion points found\n";
        return;
    }
    
    // Select a random insertion point
    std::uniform_int_distribution<size_t> pointDist(0, Finder.insertionPoints.size() - 1);
    const auto& insertionPoint = Finder.insertionPoints[pointDist(rng_)];
    
    // Calculate indentation based on depth
    std::string indentation(insertionPoint.depth * 2, ' ');
    
    if (insertionPoint.beforeStmt) {
        // Insert before the specified statement
        clang::SourceLocation insertLoc = insertionPoint.beforeStmt->getBeginLoc();
        
        // Just add the comment without markers
        std::string formattedComment = indentation + selectedComment + "\n";
        
        Rewrite.InsertTextBefore(insertLoc, formattedComment);
        llvm::outs() << "Added comment before statement at depth " << insertionPoint.depth 
                     << ", position " << insertionPoint.position << ": " << selectedComment << "\n";
    } else {
        // Insert at the end of the block
        clang::SourceLocation insertLoc = insertionPoint.parentBlock->getRBracLoc();
        
        // Just add the comment without markers
        std::string formattedComment = indentation + selectedComment + "\n";
        
        Rewrite.InsertTextBefore(insertLoc, formattedComment);
        llvm::outs() << "Added comment at end of block at depth " << insertionPoint.depth 
                     << ": " << selectedComment << "\n";
    }
}
