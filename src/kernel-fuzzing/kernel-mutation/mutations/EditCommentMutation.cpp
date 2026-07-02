#include "EditCommentMutation.h"
#include "../../kaizen/config-reader.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <sstream>

std::vector<std::string> EditCommentMutation::getCommentList() const {
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

void EditCommentMutation::apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;
    
    // Get list of available replacement comments
    auto comments = getCommentList();
    if (comments.empty()) return;
    
    // Get the source manager and source range of the function
    clang::SourceManager& SM = Rewrite.getSourceMgr();
    clang::SourceRange funcRange = KernelDecl->getSourceRange();
    
    // Get the text of the entire function
    clang::SourceLocation startLoc = funcRange.getBegin();
    clang::SourceLocation endLoc = funcRange.getEnd();
    
    // Expand to include the full function body
    if (KernelDecl->hasBody()) {
        endLoc = KernelDecl->getBody()->getEndLoc();
        // Make sure we get the closing brace
        endLoc = clang::Lexer::getLocForEndOfToken(endLoc, 0, SM, clang::LangOptions());
    }
    
    // Get the file ID and check if locations are in the same file
    clang::FileID fileID = SM.getFileID(startLoc);
    if (fileID != SM.getFileID(endLoc)) {
        llvm::errs() << "Function spans multiple files, skipping comment editing\n";
        return;
    }
    
    // Get the buffer data
    bool invalid = false;
    const char* bufferStart = SM.getBufferData(fileID, &invalid).data();
    if (invalid) {
        llvm::errs() << "Could not get buffer data\n";
        return;
    }
    
    // Get offsets for start and end of function
    unsigned startOffset = SM.getFileOffset(startLoc);
    unsigned endOffset = SM.getFileOffset(endLoc);
    
    // Find single-line comments (//) within the function
    std::vector<std::pair<unsigned, unsigned>> commentRanges;
    
    for (unsigned i = startOffset; i < endOffset - 1; ++i) {
        if (bufferStart[i] == '/' && bufferStart[i + 1] == '/') {
            // Found start of single-line comment
            unsigned commentStart = i;
            unsigned commentEnd = i + 2;
            
            // Find end of comment (newline or end of buffer)
            while (commentEnd < endOffset && bufferStart[commentEnd] != '\n' && bufferStart[commentEnd] != '\r') {
                commentEnd++;
            }
            
            commentRanges.push_back(std::make_pair(commentStart, commentEnd));
            i = commentEnd; // Skip past this comment
        }
    }
    
    if (commentRanges.empty()) {
        llvm::outs() << "No single-line comments found to edit\n";
        return;
    }
    
    // Select a random comment to replace
    std::uniform_int_distribution<size_t> commentDist(0, commentRanges.size() - 1);
    auto selectedRange = commentRanges[commentDist(rng_)];
    
    // Select a random replacement comment
    std::uniform_int_distribution<size_t> replacementDist(0, comments.size() - 1);
    std::string selectedComment = comments[replacementDist(rng_)];
    
    // Create source locations for the comment range
    clang::SourceLocation commentStartLoc = SM.getLocForStartOfFile(fileID).getLocWithOffset(selectedRange.first);
    clang::SourceLocation commentEndLoc = SM.getLocForStartOfFile(fileID).getLocWithOffset(selectedRange.second);
    
    // Extract the original comment text
    std::string originalComment(bufferStart + selectedRange.first, selectedRange.second - selectedRange.first);
    
    // Replace the comment
    Rewrite.ReplaceText(clang::SourceRange(commentStartLoc, commentEndLoc), selectedComment);
    
    llvm::outs() << "Replaced comment '" << originalComment << "' with '" << selectedComment << "'\n";
}
