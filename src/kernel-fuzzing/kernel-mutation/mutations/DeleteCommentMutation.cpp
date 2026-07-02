#include "DeleteCommentMutation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <regex>

void DeleteCommentMutation::apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;
    
    // Find all single line comments in the function
    std::vector<CommentLocation> comments = findSingleLineComments(KernelDecl, Rewrite);
    
    if (comments.empty()) {
        llvm::outs() << "No single line comments found to delete\n";
        return;
    }
    
    // Select a random comment to delete
    std::uniform_int_distribution<size_t> commentDist(0, comments.size() - 1);
    const auto& selectedComment = comments[commentDist(rng_)];
    
    // Remove the entire comment line
    Rewrite.RemoveText(clang::SourceRange(selectedComment.startLoc, selectedComment.endLoc));
    
    llvm::outs() << "Deleted single line comment at line " << selectedComment.lineNumber 
                 << ": " << selectedComment.content << "\n";
}

std::vector<DeleteCommentMutation::CommentLocation> DeleteCommentMutation::findSingleLineComments(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    std::vector<CommentLocation> comments;
    
    if (!KernelDecl->hasBody()) return comments;
    
    // Get the source manager and function body
    const clang::SourceManager& SM = Rewrite.getSourceMgr();
    clang::CompoundStmt* body = clang::dyn_cast<clang::CompoundStmt>(KernelDecl->getBody());
    if (!body) return comments;
    
    // Get the file ID and buffer for the source file
    clang::SourceLocation bodyStart = body->getBeginLoc();
    clang::FileID fileID = SM.getFileID(bodyStart);
    
    llvm::StringRef fileBuffer = SM.getBufferData(fileID);
    std::string fileContent = fileBuffer.str();
    
    // Get function body range to limit search
    clang::SourceRange bodyRange = body->getSourceRange();
    unsigned bodyStartOffset = SM.getFileOffset(bodyRange.getBegin());
    unsigned bodyEndOffset = SM.getFileOffset(bodyRange.getEnd());
    
    // Extract only the function body content
    std::string bodyContent = fileContent.substr(bodyStartOffset, bodyEndOffset - bodyStartOffset);
    
    // Regex to match single line comments (// but not ///)
    std::regex commentRegex(R"(^(\s*//(?!//).*)\n)");
    
    std::sregex_iterator iter(bodyContent.begin(), bodyContent.end(), commentRegex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        const std::smatch& match = *iter;
        
        // Calculate position in original file
        size_t matchPos = bodyStartOffset + match.position();
        size_t matchEnd = bodyStartOffset + match.position() + match.length();
        
        // Convert file positions to source locations
        clang::SourceLocation startLoc = SM.getLocForStartOfFile(fileID).getLocWithOffset(matchPos);
        clang::SourceLocation endLoc = SM.getLocForStartOfFile(fileID).getLocWithOffset(matchEnd);
        
        CommentLocation comment;
        comment.startLoc = startLoc;
        comment.endLoc = endLoc;
        comment.lineNumber = SM.getSpellingLineNumber(startLoc);
        comment.content = match[1].str(); // The comment content without newline
        
        comments.push_back(comment);
    }
    
    return comments;
}
