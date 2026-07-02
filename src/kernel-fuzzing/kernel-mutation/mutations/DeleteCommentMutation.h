#ifndef DELETE_COMMENT_MUTATION_H
#define DELETE_COMMENT_MUTATION_H

#include "BaseMutation.h"
#include "MutationResources.h"

class DeleteCommentMutation : public BaseMutation {
public:
    DeleteCommentMutation(std::mt19937& rng) : BaseMutation(rng), resources_(rng) {}
    
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    
    std::string getName() const override {
        return "Delete Comment Mutation";
    }

private:
    MutationResources resources_;
    
    struct CommentLocation {
        clang::SourceLocation startLoc;
        clang::SourceLocation endLoc;
        int lineNumber;
        std::string content;
    };
    
    // Find all single line comments (// style) in the function
    std::vector<CommentLocation> findSingleLineComments(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite);
};

#endif // DELETE_COMMENT_MUTATION_H
