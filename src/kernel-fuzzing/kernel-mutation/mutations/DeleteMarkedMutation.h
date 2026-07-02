#ifndef DELETE_MARKED_MUTATION_H
#define DELETE_MARKED_MUTATION_H

#include "BaseMutation.h"
#include "MutationResources.h"

class DeleteMarkedMutation : public BaseMutation {
public:
    DeleteMarkedMutation(std::mt19937& rng, const std::string& targetMutationType) 
        : BaseMutation(rng), resources_(rng), targetMutationType_(targetMutationType) {}
    
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    
    std::string getName() const override {
        return "Delete " + targetMutationType_ + " Mutation";
    }

private:
    MutationResources resources_;
    std::string targetMutationType_;
    
    struct MutationBlock {
        clang::SourceLocation startLoc;  // Location of the begin marker comment
        clang::SourceLocation endLoc;    // Location after the end marker comment
        int iterationNumber;
        int startLine;
        int endLine;
        std::string mutationType;        // Type of mutation (e.g., add_dead_if_stmt)
    };
    
    // Find all mutation blocks of the target type marked with begin/end markers
    std::vector<MutationBlock> findTargetMutationBlocks(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite);
};

#endif // DELETE_MARKED_MUTATION_H
