#ifndef DELETE_MUTATION_H
#define DELETE_MUTATION_H

#include "BaseMutation.h"
#include "MutationResources.h"

class DeleteMutation : public BaseMutation {
public:
    DeleteMutation(std::mt19937& rng) : BaseMutation(rng), resources_(rng) {}
    
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    
    std::string getName() const override {
        return "Delete Mutation";
    }

private:
    MutationResources resources_;
    
    struct MutationBlock {
        clang::SourceLocation startLoc;  // Location of the begin marker comment
        clang::SourceLocation endLoc;    // Location after the end marker comment
        int iterationNumber;
        int startLine;
        int endLine;
        std::string mutationType;        // Type of mutation (e.g., add_dead_if_stmt)
    };
    
    // Find all mutation blocks marked with begin/end markers
    std::vector<MutationBlock> findMutationBlocks(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite);
};

#endif // DELETE_MUTATION_H
