#ifndef ADD_DEAD_FOR_LOOP_MUTATION_H
#define ADD_DEAD_FOR_LOOP_MUTATION_H

#include "BaseMutation.h"
#include "MutationResources.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <vector>
#include <string>

class AddDeadForLoopMutation : public BaseMutation {
public:
    AddDeadForLoopMutation(std::mt19937& rng) : BaseMutation(rng), resources_(rng) {}
    
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    std::string getName() const override { return "Add dead for loop"; }

private:
    MutationResources resources_;
};

#endif // ADD_DEAD_FOR_LOOP_MUTATION_H
