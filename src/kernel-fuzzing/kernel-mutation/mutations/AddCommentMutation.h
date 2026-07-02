#ifndef ADD_COMMENT_MUTATION_H
#define ADD_COMMENT_MUTATION_H

#include "BaseMutation.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <vector>
#include <string>

class AddCommentMutation : public BaseMutation {
public:
    AddCommentMutation(std::mt19937& rng) : BaseMutation(rng) {}
    
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    std::string getName() const override { return "Add single-line comment"; }

private:
    // Load comments from external file
    std::vector<std::string> getCommentList() const;
};

#endif // ADD_COMMENT_MUTATION_H
