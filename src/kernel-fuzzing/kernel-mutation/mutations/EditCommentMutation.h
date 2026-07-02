#ifndef EDIT_COMMENT_MUTATION_H
#define EDIT_COMMENT_MUTATION_H

#include "BaseMutation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include <vector>
#include <string>
#include <random>

class EditCommentMutation : public BaseMutation {
public:
    explicit EditCommentMutation(std::mt19937& rng) : BaseMutation(rng) {}
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    std::string getName() const override { return "Edit Comment"; }

private:
    std::vector<std::string> getCommentList() const;
};

#endif // EDIT_COMMENT_MUTATION_H
