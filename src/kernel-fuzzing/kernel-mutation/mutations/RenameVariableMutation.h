#ifndef RENAME_VARIABLE_MUTATION_H
#define RENAME_VARIABLE_MUTATION_H

#include "BaseMutation.h"
#include "MutationResources.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

class RenameVariableMutation : public BaseMutation, public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    explicit RenameVariableMutation(std::mt19937& rng);
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    std::string getName() const override { return "Rename Variable"; }

    // Callback for the AST Matcher
    void run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    MutationResources resources_;
    clang::Rewriter* rewrite_; // Pointer to the rewriter for use in the callback
    const clang::VarDecl* targetDecl_; // The specific declaration we are targeting
    std::string newName_; // The new name for the variable
};

#endif // RENAME_VARIABLE_MUTATION_H
