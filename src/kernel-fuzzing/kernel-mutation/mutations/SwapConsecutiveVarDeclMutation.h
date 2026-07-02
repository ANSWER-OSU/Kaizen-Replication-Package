#ifndef SWAP_CONSECUTIVE_VAR_DECL_MUTATION_H
#define SWAP_CONSECUTIVE_VAR_DECL_MUTATION_H

#include "BaseMutation.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Decl.h"
#include <vector>
#include <string>
#include <random>

class SwapConsecutiveVarDeclMutation : public BaseMutation {
public:
    explicit SwapConsecutiveVarDeclMutation(std::mt19937& rng) : BaseMutation(rng) {}
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    std::string getName() const override { return "Swap Consecutive Variable Declarations"; }

    struct VarDeclInfo {
        clang::VarDecl* decl;
        clang::DeclStmt* declStmt;
        clang::CompoundStmt* parentBlock;
        int position;
        
        VarDeclInfo(clang::VarDecl* d, clang::DeclStmt* ds, clang::CompoundStmt* pb, int pos)
            : decl(d), declStmt(ds), parentBlock(pb), position(pos) {}
    };

private:
    std::vector<VarDeclInfo> findConsecutiveVarDeclarations(clang::FunctionDecl* KernelDecl);
    bool canSwap(const VarDeclInfo& first, const VarDeclInfo& second);
    bool hasDataDependency(const VarDeclInfo& first, const VarDeclInfo& second);
};

#endif // SWAP_CONSECUTIVE_VAR_DECL_MUTATION_H
