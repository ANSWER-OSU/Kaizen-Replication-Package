#ifndef REORDER_PREDICATES_MUTATION_H
#define REORDER_PREDICATES_MUTATION_H

#include "BaseMutation.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include <vector>
#include <string>
#include <random>

class ReorderPredicatesMutation : public BaseMutation {
public:
    explicit ReorderPredicatesMutation(std::mt19937& rng) : BaseMutation(rng) {}
    void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) override;
    std::string getName() const override { return "Reorder Predicates"; }

    struct ConditionalInfo {
        enum Type { IF_STMT, WHILE_LOOP, FOR_LOOP } type;
        clang::Stmt* stmt;
        clang::Expr* condition;
        std::vector<clang::Expr*> predicates;
        std::vector<std::string> operators; // &&, ||, etc.
        
        ConditionalInfo(Type t, clang::Stmt* s, clang::Expr* c) 
            : type(t), stmt(s), condition(c) {}
    };

private:
    std::vector<ConditionalInfo> findMultiPredicateConditionals(clang::FunctionDecl* KernelDecl);
    void extractPredicates(clang::Expr* condition, std::vector<clang::Expr*>& predicates, 
                          std::vector<std::string>& operators);
    bool canReorderPredicates(const ConditionalInfo& info);
    void reorderPredicates(const ConditionalInfo& info, clang::Rewriter& Rewrite);
    std::string reconstructCondition(const std::vector<clang::Expr*>& predicates, 
                                   const std::vector<std::string>& operators, 
                                   clang::Rewriter& Rewrite);
};

#endif // REORDER_PREDICATES_MUTATION_H
