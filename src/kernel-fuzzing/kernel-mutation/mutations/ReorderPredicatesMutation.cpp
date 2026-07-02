#include "ReorderPredicatesMutation.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "llvm/Support/raw_ostream.h"
#include <random>
#include <algorithm>

using namespace clang;

// Helper visitor to find conditional statements with multiple predicates
class ConditionalFinder : public RecursiveASTVisitor<ConditionalFinder> {
public:
    std::vector<ReorderPredicatesMutation::ConditionalInfo> conditionals;
    
    bool VisitIfStmt(IfStmt* IS) {
        if (IS->getCond()) {
            conditionals.emplace_back(ReorderPredicatesMutation::ConditionalInfo::IF_STMT, IS, IS->getCond());
        }
        return true;
    }
    
    bool VisitWhileStmt(WhileStmt* WS) {
        if (WS->getCond()) {
            conditionals.emplace_back(ReorderPredicatesMutation::ConditionalInfo::WHILE_LOOP, WS, WS->getCond());
        }
        return true;
    }
    
    bool VisitForStmt(ForStmt* FS) {
        if (FS->getCond()) {
            conditionals.emplace_back(ReorderPredicatesMutation::ConditionalInfo::FOR_LOOP, FS, FS->getCond());
        }
        return true;
    }
};

void ReorderPredicatesMutation::apply(FunctionDecl* KernelDecl, Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;

    // Find all conditional statements
    auto reorderableConditionals = findMultiPredicateConditionals(KernelDecl);
    
    if (reorderableConditionals.empty()) {
        llvm::outs() << "No conditional statements with multiple predicates found.\n";
        return;
    }

    // Select a random conditional to modify
    std::uniform_int_distribution<size_t> dist(0, reorderableConditionals.size() - 1);
    const ConditionalInfo& selectedConditional = reorderableConditionals[dist(rng_)];
    
    // Reorder the predicates
    reorderPredicates(selectedConditional, Rewrite);
    
    std::string stmtType;
    switch (selectedConditional.type) {
        case ConditionalInfo::IF_STMT: stmtType = "if statement"; break;
        case ConditionalInfo::WHILE_LOOP: stmtType = "while loop"; break;
        case ConditionalInfo::FOR_LOOP: stmtType = "for loop"; break;
    }
    
    llvm::outs() << "Reordered predicates in " << stmtType << " with " 
                 << selectedConditional.predicates.size() << " predicates\n";
}

std::vector<ReorderPredicatesMutation::ConditionalInfo> 
ReorderPredicatesMutation::findMultiPredicateConditionals(FunctionDecl* KernelDecl) {
    ConditionalFinder finder;
    finder.TraverseStmt(KernelDecl->getBody());
    
    std::vector<ConditionalInfo> result;
    
    for (auto& conditional : finder.conditionals) {
        // Extract predicates and operators from the condition
        extractPredicates(conditional.condition, conditional.predicates, conditional.operators);
        
        // Only include conditionals with multiple predicates that can be reordered
        if (conditional.predicates.size() > 1 && canReorderPredicates(conditional)) {
            result.push_back(conditional);
            
            std::string stmtType;
            switch (conditional.type) {
                case ConditionalInfo::IF_STMT: stmtType = "if"; break;
                case ConditionalInfo::WHILE_LOOP: stmtType = "while"; break;
                case ConditionalInfo::FOR_LOOP: stmtType = "for"; break;
            }
            
            llvm::outs() << "Found " << stmtType << " statement with " 
                         << conditional.predicates.size() << " predicates\n";
        }
    }
    
    return result;
}

void ReorderPredicatesMutation::extractPredicates(Expr* condition, 
                                                 std::vector<Expr*>& predicates, 
                                                 std::vector<std::string>& operators) {
    predicates.clear();
    operators.clear();
    
    if (!condition) return;
    
    // Handle binary operators (&&, ||)
    if (auto binOp = dyn_cast<BinaryOperator>(condition)) {
        if (binOp->getOpcode() == BO_LAnd || binOp->getOpcode() == BO_LOr) {
            // Recursively extract from left side
            extractPredicates(binOp->getLHS(), predicates, operators);
            
            // Add the operator
            operators.push_back(binOp->getOpcode() == BO_LAnd ? "&&" : "||");
            
            // Extract from right side
            std::vector<Expr*> rightPredicates;
            std::vector<std::string> rightOperators;
            extractPredicates(binOp->getRHS(), rightPredicates, rightOperators);
            
            // Merge right side results
            predicates.insert(predicates.end(), rightPredicates.begin(), rightPredicates.end());
            operators.insert(operators.end(), rightOperators.begin(), rightOperators.end());
            
            return;
        }
    }
    
    // Base case: single predicate
    predicates.push_back(condition);
}

bool ReorderPredicatesMutation::canReorderPredicates(const ConditionalInfo& info) {
    // We can reorder predicates if they are all connected by the same logical operator
    // This is a conservative approach to avoid changing semantics
    
    if (info.operators.empty()) return false;
    
    // Check if all operators are the same
    std::string firstOp = info.operators[0];
    for (const auto& op : info.operators) {
        if (op != firstOp) {
            llvm::outs() << "Cannot reorder: mixed operators (" << firstOp << " and " << op << ")\n";
            return false;
        }
    }
    
    // For && operations, reordering is generally safe due to short-circuiting behavior
    // For || operations, reordering is also generally safe
    // We allow both as the semantic change is usually minimal
    return true;
}

void ReorderPredicatesMutation::reorderPredicates(const ConditionalInfo& info, Rewriter& Rewrite) {
    if (info.predicates.size() < 2) return;
    
    // Create a copy of predicates to shuffle
    std::vector<Expr*> shuffledPredicates = info.predicates;
    
    // Shuffle the predicates
    std::shuffle(shuffledPredicates.begin(), shuffledPredicates.end(), rng_);
    
    // Make sure we actually changed the order
    bool changed = false;
    for (size_t i = 0; i < shuffledPredicates.size(); ++i) {
        if (shuffledPredicates[i] != info.predicates[i]) {
            changed = true;
            break;
        }
    }
    
    // If no change occurred, manually swap first two
    if (!changed && shuffledPredicates.size() >= 2) {
        std::swap(shuffledPredicates[0], shuffledPredicates[1]);
    }
    
    // Reconstruct the condition with reordered predicates
    std::string newCondition = reconstructCondition(shuffledPredicates, info.operators, Rewrite);
    
    // Replace the original condition
    SourceRange conditionRange = info.condition->getSourceRange();
    Rewrite.ReplaceText(conditionRange, newCondition);
    
    llvm::outs() << "Reordered predicates: " << newCondition << "\n";
}

std::string ReorderPredicatesMutation::reconstructCondition(const std::vector<Expr*>& predicates, 
                                                          const std::vector<std::string>& operators, 
                                                          Rewriter& Rewrite) {
    if (predicates.empty()) return "";
    if (predicates.size() == 1) {
        return Rewrite.getRewrittenText(predicates[0]->getSourceRange());
    }
    
    std::string result;
    
    for (size_t i = 0; i < predicates.size(); ++i) {
        if (i > 0 && i - 1 < operators.size()) {
            result += " " + operators[i - 1] + " ";
        }
        
        // Get the text of the predicate
        std::string predicateText = Rewrite.getRewrittenText(predicates[i]->getSourceRange());
        
        // Add parentheses around complex expressions for safety
        if (auto binOp = dyn_cast<BinaryOperator>(predicates[i])) {
            if (binOp->getOpcode() == BO_LAnd || binOp->getOpcode() == BO_LOr) {
                result += "(" + predicateText + ")";
            } else {
                result += predicateText;
            }
        } else {
            result += predicateText;
        }
    }
    
    return result;
}
