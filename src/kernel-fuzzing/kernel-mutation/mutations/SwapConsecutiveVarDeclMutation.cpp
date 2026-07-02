#include "SwapConsecutiveVarDeclMutation.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/raw_ostream.h"
#include <random>
#include <algorithm>

using namespace clang;

// Helper visitor to find variable declarations and their positions
class VarDeclFinder : public RecursiveASTVisitor<VarDeclFinder> {
public:
    std::vector<SwapConsecutiveVarDeclMutation::VarDeclInfo> varDecls;
    
    bool VisitCompoundStmt(CompoundStmt* CS) {
        int position = 0;
        for (auto stmt : CS->body()) {
            if (auto declStmt = dyn_cast<DeclStmt>(stmt)) {
                // Check if this DeclStmt contains a single VarDecl
                if (declStmt->isSingleDecl()) {
                    if (auto varDecl = dyn_cast<VarDecl>(declStmt->getSingleDecl())) {
                        varDecls.emplace_back(varDecl, declStmt, CS, position);
                    }
                }
            }
            position++;
        }
        return true;
    }
};

// Helper visitor to check for dependencies between variables
class DependencyChecker : public RecursiveASTVisitor<DependencyChecker> {
public:
    std::string targetVarName;
    bool found = false;
    
    DependencyChecker(const std::string& varName) : targetVarName(varName) {}
    
    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
        if (auto varDecl = dyn_cast<VarDecl>(DRE->getDecl())) {
            if (varDecl->getNameAsString() == targetVarName) {
                found = true;
                return false; // Stop traversal
            }
        }
        return true;
    }
};

void SwapConsecutiveVarDeclMutation::apply(FunctionDecl* KernelDecl, Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;

    // Find all variable declarations with their positions
    auto swappablePairs = findConsecutiveVarDeclarations(KernelDecl);
    
    if (swappablePairs.empty()) {
        llvm::outs() << "No consecutive variable declarations found that can be swapped.\n";
        return;
    }

    // Select a random pair to swap
    std::uniform_int_distribution<size_t> dist(0, swappablePairs.size() - 1);
    size_t pairIndex = dist(rng_) / 2 * 2; // Ensure we get an even index (pairs are stored consecutively)
    
    const VarDeclInfo& first = swappablePairs[pairIndex];
    const VarDeclInfo& second = swappablePairs[pairIndex + 1];
    
    // Get the source text for both declarations
    SourceManager& SM = Rewrite.getSourceMgr();
    
    SourceRange firstRange = first.declStmt->getSourceRange();
    SourceRange secondRange = second.declStmt->getSourceRange();
    
    // Get the text of both declarations
    std::string firstText = Rewrite.getRewrittenText(firstRange);
    std::string secondText = Rewrite.getRewrittenText(secondRange);
    
    // Replace the first declaration with the second
    Rewrite.ReplaceText(firstRange, secondText);
    
    // Replace the second declaration with the first
    Rewrite.ReplaceText(secondRange, firstText);
    
    llvm::outs() << "Swapped consecutive variable declarations: '" 
                 << first.decl->getNameAsString() << "' and '" 
                 << second.decl->getNameAsString() << "'\n";
}

std::vector<SwapConsecutiveVarDeclMutation::VarDeclInfo> 
SwapConsecutiveVarDeclMutation::findConsecutiveVarDeclarations(FunctionDecl* KernelDecl) {
    VarDeclFinder finder;
    finder.TraverseStmt(KernelDecl->getBody());
    
    std::vector<VarDeclInfo> swappablePairs;
    
    // Group by parent block and find consecutive declarations
    std::map<CompoundStmt*, std::vector<VarDeclInfo*>> blockDecls;
    for (auto& declInfo : finder.varDecls) {
        blockDecls[declInfo.parentBlock].push_back(&declInfo);
    }
    
    // Check each block for consecutive declarations
    for (auto& [block, decls] : blockDecls) {
        // Sort by position
        std::sort(decls.begin(), decls.end(), 
                  [](const VarDeclInfo* a, const VarDeclInfo* b) {
                      return a->position < b->position;
                  });
        
        // Find consecutive pairs
        for (size_t i = 0; i < decls.size() - 1; ++i) {
            const VarDeclInfo& first = *decls[i];
            const VarDeclInfo& second = *decls[i + 1];
            
            // Check if they are consecutive (position differs by 1)
            if (second.position == first.position + 1) {
                // Check if they can be safely swapped
                if (canSwap(first, second)) {
                    swappablePairs.push_back(first);
                    swappablePairs.push_back(second);
                    llvm::outs() << "Found swappable consecutive declarations: " 
                                 << first.decl->getNameAsString() << " and " 
                                 << second.decl->getNameAsString() << "\n";
                }
            }
        }
    }
    
    return swappablePairs;
}

bool SwapConsecutiveVarDeclMutation::canSwap(const VarDeclInfo& first, const VarDeclInfo& second) {
    // Check for data dependencies between the two declarations
    return !hasDataDependency(first, second);
}

bool SwapConsecutiveVarDeclMutation::hasDataDependency(const VarDeclInfo& first, const VarDeclInfo& second) {
    // Check if second declaration depends on first variable
    if (second.decl->hasInit()) {
        DependencyChecker checker(first.decl->getNameAsString());
        checker.TraverseStmt(second.decl->getInit());
        if (checker.found) {
            llvm::outs() << "Cannot swap: " << second.decl->getNameAsString() 
                         << " depends on " << first.decl->getNameAsString() << "\n";
            return true;
        }
    }
    
    // Check if first declaration depends on second variable (shouldn't happen for consecutive, but check anyway)
    if (first.decl->hasInit()) {
        DependencyChecker checker(second.decl->getNameAsString());
        checker.TraverseStmt(first.decl->getInit());
        if (checker.found) {
            llvm::outs() << "Cannot swap: " << first.decl->getNameAsString() 
                         << " depends on " << second.decl->getNameAsString() << "\n";
            return true;
        }
    }
    
    return false; // No dependency found, can swap
}
