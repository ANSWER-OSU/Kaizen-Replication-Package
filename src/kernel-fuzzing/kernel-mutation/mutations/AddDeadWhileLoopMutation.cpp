#include "AddDeadWhileLoopMutation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"

void AddDeadWhileLoopMutation::apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;
    
    InsertionPointFinder Finder;
    Finder.TraverseDecl(KernelDecl);
    
    if (Finder.compoundStmts.empty()) return;
    
    // Load resources
    auto deadWhileLoops = resources_.loadDeadWhileLoops();
    auto deadCodeStatements = resources_.loadDeadCodeStatements();
    auto falseConditions = resources_.loadFalseConditions();
    
    if (deadWhileLoops.empty() || deadCodeStatements.empty()) {
        llvm::errs() << "Error: Could not load dead while loop patterns or dead code statements\n";
        return;
    }
    
    // Select a while loop template
    std::string deadLoopTemplate = resources_.getRandomElement(deadWhileLoops);
    
    // Get dead code and prepare replacements
    std::string deadCodeTemplate = resources_.getRandomElement(deadCodeStatements);
    std::map<std::string, std::string> replacements;
    replacements["code"] = resources_.replacePlaceholders(deadCodeTemplate, 
                          {{"var", resources_.generateRandomVariableName()}});
    
    // Add false condition for while loops
    if (!falseConditions.empty()) {
        replacements["false_condition"] = resources_.getRandomElement(falseConditions);
    }
    
    std::string deadLoop = resources_.replacePlaceholders(deadLoopTemplate, replacements);
    
    // Use the improved insertion point finding
    if (Finder.insertionPoints.empty()) {
        llvm::errs() << "No valid insertion points found\n";
        return;
    }
    
    // Select a random insertion point
    std::uniform_int_distribution<size_t> pointDist(0, Finder.insertionPoints.size() - 1);
    const auto& insertionPoint = Finder.insertionPoints[pointDist(rng_)];
    
    // Calculate indentation based on depth
    std::string indentation(insertionPoint.depth * 2, ' ');
    
    if (insertionPoint.beforeStmt) {
        // Insert before the specified statement
        clang::SourceLocation insertLoc = insertionPoint.beforeStmt->getBeginLoc();
        
        // Generate begin and end markers for mutation
        std::string beginMarker = indentation + generateBeginMarker("add_dead_while_loop");
        std::string endMarker = indentation + generateEndMarker("add_dead_while_loop");
        std::string formattedLoop = indentation + deadLoop + "\n";
        
        Rewrite.InsertTextBefore(insertLoc, beginMarker + formattedLoop + endMarker);
        llvm::outs() << "Added dead while loop before statement at depth " << insertionPoint.depth 
                     << ", position " << insertionPoint.position << "\n";
    } else {
        // Insert at the end of the block
        clang::SourceLocation insertLoc = insertionPoint.parentBlock->getRBracLoc();
        
        // Generate begin and end markers for mutation
        std::string beginMarker = indentation + generateBeginMarker("add_dead_while_loop");
        std::string endMarker = indentation + generateEndMarker("add_dead_while_loop");
        std::string formattedLoop = indentation + deadLoop + "\n";
        
        Rewrite.InsertTextBefore(insertLoc, beginMarker + formattedLoop + endMarker);
        llvm::outs() << "Added dead while loop at end of block at depth " << insertionPoint.depth << "\n";
    }
}
