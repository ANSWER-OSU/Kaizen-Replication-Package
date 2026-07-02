#include "AddDeadIfStmtMutation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"

void AddDeadIfStmtMutation::apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;
    
    InsertionPointFinder Finder;
    Finder.TraverseDecl(KernelDecl);
    
    if (Finder.compoundStmts.empty()) return;
    
    // Load resources
    auto falseConditions = resources_.loadFalseConditions();
    auto deadCodeStatements = resources_.loadDeadCodeStatements();
    
    if (falseConditions.empty() || deadCodeStatements.empty()) {
        llvm::errs() << "Error: Could not load false conditions or dead code statements\n";
        return;
    }
    
    // Select a random compound statement to insert the dead if
    std::uniform_int_distribution<size_t> blockDist(0, Finder.compoundStmts.size() - 1);
    clang::CompoundStmt* targetBlock = Finder.compoundStmts[blockDist(rng_)];
    
    // Generate the dead if statement
    std::string condition = resources_.getRandomElement(falseConditions);
    std::string deadCodeTemplate = resources_.getRandomElement(deadCodeStatements);
    
    // Replace placeholders in dead code
    std::map<std::string, std::string> replacements;
    replacements["var"] = resources_.generateRandomVariableName();
    std::string deadCode = resources_.replacePlaceholders(deadCodeTemplate, replacements);
    
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
    std::string bodyIndentation((insertionPoint.depth + 1) * 2, ' ');
    std::string formattedIfStmt = indentation + "if (" + condition + ") {\n" + 
                                  bodyIndentation + deadCode + "\n" + 
                                  indentation + "}\n";
    
    if (insertionPoint.beforeStmt) {
        // Insert before the specified statement
        clang::SourceLocation insertLoc = insertionPoint.beforeStmt->getBeginLoc();
        
        // Generate begin and end markers for mutation
        std::string beginMarker = indentation + generateBeginMarker("add_dead_if_stmt");
        std::string endMarker = indentation + generateEndMarker("add_dead_if_stmt");
        
        Rewrite.InsertTextBefore(insertLoc, beginMarker + formattedIfStmt + endMarker);
        llvm::outs() << "Added dead if statement before statement at depth " << insertionPoint.depth 
                     << ", position " << insertionPoint.position << " with condition: " << condition << "\n";
    } else {
        // Insert at the end of the block
        clang::SourceLocation insertLoc = insertionPoint.parentBlock->getRBracLoc();
        
        // Generate begin and end markers for mutation
        std::string beginMarker = indentation + generateBeginMarker("add_dead_if_stmt");
        std::string endMarker = indentation + generateEndMarker("add_dead_if_stmt");
        
        Rewrite.InsertTextBefore(insertLoc, beginMarker + formattedIfStmt + endMarker);
        llvm::outs() << "Added dead if statement at end of block at depth " << insertionPoint.depth 
                     << " with condition: " << condition << "\n";
    }
}
