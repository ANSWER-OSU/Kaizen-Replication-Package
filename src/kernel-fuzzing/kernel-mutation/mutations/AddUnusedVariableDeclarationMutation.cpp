#include "AddUnusedVariableDeclarationMutation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"

void AddUnusedVariableDeclarationMutation::apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;
    
    InsertionPointFinder Finder;
    Finder.TraverseDecl(KernelDecl);
    
    if (Finder.compoundStmts.empty()) return;
    
    // Load resources
    auto variableTypes = resources_.loadVariableTypes();
    auto variableValues = resources_.loadVariableValues();
    
    if (variableTypes.empty()) {
        llvm::errs() << "Error: Could not load variable types\n";
        return;
    }
    
    // Select a random compound statement to insert the unused variable
    std::uniform_int_distribution<size_t> blockDist(0, Finder.compoundStmts.size() - 1);
    clang::CompoundStmt* targetBlock = Finder.compoundStmts[blockDist(rng_)];
    
    // Generate the unused variable declaration
    std::string type = resources_.getRandomElement(variableTypes);
    std::string varName = resources_.generateRandomVariableName();
    
    // 50% chance to initialize the variable
    std::uniform_int_distribution<int> initDist(0, 1);
    std::string varDecl;
    
    if (initDist(rng_) == 1 && variableValues.count(type) > 0) {
        // Initialize with a random value for this type
        std::string value = resources_.getRandomElement(variableValues.at(type));
        varDecl = type + " " + varName + " = " + value + ";";
    } else {
        // Just declare without initialization
        varDecl = type + " " + varName + ";";
    }
    
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
        std::string beginMarker = indentation + generateBeginMarker("add_unused_variable");
        std::string endMarker = indentation + generateEndMarker("add_unused_variable");
        std::string formattedDecl = indentation + varDecl + "\n";
        
        Rewrite.InsertTextBefore(insertLoc, beginMarker + formattedDecl + endMarker);
        llvm::outs() << "Added unused variable declaration before statement at depth " << insertionPoint.depth 
                     << ", position " << insertionPoint.position << ": " << varDecl << "\n";
    } else {
        // Insert at the end of the block
        clang::SourceLocation insertLoc = insertionPoint.parentBlock->getRBracLoc();
        
        // Generate begin and end markers for mutation
        std::string beginMarker = indentation + generateBeginMarker("add_unused_variable");
        std::string endMarker = indentation + generateEndMarker("add_unused_variable");
        std::string formattedDecl = indentation + varDecl + "\n";
        
        Rewrite.InsertTextBefore(insertLoc, beginMarker + formattedDecl + endMarker);
        llvm::outs() << "Added unused variable declaration at end of block at depth " << insertionPoint.depth 
                     << ": " << varDecl << "\n";
    }
}
