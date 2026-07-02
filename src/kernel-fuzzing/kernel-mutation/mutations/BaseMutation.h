#ifndef BASE_MUTATION_H
#define BASE_MUTATION_H

#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include <string>
#include <vector>
#include <random>

// Helper class to find valid insertion points in kernel functions
class InsertionPointFinder : public clang::RecursiveASTVisitor<InsertionPointFinder> {
public:
    struct InsertionPoint {
        clang::CompoundStmt* parentBlock;
        clang::Stmt* beforeStmt;  // Insert before this statement (can be nullptr for end of block)
        int position;             // Position within the block
        int depth;               // Nesting depth from function root
        
        InsertionPoint(clang::CompoundStmt* parent, clang::Stmt* before, int pos, int d) 
            : parentBlock(parent), beforeStmt(before), position(pos), depth(d) {}
    };
    
    std::vector<clang::Stmt*> statements;
    std::vector<clang::CompoundStmt*> compoundStmts;
    std::vector<InsertionPoint> insertionPoints;
    
    bool VisitStmt(clang::Stmt* S) {
        statements.push_back(S);
        return true;
    }
    
    bool VisitCompoundStmt(clang::CompoundStmt* CS) {
        compoundStmts.push_back(CS);
        findInsertionPointsInBlock(CS);
        return true;
    }
    
private:
    int currentDepth = 0;
    
    void findInsertionPointsInBlock(clang::CompoundStmt* block) {
        currentDepth++;
        
        // Don't insert in the very first position of the function body if it's the main block
        bool isMainBlock = (currentDepth == 1);
        int startPos = isMainBlock ? 1 : 0;  // Skip first position for main function block
        
        int position = 0;
        for (auto stmt : block->body()) {
            if (position >= startPos) {
                // Add insertion point before this statement
                insertionPoints.emplace_back(block, stmt, position, currentDepth);
            }
            position++;
        }
        
        // Add insertion point at the end of the block (if not empty and not main block)
        if (!block->body().empty() && !isMainBlock) {
            insertionPoints.emplace_back(block, nullptr, position, currentDepth);
        }
        
        currentDepth--;
    }
};

// Base class for all mutations
class BaseMutation {
public:
    BaseMutation(std::mt19937& rng) : rng_(rng) {}
    virtual ~BaseMutation() = default;
    
    // Pure virtual function that each mutation must implement
    virtual void apply(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) = 0;
    
    // Get the name of the mutation type
    virtual std::string getName() const = 0;
    
    // Static methods to set iteration information for tracking
    static void setCurrentIteration(int iteration) { currentIteration_ = iteration; }
    static int getCurrentIteration() { return currentIteration_; }

protected:
    std::mt19937& rng_;
    
    // Static iteration tracking
    static int currentIteration_;
    
    // Helper function to generate triple-slash comment with iteration and line info
    std::string generateTripleSlashComment(const clang::SourceManager& SM, clang::SourceLocation startLoc, int numLines) {
        unsigned int lineNum = SM.getSpellingLineNumber(startLoc);
        unsigned int endLine = lineNum + numLines - 1;
        
        return "/// Iteration " + std::to_string(currentIteration_) + 
               " mutation: line " + std::to_string(lineNum) + " to " + std::to_string(endLine) + "\n";
    }
    
    // Helper function to generate begin marker for mutation
    std::string generateBeginMarker(const std::string& mutationType) {
        return "/// Begin of Iteration " + std::to_string(currentIteration_) + 
               " mutation: " + mutationType + "\n";
    }
    
    // Helper function to generate end marker for mutation  
    std::string generateEndMarker(const std::string& mutationType) {
        return "/// End of Iteration " + std::to_string(currentIteration_) + 
               " mutation: " + mutationType + "\n";
    }
    
    // Helper function to generate random variable names
    std::string generateRandomVariableName() {
        static const char* prefixes[] = {"dead_", "unused_", "tmp_", "temp_", "dummy_"};
        static const char* suffixes[] = {"var", "val", "data", "buf", "ptr"};
        
        std::uniform_int_distribution<int> prefixDist(0, 4);
        std::uniform_int_distribution<int> suffixDist(0, 4);
        std::uniform_int_distribution<int> numDist(1, 999);
        
        return std::string(prefixes[prefixDist(rng_)]) + 
               suffixes[suffixDist(rng_)] + 
               std::to_string(numDist(rng_));
    }
};

#endif // BASE_MUTATION_H
