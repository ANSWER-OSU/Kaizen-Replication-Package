#include "RenameVariableMutation.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/raw_ostream.h"
#include <random>

using namespace clang;
using namespace clang::ast_matchers;

// Helper visitor to find local variable declarations
class VariableFinder : public RecursiveASTVisitor<VariableFinder> {
public:
    std::vector<VarDecl*> localVars;
    
    bool VisitVarDecl(VarDecl* VD) {
        if (VD && VD->isLocalVarDecl()) {
            localVars.push_back(VD);
            llvm::outs() << "Found local variable: " << VD->getNameAsString() << "\n";
        }
        return true;
    }
};

RenameVariableMutation::RenameVariableMutation(std::mt19937& rng)
    : BaseMutation(rng), resources_(rng), rewrite_(nullptr), targetDecl_(nullptr) {}

void RenameVariableMutation::apply(FunctionDecl* KernelDecl, Rewriter& Rewrite) {
    if (!KernelDecl || !KernelDecl->hasBody()) return;

    rewrite_ = &Rewrite;
    ASTContext& Context = KernelDecl->getASTContext();

    // 1. Find all local variable declarations in the function
    VariableFinder Finder;
    Finder.TraverseStmt(KernelDecl->getBody());

    if (Finder.localVars.empty()) {
        llvm::outs() << "No local variables to rename.\n";
        return;
    }

    // 2. Select one variable declaration at random
    std::uniform_int_distribution<size_t> dist(0, Finder.localVars.size() - 1);
    targetDecl_ = Finder.localVars[dist(rng_)];
    if (!targetDecl_) return;

    std::string oldName = targetDecl_->getNameAsString();
    
    // 3. Generate a new random name for the variable
    newName_ = generateRandomVariableName();

    // 4. Rename the original declaration
    SourceLocation nameLoc = targetDecl_->getLocation();
    rewrite_->ReplaceText(nameLoc, oldName.length(), newName_);

    // 5. Find all references to this variable and rename them
    MatchFinder RefFinder;
    // Matcher for all expressions that refer to our chosen variable declaration
    auto matcher = declRefExpr(to(varDecl(equalsNode(targetDecl_)))).bind("varRef");
    RefFinder.addMatcher(matcher, this);
    RefFinder.matchAST(Context);

    llvm::outs() << "Renamed variable '" << oldName << "' to '" << newName_ << "'\n";
}

// This callback is executed for each found reference
void RenameVariableMutation::run(const MatchFinder::MatchResult& Result) {
    if (const DeclRefExpr* DRE = Result.Nodes.getNodeAs<DeclRefExpr>("varRef")) {
        SourceLocation refLoc = DRE->getLocation();
        std::string oldName = DRE->getNameInfo().getAsString();
        rewrite_->ReplaceText(refLoc, oldName.length(), newName_);
    }
}
