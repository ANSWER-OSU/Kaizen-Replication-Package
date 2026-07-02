#ifndef KERNEL_MUTATOR_H
#define KERNEL_MUTATOR_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Basic/SourceManager.h"
#include "mutations/BaseMutation.h"
#include <vector>
#include <random>
#include <ctime>
#include <functional>

// Function pointer type for kernel mutation
using KernelMutatorFn = std::function<void(clang::FunctionDecl*, clang::Rewriter&)>;

// Combined mutation function that randomly selects one type of mutation
void mutateKernel(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite);

// Version that accepts external RNG
void mutateKernel(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite, std::mt19937& externalRng);

// Set the shared RNG seed (for workflow orchestrator)
void setSharedRNGSeed(unsigned int seed);

#endif // KERNEL_MUTATOR_H
