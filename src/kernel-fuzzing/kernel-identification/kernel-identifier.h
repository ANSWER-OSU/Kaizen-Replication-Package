#ifndef KERNEL_IDENTIFIER_H
#define KERNEL_IDENTIFIER_H

#include <string>
#include <vector>
#include <random>
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "../kernel-mutation/KernelMutator.h"

// Main kernel extraction function
int run_kernel_extractor(int argc, const char **argv);

// Get the output filename from the last run
std::string get_output_filename();

// Core kernel extraction logic without command-line parsing
int extract_kernels_from_path(const std::string& buildPath, const std::string& outputFilename, bool mutate, KernelMutatorFn mutator, const std::string& mutated_file_path);

#endif // KERNEL_IDENTIFIER_H
