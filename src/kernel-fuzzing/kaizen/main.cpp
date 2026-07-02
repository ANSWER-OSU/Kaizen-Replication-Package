#include "../kernel-identification/KernelExtractor.h"
#include "../kernel-identification/kernel-identifier.h"
#include "workflow-orchestrator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include <iostream>
#include <string>
#include <chrono>
#include <random>

// Command-line argument for runtime in seconds
static llvm::cl::opt<int> RuntimeSeconds(llvm::cl::Positional, llvm::cl::desc("<runtime-seconds>"), llvm::cl::Required);

// Command-line argument for optional seed
static llvm::cl::opt<unsigned> Seed("seed", llvm::cl::desc("Random seed (uses time-based seed if not specified)"), llvm::cl::init(0));

// Command-line argument for benchmark name
static llvm::cl::opt<std::string> BenchmarkName("benchmark", llvm::cl::desc("Benchmark name for workflow"), llvm::cl::Required);

int main(int argc, const char **argv) {
    // print version
    std::cout << "Kernel Tool v0.2.0 - Mutation Loop Mode\n";
    
    llvm::cl::ParseCommandLineOptions(argc, argv);

    // Initialize random number generator
    unsigned actualSeed;
    if (Seed == 0) {
        actualSeed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::cout << "Using time-based seed: " << actualSeed << std::endl;
    } else {
        actualSeed = Seed;
        std::cout << "Using provided seed: " << actualSeed << std::endl;
    }
    
    std::mt19937 rng(actualSeed);
    
    // Create workflow orchestrator with mutation loop support, passing the seed for directory naming
    WorkflowOrchestrator orchestrator(BenchmarkName, rng, actualSeed);
    
    if (!orchestrator.initialize()) {
        std::cerr << "Error: Failed to initialize workflow" << std::endl;
        return 1;
    }
    
    // Run mutation loop with time limit
    std::cout << "Starting mutation loop for " << RuntimeSeconds << " seconds..." << std::endl;
    return orchestrator.runMutationLoop(RuntimeSeconds);
}