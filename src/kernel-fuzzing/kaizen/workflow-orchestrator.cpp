#include "workflow-orchestrator.h"
#include "environment-config.h"
#include "system-utils.h"
#include "../kernel-mutation/KernelMutator.h"
#include "../kernel-mutation/mutations/BaseMutation.h"
#include "../kernel-identification/kernel-identifier.h"
#include <iostream>
#include <filesystem>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <chrono>

WorkflowOrchestrator::WorkflowOrchestrator(const std::string& benchmarkName, std::mt19937& rng, uint32_t seed) 
    : benchmarkName(benchmarkName), rng(rng), fileCounter(0) {
    
    // Generate timestamp string (YYYYMMDD-HHMMSS format)
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::ostringstream timestamp_stream;
    timestamp_stream << std::put_time(&tm, "%Y%m%d-%H%M%S");
    std::string timestamp = timestamp_stream.str();
    
    // Generate directory identifier using seed and timestamp
    std::string dirIdentifier;
    if (seed != 0) {
        dirIdentifier = "seed" + std::to_string(seed) + "-" + timestamp;
    } else {
        dirIdentifier = "noseed-" + timestamp;
    }
    
    uuid = dirIdentifier;  // Keep for compatibility with existing code
    uuidWithName = benchmarkName + "-cuda-" + dirIdentifier;
    sessionDir = "kaizen-output/" + benchmarkName + "-" + dirIdentifier;
    jsonOutputFilePath = sessionDir + "/" + benchmarkName + "-out.json";
    
    // Setup mutation loop directories
    sourceQueueDir = sessionDir + "/source-queue";
    translatedOutputsDir = sessionDir + "/translated-outputs";
    workingCopyDir = sessionDir + "/working-copy";
    compilationFailedDir = sessionDir + "/compilation-failed-cuda";
    runFailedDir = sessionDir + "/run-failed-cuda";
    translatedSuccessfullyOmpDir = sessionDir + "/translated-successfully-omp";
    compilationFailedOmpDir = sessionDir + "/compilation-failed-omp";
    runFailedOmpDir = sessionDir + "/run-failed-omp";
}

bool WorkflowOrchestrator::initialize() {
    std::cout << "=== Initializing Workflow for Benchmark: " << benchmarkName << " ===" << std::endl;
    
    // Validate environment and load variables
    if (!validateEnvironment()) {
        return false;
    }
    
    // Setup session directory
    if (!setupSessionDirectory()) {
        return false;
    }
    
    // Setup mutation loop directories
    if (!setupMutationDirectories()) {
        return false;
    }
    
    // Initialize source queue with original file
    if (!initializeSourceQueue()) {
        return false;
    }
    
    std::cout << ">>> Using benchmark: " << benchmarkName << std::endl;
    std::cout << ">>> Source directory: " << hostSrcDir << std::endl;
    std::cout << ">>> Session directory: " << sessionDir << std::endl;
    std::cout << ">>> Source queue: " << sourceQueueDir << std::endl;
    std::cout << ">>> Translated outputs: " << translatedOutputsDir << std::endl;
    std::cout << ">>> Compilation failed: " << compilationFailedDir << std::endl;
    std::cout << ">>> Runtime failed: " << runFailedDir << std::endl;
    std::cout << ">>> UUID: " << uuid << std::endl;
    
    return true;
}

bool WorkflowOrchestrator::validateEnvironment() {
    try {
        hecbenchSrcDir = EnvironmentConfig::getRequiredEnv("HECBENCH_SRC_DIR");
        hostProjectDir = EnvironmentConfig::getRequiredEnv("HOST_PROJECT_DIR");
        translatorScript = EnvironmentConfig::getRequiredEnv("TRANSLATOR_SCRIPT");
        apptainerImage = EnvironmentConfig::getRequiredEnv("APPTAINER_IMAGE");
        kaizenConfigFile = EnvironmentConfig::getRequiredEnv("KAIZEN_CONFIG_FILE");
        containerAppDir = EnvironmentConfig::getRequiredEnv("CONTAINER_APP_DIR");
        containerDataDir = EnvironmentConfig::getRequiredEnv("CONTAINER_DATA_DIR");
        kernelExtractorBin = EnvironmentConfig::getRequiredEnv("KERNEL_EXTRACTOR_BIN");
        
        // LLM Inference paths
        llmInferenceScriptDir = EnvironmentConfig::getRequiredEnv("LLM_INFERENCE_SCRIPT_DIR");
        llmInferenceInputFile = EnvironmentConfig::getRequiredEnv("LLM_INFERENCE_INPUT_FILE");
        llmInferenceOutputFile = EnvironmentConfig::getRequiredEnv("LLM_INFERENCE_OUTPUT_FILE");
        
        // Optional PEFT checkpoint directory
        peftCheckpointDir = EnvironmentConfig::getEnv("PEFT_CHECKPOINT_DIR", "");
        
        // Construct paths based on benchmark name
        hostSrcDir = hecbenchSrcDir + "/" + benchmarkName + "-cuda";
        
        // Validate that source directory exists
        if (!SystemUtils::exists(hostSrcDir)) {
            std::cerr << "Error: Source directory does not exist: " << hostSrcDir << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error validating environment: " << e.what() << std::endl;
        return false;
    }
}

bool WorkflowOrchestrator::setupSessionDirectory() {
    // When running in container, use the container app directory
    std::string baseDir = SystemUtils::getCurrentWorkingDirectory();
    
    // Check if we have the container app directory environment variable
    const char* containerAppDir = std::getenv("CONTAINER_APP_DIR");
    if (containerAppDir != nullptr) {
        baseDir = containerAppDir;
        std::cout << ">>> Using container app directory: " << baseDir << std::endl;
    } else {
        std::cout << ">>> Current working directory: " << baseDir << std::endl;
    }
    
    // sessionDir should be relative to the base directory (which is the bound project root)
    std::string fullSessionDir;
    if (sessionDir.front() == '/') {
        // Already absolute path
        fullSessionDir = sessionDir;
    } else {
        // Relative path - make it relative to base directory
        fullSessionDir = baseDir + "/" + sessionDir;
    }
    
    std::cout << ">>> Creating session directory: " << fullSessionDir << std::endl;
    
    // Create session directory
    if (!SystemUtils::createDirectories(fullSessionDir)) {
        std::cerr << "Error: Could not create session directory: " << fullSessionDir << std::endl;
        return false;
    }
    
    // Update sessionDir to be the full path
    sessionDir = fullSessionDir;
    
    // Update all directory paths to use the full session directory path
    sourceQueueDir = sessionDir + "/source-queue";
    translatedOutputsDir = sessionDir + "/translated-outputs";
    workingCopyDir = sessionDir + "/working-copy";
    compilationFailedDir = sessionDir + "/compilation-failed-cuda";
    runFailedDir = sessionDir + "/run-failed-cuda";
    translatedSuccessfullyOmpDir = sessionDir + "/translated-successfully-omp";
    compilationFailedOmpDir = sessionDir + "/compilation-failed-omp";
    runFailedOmpDir = sessionDir + "/run-failed-omp";
    
    // Update jsonOutputFilePath to use full path
    jsonOutputFilePath = sessionDir + "/" + benchmarkName + "-out.json";
    
    std::cout << ">>> JSON output file will be: " << jsonOutputFilePath << std::endl;
    
    return true;
}

int WorkflowOrchestrator::runStage1_KernelExtraction(const std::string& sourceFile) {
    printStageHeader("Kernel Extraction and Mutation", 1);
    
    // Use working copy directory as the source
    std::string sourceDir = workingCopyDir;
    if (sourceFile == "default") {
        sourceDir = hostSrcDir;
    }
    
    std::cout << "    Source: " << sourceDir << std::endl;
    
    // Build command arguments for the kernel extractor
    std::vector<std::string> args = {
        "kaizen",  // program name
        "--mutate",
        "-p", sourceDir,
        "-o", jsonOutputFilePath
    };
    
    // Convert to argc/argv format for the kernel extractor
    std::vector<const char*> argv;
    for (const auto& arg : args) {
        argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);
    
    std::cout << "Running kernel extraction with args: ";
    for (size_t i = 1; i < args.size(); ++i) {
        std::cout << args[i] << " ";
    }
    std::cout << std::endl;
    
    // Set environment variables for the extraction
    setenv("KAIZEN_SESSION_DIR", sessionDir.c_str(), 1);
    
    // Set the shared RNG seed for consistent mutation behavior
    setSharedRNGSeed(rng());
    
    // Call the kernel extractor directly
    KernelMutatorFn mutatorFn = static_cast<void(*)(clang::FunctionDecl*, clang::Rewriter&)>(mutateKernel);
    std::string mutatedDir = sessionDir + "/mutated";
    int result = extract_kernels_from_path(sourceDir, jsonOutputFilePath, true, mutatorFn, mutatedDir);
    
    if (result == 0) {
        std::cout << ">>> Kernel extraction complete." << std::endl;
    }
    
    return result;
}

int WorkflowOrchestrator::runStage2_CompilationCheck(const std::string& mutatedFile) {
    printStageHeader("Compilation Check", 2);
    std::cout << ">>> Checking if the mutated version compiles" << std::endl;
    
    // Copy the mutated file to working-copy/main.cu for compilation testing
    std::string mutatedSource = sessionDir + "/mutated/main.cu";
    std::string workingCopyMainCu = workingCopyDir + "/main.cu";
    
    if (!SystemUtils::copyFile(mutatedSource, workingCopyMainCu)) {
        std::cerr << "Error: Could not copy mutated file to working copy for compilation test" << std::endl;
        return 1;
    }
    
    // Run make commands in working copy directory to test compilation
    std::string originalDir = SystemUtils::getCurrentWorkingDirectory();
    if (!SystemUtils::changeWorkingDirectory(workingCopyDir)) {
        std::cerr << "Error: Could not change to working copy directory: " << workingCopyDir << std::endl;
        return 1;
    }
    
    // Run make clean and make
    int cleanResult = SystemUtils::runCommand("make clean", false);
    int makeResult = SystemUtils::runCommand("make", false);
    
    // Change back to original directory
    SystemUtils::changeWorkingDirectory(originalDir);
    
    if (makeResult != 0) {
        std::cout << ">>> Compilation failed - mutation will not be added to source queue" << std::endl;
        return 1;
    }
    
    std::cout << ">>> Compilation successful - mutation will be added to source queue" << std::endl;
    std::cout << ">>> Compilation check complete." << std::endl;
    return 0;
}

int WorkflowOrchestrator::runStage2p5_RuntimeTest(const std::string& mutatedFile) {
    printStageHeader("Runtime Test", 2);
    std::cout << ">>> Testing if the mutated version runs correctly" << std::endl;
    
    // Run make run command in working copy directory
    std::string originalDir = SystemUtils::getCurrentWorkingDirectory();
    if (!SystemUtils::changeWorkingDirectory(workingCopyDir)) {
        std::cerr << "Error: Could not change to working copy directory: " << workingCopyDir << std::endl;
        return 1;
    }
    
    // Run make run
    int runResult = SystemUtils::runCommand("make run", false);
    
    // Change back to original directory
    SystemUtils::changeWorkingDirectory(originalDir);
    
    if (runResult != 0) {
        std::cout << ">>> Runtime test failed - mutation will be placed in run-failed directory" << std::endl;
        return 1;
    }
    
    std::cout << ">>> Runtime test successful - mutation will be added to source queue" << std::endl;
    std::cout << ">>> Runtime test complete." << std::endl;
    return 0;
}

int WorkflowOrchestrator::runStage3p5_OMPTest(const std::string& translatedFile, const std::string& nextFileName) {
    printStageHeader("OMP Test", 3);
    
    // The Cross-File Kernel Replacement should have created the OMP version
    std::string ompSrcDir = hecbenchSrcDir + "/" + benchmarkName + "-omp";
    std::string ompMainFile = ompSrcDir + "/main.cpp";
    
    if (!SystemUtils::exists(ompSrcDir)) {
        std::cerr << "Error: OMP source directory does not exist: " << ompSrcDir << std::endl;
        std::cerr << "Cross-File Kernel Replacement may have failed." << std::endl;
        return 1;
    }
    
    if (!SystemUtils::exists(ompMainFile)) {
        std::cerr << "Error: OMP main file does not exist: " << ompMainFile << std::endl;
        return 1;
    }
    
    std::cout << ">>> Testing OMP version at: " << ompSrcDir << std::endl;
    
    // Change to OMP src directory for compilation and testing
    std::string originalDir = SystemUtils::getCurrentWorkingDirectory();
    if (!SystemUtils::changeWorkingDirectory(ompSrcDir)) {
        std::cerr << "Error: Failed to change to OMP directory: " << ompSrcDir << std::endl;
        return 1;
    }
    
    // Run make clean && make
    std::cout << ">>> Running make clean && make..." << std::endl;
    int cleanResult = SystemUtils::runCommand("make clean", false);
    if (cleanResult != 0) {
        std::cerr << "Warning: make clean failed" << std::endl;
    }
    
    int compileResult = SystemUtils::runCommand("make", false);
    
    if (compileResult != 0) {
        std::cout << ">>> OMP compilation failed - will be placed in compilation-failed-omp directory" << std::endl;
        // Change back to original directory
        SystemUtils::changeWorkingDirectory(originalDir);
        
        // Save to compilation-failed-omp directory
        std::string failedCompilationFile = compilationFailedOmpDir + "/" + nextFileName.substr(0, nextFileName.find_last_of('.')) + "-translated.cpp";
        if (SystemUtils::copyFile(ompMainFile, failedCompilationFile)) {
            std::cout << ">>> Saved compilation-failed OMP translation to: " << failedCompilationFile << std::endl;
        }
        return 1;
    }
    
    std::cout << ">>> OMP compilation successful, now running tests..." << std::endl;
    
    // Run make run
    int runResult = SystemUtils::runCommand("make run", false);
    
    // Change back to original directory
    SystemUtils::changeWorkingDirectory(originalDir);
    
    if (runResult != 0) {
        std::cout << ">>> OMP runtime test failed - will be placed in run-failed-omp directory" << std::endl;
        
        // Save to run-failed-omp directory
        std::string failedRuntimeFile = runFailedOmpDir + "/" + nextFileName.substr(0, nextFileName.find_last_of('.')) + "-translated.cpp";
        if (SystemUtils::copyFile(ompMainFile, failedRuntimeFile)) {
            std::cout << ">>> Saved runtime-failed OMP translation to: " << failedRuntimeFile << std::endl;
        }
        return 1;
    }
    
    std::cout << ">>> OMP test successful - will be placed in translated-successfully-omp directory" << std::endl;
    
    // Save to translated-successfully-omp directory
    std::string successfulTranslationFile = translatedSuccessfullyOmpDir + "/" + nextFileName.substr(0, nextFileName.find_last_of('.')) + "-translated.cpp";
    if (SystemUtils::copyFile(ompMainFile, successfulTranslationFile)) {
        std::cout << ">>> Saved successful OMP translation to: " << successfulTranslationFile << std::endl;
    }
    
    std::cout << ">>> OMP test complete." << std::endl;
    return 0;
}

int WorkflowOrchestrator::runStage3_LLMTranslation(const std::string& mutatedFile, const std::string& outputFile) {
    printStageHeader("LLM Translation", 3);
    
    std::cout << ">>> Debug: LLM paths configuration:" << std::endl;
    std::cout << "    Script dir: " << llmInferenceScriptDir << std::endl;
    std::cout << "    Input file: " << llmInferenceInputFile << std::endl;
    std::cout << "    Output file: " << llmInferenceOutputFile << std::endl;
    if (!peftCheckpointDir.empty()) {
        std::cout << "    PEFT checkpoint: " << peftCheckpointDir << std::endl;
    }
    
    // Set environment variables for the inference script
    if (!peftCheckpointDir.empty()) {
        setenv("PEFT_CHECKPOINT_DIR", peftCheckpointDir.c_str(), 1);
    }
    
    // Run the Python translation script with the proper arguments
    std::vector<std::string> args = {
        jsonOutputFilePath,
        "--inference-script-dir", llmInferenceScriptDir,
        "--inference-input-file", llmInferenceInputFile,
        "--inference-output-file", llmInferenceOutputFile
    };
    
    std::cout << ">>> Running Python script with args: ";
    for (const auto& arg : args) {
        std::cout << "'" << arg << "' ";
    }
    std::cout << std::endl;
    
    int result = SystemUtils::runPythonScript(translatorScript, args);
    
    if (result == 0) {
        std::cout << ">>> Translation complete." << std::endl;
    } else {
        std::cerr << "Translation script failed with exit code: " << result << std::endl;
    }
    
    return result;
}

int WorkflowOrchestrator::runStage4_CrossFileReplacement(const std::string& translatedFile, const std::string& targetDir) {
    printStageHeader("Cross-File Kernel Replacement", 4);
    
    // Use container paths since we're running inside the container
    std::string replacementScript = "/llm_inference/verifier/process_file.py";
    std::string workingDir = "/llm_inference/verifier";
    
    // Use the LLM output file path (this is already a container path)
    std::string jsonPath = llmInferenceOutputFile;
    
    std::vector<std::string> args = {
        "--json_path", jsonPath,
        "--number_samples", "1",
        "--config_path", "../config/omp-benchmarks.config",
        "--bench_path", hecbenchSrcDir
    };
    
    int result = SystemUtils::runPythonScript(replacementScript, args, workingDir);
    
    if (result == 0) {
        std::cout << ">>> Cross-file kernel replacement complete." << std::endl;
        
        // Copy the translated file from HeCBench source to translated-outputs
        if (!targetDir.empty()) {
            std::string sourceFile = hecbenchSrcDir + "/" + benchmarkName + "-omp/main.cpp";
            std::string destFile = translatedOutputsDir + "/" + targetDir;
            
            if (SystemUtils::copyFile(sourceFile, destFile)) {
                std::cout << ">>> Saved translated file to: " << destFile << std::endl;
            } else {
                std::cerr << "Warning: Failed to save translated file" << std::endl;
            }
        }
    } else {
        std::cerr << "Cross-file replacement failed with exit code: " << result << std::endl;
    }
    
    return result;
}

void WorkflowOrchestrator::printStageHeader(const std::string& stageName, int stageNumber) {
    std::cout << "\n>>> STAGE " << stageNumber << ": " << stageName << "..." << std::endl;
}

std::string WorkflowOrchestrator::buildApptainerCommand(const std::vector<std::string>& programArgs) {
    std::ostringstream cmd;
    
    cmd << "apptainer exec --nv";
    cmd << " --env KAIZEN_SESSION_DIR='" << sessionDir << "'";
    cmd << " --env KAIZEN_CONFIG_FILE='" << kaizenConfigFile << "'";
    cmd << " --bind '" << hostSrcDir << ":" << containerDataDir << "'";
    cmd << " --bind '" << hostProjectDir << ":" << containerAppDir << "'";
    cmd << " '" << apptainerImage << "'";
    cmd << " '" << kernelExtractorBin << "'";
    
    for (const auto& arg : programArgs) {
        if (arg.find(' ') != std::string::npos || arg.find('=') != std::string::npos) {
            cmd << " '" << arg << "'";
        } else {
            cmd << " " << arg;
        }
    }
    
    return cmd.str();
}

bool WorkflowOrchestrator::runMakeCommands() {
    std::string mutatedDir = sessionDir + "/mutated";
    
    // Run make clean
    int result = SystemUtils::runCommandInDir("make clean", mutatedDir);
    if (result != 0) {
        std::cerr << "Warning: make clean failed with exit code: " << result << std::endl;
    }
    
    // Run make libcuda.a
    result = SystemUtils::runCommandInDir("make libcuda.a", mutatedDir);
    if (result != 0) {
        std::cerr << "Error: make libcuda.a failed with exit code: " << result << std::endl;
        return false;
    }
    
    return true;
}

bool WorkflowOrchestrator::copyMutatedToHecbench() {
    std::string mutatedDir = sessionDir + "/mutated";
    std::string targetDir = hecbenchSrcDir + "/" + uuidWithName;
    
    // Create target directory
    if (!SystemUtils::createDirectories(targetDir)) {
        std::cerr << "Error: Could not create target directory: " << targetDir << std::endl;
        return false;
    }
    
    // Copy mutated source to hecbench for new seed
    if (!SystemUtils::copyDirectory(mutatedDir, targetDir)) {
        std::cerr << "Error: Could not copy mutated source to: " << targetDir << std::endl;
        return false;
    }
    
    std::cout << ">>> Copied mutated version to: " << targetDir << std::endl;
    return true;
}

int WorkflowOrchestrator::runMutationLoop(int runtimeSeconds) {
    startTime = std::chrono::steady_clock::now();
    
    std::cout << "=== Starting Sequential Mutation Loop ===" << std::endl;
    std::cout << "Runtime: " << runtimeSeconds << " seconds" << std::endl;
    
    int iteration = 0;
    int successfulMutations = 0;
    int successfulTranslations = 0;
    
    while (!isTimeExpired(runtimeSeconds)) {
        iteration++;
        std::cout << "\n--- Mutation Iteration " << iteration << " ---" << std::endl;
        
        // Select random source file from queue
        std::string selectedFile = selectRandomSourceFile();
        if (selectedFile.empty()) {
            std::cerr << "Error: No source files available in queue" << std::endl;
            break;
        }
        
        std::cout << "Selected source file: " << selectedFile << std::endl;
        
        
        // Copy working directory
        if (!copyWorkingDirectory(selectedFile)) {
            std::cerr << "Warning: Failed to prepare working directory" << std::endl;
            continue;
        }
        
        // Set iteration information for mutation tracking
        BaseMutation::setCurrentIteration(iteration);
        
        // Stage 1: Kernel extraction and mutation
        if (runStage1_KernelExtraction(selectedFile) != 0) {
            std::cerr << "Warning: Stage 1 failed" << std::endl;
            continue;
        }
        
        // Check if mutation actually produced a result
        std::string mutatedSource = sessionDir + "/mutated/main.cu";
        if (!SystemUtils::exists(mutatedSource)) {
            std::cout << "Info: No mutation produced" << std::endl;
            continue;
        }
        
        // Generate next file name for mutation output
        std::string nextFileName = getNextFileName();
        
	// Stage 2: Compilation check
        bool compilationPassed = (runStage2_CompilationCheck(nextFileName) == 0);
        
        if (!compilationPassed) {
            // Save to compilation-failed-cuda directory
            std::string failedCompilationFile = compilationFailedDir + "/" + nextFileName;
            if (SystemUtils::copyFile(mutatedSource, failedCompilationFile)) {
                std::cout << ">>> Saved compilation-failed mutation to: " << failedCompilationFile << std::endl;
            }
            continue;
        }
        
        // Stage 2.5: Runtime test
        bool runtimePassed = (runStage2p5_RuntimeTest(nextFileName) == 0);
        
        if (!runtimePassed) {
            // Save to run-failed-cuda directory
            std::string failedRuntimeFile = runFailedDir + "/" + nextFileName;
            if (SystemUtils::copyFile(mutatedSource, failedRuntimeFile)) {
                std::cout << ">>> Saved runtime-failed mutation to: " << failedRuntimeFile << std::endl;
            }
            continue;
        }
        
        // Both compilation and runtime passed - add to source queue for future mutations
        successfulMutations++;
        std::string mutatedFilePath = sourceQueueDir + "/" + nextFileName;
        if (!SystemUtils::copyFile(mutatedSource, mutatedFilePath)) {
            std::cerr << "Warning: Failed to copy successful mutation to source queue" << std::endl;
            continue;
        }
        
        std::cout << ">>> Added successful mutation to source queue: " << nextFileName << std::endl;
        // Removed LLM translation and subsequent stages for faster mutation loops
        // // Now process LLM translation for this successful mutation
        // std::cout << "\n>>> Starting LLM Translation for: " << nextFileName << " (iteration " << iteration << ")" << std::endl;
        
        // // Stage 3: LLM Translation
        // std::string translatedFile = translatedOutputsDir + "/" + 
        //     nextFileName.substr(0, nextFileName.find_last_of('.')) + "-translated.cpp";
        
        // if (runStage3_LLMTranslation(mutatedFilePath, translatedFile) != 0) {
        //     std::cerr << "Warning: LLM translation failed for " << nextFileName << std::endl;
        // } else {
        //     // Stage 4: Cross-File Kernel Replacement (creates the actual OMP file)
        //     std::string translatedOutputFile = nextFileName.substr(0, nextFileName.find_last_of('.')) + "-translated.cpp";
        //     if (runStage4_CrossFileReplacement(translatedFile, translatedOutputFile) != 0) {
        //         std::cerr << "Warning: Cross-file replacement failed for " << nextFileName << std::endl;
        //     } else {
        //         // Stage 3.5: OMP Test (test the OMP file created by cross-file replacement)
        //         if (runStage3p5_OMPTest(translatedFile, nextFileName) != 0) {
        //             std::cerr << "Info: OMP test failed for " << nextFileName << ", translation saved to appropriate failure directory" << std::endl;
        //         } else {
        //             successfulTranslations++;
        //             std::cout << ">>> Successfully completed LLM translation and OMP test for: " << nextFileName << std::endl;
        //         }
        //     }
        // }
        
        std::cout << "Successfully completed mutation iteration " << iteration << std::endl;
        
        // Check time after each complete iteration
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
        std::cout << "Elapsed time: " << elapsed << "/" << runtimeSeconds << " seconds" << std::endl;
    }
    
    std::cout << "\n=== Sequential Mutation Loop Complete ===" << std::endl;
    std::cout << "Total iterations: " << iteration << std::endl;
    std::cout << "Successful mutations: " << successfulMutations << std::endl;
    std::cout << "Successful translations: " << successfulTranslations << std::endl;
    
    return 0;
}

bool WorkflowOrchestrator::setupMutationDirectories() {
    // Create mutation loop directories
    if (!SystemUtils::createDirectories(sourceQueueDir)) {
        std::cerr << "Error: Could not create source queue directory: " << sourceQueueDir << std::endl;
        return false;
    }
    
    if (!SystemUtils::createDirectories(translatedOutputsDir)) {
        std::cerr << "Error: Could not create translated outputs directory: " << translatedOutputsDir << std::endl;
        return false;
    }
    
    if (!SystemUtils::createDirectories(workingCopyDir)) {
        std::cerr << "Error: Could not create working copy directory: " << workingCopyDir << std::endl;
        return false;
    }
    
    if (!SystemUtils::createDirectories(compilationFailedDir)) {
        std::cerr << "Error: Could not create compilation failed directory: " << compilationFailedDir << std::endl;
        return false;
    }
    
    if (!SystemUtils::createDirectories(runFailedDir)) {
        std::cerr << "Error: Could not create run failed directory: " << runFailedDir << std::endl;
        return false;
    }
    
    if (!SystemUtils::createDirectories(translatedSuccessfullyOmpDir)) {
        std::cerr << "Error: Could not create translated successfully OMP directory: " << translatedSuccessfullyOmpDir << std::endl;
        return false;
    }
    
    if (!SystemUtils::createDirectories(compilationFailedOmpDir)) {
        std::cerr << "Error: Could not create compilation failed OMP directory: " << compilationFailedOmpDir << std::endl;
        return false;
    }
    
    if (!SystemUtils::createDirectories(runFailedOmpDir)) {
        std::cerr << "Error: Could not create run failed OMP directory: " << runFailedOmpDir << std::endl;
        return false;
    }
    
    return true;
}

bool WorkflowOrchestrator::initializeSourceQueue() {
    // Copy original main.cu to source queue as main-000.cu
    std::string originalFile = hostSrcDir + "/main.cu";
    std::string queueFile = sourceQueueDir + "/main-000.cu";
    
    if (!std::filesystem::exists(originalFile)) {
        std::cerr << "Error: Original main.cu not found at: " << originalFile << std::endl;
        return false;
    }
    
    if (!SystemUtils::copyFile(originalFile, queueFile)) {
        std::cerr << "Error: Could not copy original file to queue: " << queueFile << std::endl;
        return false;
    }
    
    std::cout << ">>> Initialized source queue with: " << queueFile << std::endl;
    return true;
}

std::string WorkflowOrchestrator::selectRandomSourceFile() {
    // Get all files in source queue directory
    std::vector<std::string> sourceFiles;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(sourceQueueDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".cu") {
                sourceFiles.push_back(entry.path().filename().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading source queue directory: " << e.what() << std::endl;
        return "";
    }
    
    if (sourceFiles.empty()) {
        return "";
    }
    
    // Select random file using our RNG
    std::uniform_int_distribution<size_t> dist(0, sourceFiles.size() - 1);
    return sourceFiles[dist(rng)];
}

std::string WorkflowOrchestrator::getNextFileName() {
    fileCounter++;
    std::ostringstream oss;
    oss << "main-" << std::setfill('0') << std::setw(3) << fileCounter << ".cu";
    return oss.str();
}

bool WorkflowOrchestrator::isTimeExpired(int runtimeSeconds) const {
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
    return elapsed >= runtimeSeconds;
}

bool WorkflowOrchestrator::copyWorkingDirectory(const std::string& sourceFile) {
    // Clear and recreate working copy directory
    if (std::filesystem::exists(workingCopyDir)) {
        std::filesystem::remove_all(workingCopyDir);
    }
    
    if (!SystemUtils::createDirectories(workingCopyDir)) {
        return false;
    }
    
    // Copy entire source directory to working copy
    if (!SystemUtils::copyDirectory(hostSrcDir, workingCopyDir)) {
        std::cerr << "Error: Could not copy source directory to working copy" << std::endl;
        return false;
    }
    
    // Replace main.cu with selected source file
    std::string sourceFilePath = sourceQueueDir + "/" + sourceFile;
    std::string targetMainCu = workingCopyDir + "/main.cu";
    
    if (!SystemUtils::copyFile(sourceFilePath, targetMainCu)) {
        std::cerr << "Error: Could not copy selected source file to working directory" << std::endl;
        return false;
    }
    
    return true;
}
