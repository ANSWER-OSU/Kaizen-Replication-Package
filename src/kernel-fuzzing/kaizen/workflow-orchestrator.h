#ifndef WORKFLOW_ORCHESTRATOR_H
#define WORKFLOW_ORCHESTRATOR_H

#include <string>
#include <vector>
#include <random>
#include <chrono>

class WorkflowOrchestrator {
public:
    public:
    // Constructor takes benchmark name, RNG, and optional seed
    explicit WorkflowOrchestrator(const std::string& benchmarkName, std::mt19937& rng, uint32_t seed = 0);
    
    // Initialize the workflow - loads environment, validates paths, sets up directories
    bool initialize();
    
    // Run mutation loop with time limit (main entry point)
    int runMutationLoop(int runtimeSeconds);
    
    // Individual workflow stages
    int runStage1_KernelExtraction(const std::string& sourceFile);
    int runStage2_CompilationCheck(const std::string& mutatedFile);
    int runStage2p5_RuntimeTest(const std::string& mutatedFile);
    int runStage3_LLMTranslation(const std::string& mutatedFile, const std::string& outputFile);
    int runStage3p5_OMPTest(const std::string& translatedFile, const std::string& nextFileName);
    int runStage4_CrossFileReplacement(const std::string& translatedFile, const std::string& targetDir);

private:
    // Configuration
    std::string benchmarkName;
    std::string uuid;
    std::string sessionDir;
    std::string jsonOutputFilePath;
    std::string hostSrcDir;
    std::string uuidWithName;
    
    // Mutation loop directories
    std::string sourceQueueDir;
    std::string translatedOutputsDir; 
    std::string workingCopyDir;
    std::string compilationFailedDir;
    std::string runFailedDir;
    std::string translatedSuccessfullyOmpDir;
    std::string compilationFailedOmpDir;
    std::string runFailedOmpDir;
    int fileCounter;
    
    // Random number generator reference
    std::mt19937& rng;
    
    // Timing
    std::chrono::steady_clock::time_point startTime;
    
    // Environment variables
    std::string hecbenchSrcDir;
    std::string hostProjectDir;
    std::string translatorScript;
    std::string apptainerImage;
    std::string kaizenConfigFile;
    std::string containerAppDir;
    std::string containerDataDir;
    std::string kernelExtractorBin;
    std::string llmInferenceScriptDir;
    std::string llmInferenceInputFile;
    std::string llmInferenceOutputFile;
    std::string peftCheckpointDir;  // Optional PEFT checkpoint directory
    
    // Helper methods
    bool validateEnvironment();
    bool setupSessionDirectory();
    bool setupMutationDirectories();
    bool initializeSourceQueue();
    std::string selectRandomSourceFile();
    std::string getNextFileName();
    bool isTimeExpired(int runtimeSeconds) const;
    void printStageHeader(const std::string& stageName, int stageNumber = 0);
    std::string buildApptainerCommand(const std::vector<std::string>& programArgs);
    bool runMakeCommands();
    bool copyMutatedToHecbench();
    bool copyWorkingDirectory(const std::string& sourceFile);
};

#endif // WORKFLOW_ORCHESTRATOR_H
