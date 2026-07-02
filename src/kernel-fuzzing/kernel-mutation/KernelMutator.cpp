#include "KernelMutator.h"
#include "../kaizen/config-reader.h"
#include "mutations/AddCommentMutation.h"
#include "mutations/EditCommentMutation.h"
#include "mutations/AddDeadIfStmtMutation.h"
#include "mutations/AddDeadForLoopMutation.h"
#include "mutations/AddDeadWhileLoopMutation.h"
#include "mutations/AddUnusedVariableDeclarationMutation.h"
#include "mutations/RenameVariableMutation.h"
#include "mutations/SwapConsecutiveVarDeclMutation.h"
#include "mutations/ReorderPredicatesMutation.h"
#include "mutations/DeleteMutation.h"
#include "mutations/DeleteCommentMutation.h"
#include "mutations/DeleteMarkedMutation.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <random>
#include <ctime>
#include <memory>
#include <vector>
#include <map>

using namespace clang;

// Initialize random number generator based on kaizen.cfg
static std::mt19937 initializeRNG() {
    std::string kaizenConfigPath = ConfigReader::getKaizenConfigPath();
    if (kaizenConfigPath.empty()) {
        llvm::errs() << "Error: Cannot initialize RNG without kaizen.cfg path\n";
        std::exit(1);
    }
    
    std::string useTimeSeed = ConfigReader::readConfigValue(kaizenConfigPath, "seed", "use_time_seed");
    std::string fixedSeed = ConfigReader::readConfigValue(kaizenConfigPath, "seed", "fixed_seed");
    
    if (useTimeSeed == "false" && !fixedSeed.empty()) {
        unsigned int seed = static_cast<unsigned int>(std::stoul(fixedSeed));
        llvm::outs() << "Using fixed seed: " << seed << "\n";
        return std::mt19937(seed);
    } else {
        unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
        llvm::outs() << "Using time-based seed: " << seed << "\n";
        return std::mt19937(seed);
    }
}

static std::mt19937 rng = initializeRNG();

void mutateKernel(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite) {
    if (!KernelDecl) return;
    
    // Get kaizen.cfg path from environment
    std::string kaizenConfigPath = ConfigReader::getKaizenConfigPath();
    if (kaizenConfigPath.empty()) {
        llvm::errs() << "Error: Cannot read mutation config without kaizen.cfg path\n";
        return;
    }
    
    // Get mutation config file path from kaizen.cfg
    std::string mutationConfigFile = ConfigReader::readConfigValue(kaizenConfigPath, "mutation", "mutation_options_config");
    if (mutationConfigFile.empty()) {
        llvm::errs() << "Error: mutation_options_config not found in kaizen.cfg\n";
        return;
    }
    
    // Create available mutations based on weights
    std::vector<std::unique_ptr<BaseMutation>> mutations;
    
    // Check mutation weights and create mutations
    double addCommentWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_comment_weight");
    if (addCommentWeight > 0.0) {
        mutations.push_back(std::make_unique<AddCommentMutation>(rng));
        llvm::outs() << "Enabled add_comment mutation with weight: " << addCommentWeight << "\n";
    }
    
    double editCommentWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "edit_comment_weight");
    if (editCommentWeight > 0.0) {
        mutations.push_back(std::make_unique<EditCommentMutation>(rng));
        llvm::outs() << "Enabled edit_comment mutation with weight: " << editCommentWeight << "\n";
    }
    
    double addDeadIfWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_dead_if_stmt_weight");
    if (addDeadIfWeight > 0.0) {
        mutations.push_back(std::make_unique<AddDeadIfStmtMutation>(rng));
        llvm::outs() << "Enabled add_dead_if_stmt mutation with weight: " << addDeadIfWeight << "\n";
    }
    
    double addDeadForLoopWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_dead_for_loop_weight");
    if (addDeadForLoopWeight > 0.0) {
        mutations.push_back(std::make_unique<AddDeadForLoopMutation>(rng));
        llvm::outs() << "Enabled add_dead_for_loop mutation with weight: " << addDeadForLoopWeight << "\n";
    }
    
    double addDeadWhileLoopWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_dead_while_loop_weight");
    if (addDeadWhileLoopWeight > 0.0) {
        mutations.push_back(std::make_unique<AddDeadWhileLoopMutation>(rng));
        llvm::outs() << "Enabled add_dead_while_loop mutation with weight: " << addDeadWhileLoopWeight << "\n";
    }
    
    double addUnusedVarWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_unused_variable_weight");
    if (addUnusedVarWeight > 0.0) {
        mutations.push_back(std::make_unique<AddUnusedVariableDeclarationMutation>(rng));
        llvm::outs() << "Enabled add_unused_variable mutation with weight: " << addUnusedVarWeight << "\n";
    }
    
    double renameVariableWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "rename_variable_weight");
    if (renameVariableWeight > 0.0) {
        mutations.push_back(std::make_unique<RenameVariableMutation>(rng));
        llvm::outs() << "Enabled rename_variable mutation with weight: " << renameVariableWeight << "\n";
    }
    
    double swapConsecutiveVarDeclWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "swap_consecutive_var_decl_weight");
    if (swapConsecutiveVarDeclWeight > 0.0) {
        mutations.push_back(std::make_unique<SwapConsecutiveVarDeclMutation>(rng));
        llvm::outs() << "Enabled swap_consecutive_var_decl mutation with weight: " << swapConsecutiveVarDeclWeight << "\n";
    }
    
    double reorderPredicatesWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "reorder_predicates_weight");
    if (reorderPredicatesWeight > 0.0) {
        mutations.push_back(std::make_unique<ReorderPredicatesMutation>(rng));
        llvm::outs() << "Enabled reorder_predicates mutation with weight: " << reorderPredicatesWeight << "\n";
    }
    
    // Check delete mutation weights and create specific delete mutations
    double deleteCommentWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_comment_weight");
    if (deleteCommentWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteCommentMutation>(rng));
        llvm::outs() << "Enabled delete_comment mutation with weight: " << deleteCommentWeight << "\n";
    }
    
    double deleteDeadIfWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_dead_if_stmt_weight");
    if (deleteDeadIfWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteMarkedMutation>(rng, "add_dead_if_stmt"));
        llvm::outs() << "Enabled delete_dead_if_stmt mutation with weight: " << deleteDeadIfWeight << "\n";
    }
    
    double deleteDeadForLoopWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_dead_for_loop_weight");
    if (deleteDeadForLoopWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteMarkedMutation>(rng, "add_dead_for_loop"));
        llvm::outs() << "Enabled delete_dead_for_loop mutation with weight: " << deleteDeadForLoopWeight << "\n";
    }
    
    double deleteDeadWhileLoopWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_dead_while_loop_weight");
    if (deleteDeadWhileLoopWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteMarkedMutation>(rng, "add_dead_while_loop"));
        llvm::outs() << "Enabled delete_dead_while_loop mutation with weight: " << deleteDeadWhileLoopWeight << "\n";
    }
    
    double deleteUnusedVarWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_unused_variable_weight");
    if (deleteUnusedVarWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteMarkedMutation>(rng, "add_unused_variable"));
        llvm::outs() << "Enabled delete_unused_variable mutation with weight: " << deleteUnusedVarWeight << "\n";
    }
    
    if (mutations.empty()) {
        llvm::outs() << "No mutations enabled in configuration\n";
        return;
    }
    
    // Randomly choose a mutation
    std::uniform_int_distribution<size_t> mutationDist(0, mutations.size() - 1);
    size_t mutationIndex = mutationDist(rng);
    
    llvm::outs() << "Applying kernel mutation to kernel: " << KernelDecl->getNameAsString() << "\n";
    llvm::outs() << "Mutation type: " << mutations[mutationIndex]->getName() << "\n";
    
    // Apply the selected mutation
    mutations[mutationIndex]->apply(KernelDecl, Rewrite);
}

void mutateKernel(clang::FunctionDecl* KernelDecl, clang::Rewriter& Rewrite, std::mt19937& externalRng) {
    if (!KernelDecl) return;
    
    // Get kaizen.cfg path from environment
    std::string kaizenConfigPath = ConfigReader::getKaizenConfigPath();
    if (kaizenConfigPath.empty()) {
        llvm::errs() << "Error: Cannot read mutation config without kaizen.cfg path\n";
        return;
    }
    
    // Get mutation config file path from kaizen.cfg
    std::string mutationConfigFile = ConfigReader::readConfigValue(kaizenConfigPath, "mutation", "mutation_options_config");
    if (mutationConfigFile.empty()) {
        llvm::errs() << "Error: mutation_options_config not found in kaizen.cfg\n";
        return;
    }
    
    // Create available mutations based on weights (using external RNG)
    std::vector<std::unique_ptr<BaseMutation>> mutations;
    
    // Check mutation weights and create mutations (reusing same logic with external RNG)
    double addCommentWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_comment_weight");
    if (addCommentWeight > 0.0) {
        mutations.push_back(std::make_unique<AddCommentMutation>(externalRng));
        llvm::outs() << "Enabled add_comment mutation with weight: " << addCommentWeight << "\n";
    }
    
    double editCommentWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "edit_comment_weight");
    if (editCommentWeight > 0.0) {
        mutations.push_back(std::make_unique<EditCommentMutation>(externalRng));
        llvm::outs() << "Enabled edit_comment mutation with weight: " << editCommentWeight << "\n";
    }
    
    double addDeadIfWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_dead_if_stmt_weight");
    if (addDeadIfWeight > 0.0) {
        mutations.push_back(std::make_unique<AddDeadIfStmtMutation>(externalRng));
        llvm::outs() << "Enabled add_dead_if_stmt mutation with weight: " << addDeadIfWeight << "\n";
    }
    
    double addDeadForLoopWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_dead_for_loop_weight");
    if (addDeadForLoopWeight > 0.0) {
        mutations.push_back(std::make_unique<AddDeadForLoopMutation>(externalRng));
        llvm::outs() << "Enabled add_dead_for_loop mutation with weight: " << addDeadForLoopWeight << "\n";
    }
    
    double addDeadWhileLoopWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_dead_while_loop_weight");
    if (addDeadWhileLoopWeight > 0.0) {
        mutations.push_back(std::make_unique<AddDeadWhileLoopMutation>(externalRng));
        llvm::outs() << "Enabled add_dead_while_loop mutation with weight: " << addDeadWhileLoopWeight << "\n";
    }
    
    double addUnusedVarWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "add_unused_variable_weight");
    if (addUnusedVarWeight > 0.0) {
        mutations.push_back(std::make_unique<AddUnusedVariableDeclarationMutation>(externalRng));
        llvm::outs() << "Enabled add_unused_variable mutation with weight: " << addUnusedVarWeight << "\n";
    }
    
    double renameVariableWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "rename_variable_weight");
    if (renameVariableWeight > 0.0) {
        mutations.push_back(std::make_unique<RenameVariableMutation>(externalRng));
        llvm::outs() << "Enabled rename_variable mutation with weight: " << renameVariableWeight << "\n";
    }
    
    double swapConsecutiveVarDeclWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "swap_consecutive_var_decl_weight");
    if (swapConsecutiveVarDeclWeight > 0.0) {
        mutations.push_back(std::make_unique<SwapConsecutiveVarDeclMutation>(externalRng));
        llvm::outs() << "Enabled swap_consecutive_var_decl mutation with weight: " << swapConsecutiveVarDeclWeight << "\n";
    }
    
    double reorderPredicatesWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "reorder_predicates_weight");
    if (reorderPredicatesWeight > 0.0) {
        mutations.push_back(std::make_unique<ReorderPredicatesMutation>(externalRng));
        llvm::outs() << "Enabled reorder_predicates mutation with weight: " << reorderPredicatesWeight << "\n";
    }
    
    // Check delete mutation weights and create specific delete mutations (using external RNG)
    double deleteCommentWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_comment_weight");
    if (deleteCommentWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteCommentMutation>(externalRng));
        llvm::outs() << "Enabled delete_comment mutation with weight: " << deleteCommentWeight << "\n";
    }
    
    double deleteDeadIfWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_dead_if_stmt_weight");
    if (deleteDeadIfWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteMarkedMutation>(externalRng, "add_dead_if_stmt"));
        llvm::outs() << "Enabled delete_dead_if_stmt mutation with weight: " << deleteDeadIfWeight << "\n";
    }
    
    double deleteDeadForLoopWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_dead_for_loop_weight");
    if (deleteDeadForLoopWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteMarkedMutation>(externalRng, "add_dead_for_loop"));
        llvm::outs() << "Enabled delete_dead_for_loop mutation with weight: " << deleteDeadForLoopWeight << "\n";
    }
    
    double deleteDeadWhileLoopWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_dead_while_loop_weight");
    if (deleteDeadWhileLoopWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteMarkedMutation>(externalRng, "add_dead_while_loop"));
        llvm::outs() << "Enabled delete_dead_while_loop mutation with weight: " << deleteDeadWhileLoopWeight << "\n";
    }
    
    double deleteUnusedVarWeight = ConfigReader::readConfigDouble(mutationConfigFile, "mutations", "delete_unused_variable_weight");
    if (deleteUnusedVarWeight > 0.0) {
        mutations.push_back(std::make_unique<DeleteMarkedMutation>(externalRng, "add_unused_variable"));
        llvm::outs() << "Enabled delete_unused_variable mutation with weight: " << deleteUnusedVarWeight << "\n";
    }
    
    if (mutations.empty()) {
        llvm::outs() << "No mutations enabled or available\n";
        return;
    }
    
    // Select a random mutation using external RNG
    std::uniform_int_distribution<size_t> dist(0, mutations.size() - 1);
    size_t mutationIndex = dist(externalRng);
    
    llvm::outs() << "Applying kernel mutation to kernel: " << KernelDecl->getNameAsString() << "\n";
    llvm::outs() << "Mutation type: " << mutations[mutationIndex]->getName() << "\n";
    
    // Apply the selected mutation
    mutations[mutationIndex]->apply(KernelDecl, Rewrite);
}

void setSharedRNGSeed(unsigned int seed) {
    rng.seed(seed);
}
