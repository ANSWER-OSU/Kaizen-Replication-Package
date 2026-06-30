#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <sys/wait.h>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <algorithm>

// Compile-time constant for log path
constexpr const char* LOG_PATH = "/scratch/projects/oscar/llm-testing/LLM-Code-Porting-Testing/logs/differential-testing";

// Function to get current date string and construct log file path
std::string get_log_file_path() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);

    std::ostringstream oss;
    oss << LOG_PATH << "/differential-testing-"
        << std::put_time(&tm, "%Y-%m-%d") << ".log";
    return oss.str();
}

// Function to append message to log file
void log_message(const std::string& message) {
    std::ofstream logfile(get_log_file_path(), std::ios::app);
    if (logfile.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        logfile << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] " << message;
        logfile.close();
    }
    // Also write to stderr for immediate visibility
    std::cerr << message;
}

// Forward declaration for OpenMP main function
namespace omp_program {
    int omp_main(int argc, char* argv[]);
}

// Helper function to properly escape shell arguments
std::string shell_escape(const std::string& arg) {
    std::string escaped;
    escaped.reserve(arg.size() + 2);
    escaped += '\'';
    for (char c : arg) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped += c;
        }
    }
    escaped += '\'';
    return escaped;
}

// StdoutRedirect class for capturing output with proper flushing
class StdoutRedirect {
    int old_stdout_fd;
    int old_stderr_fd;
    int file_fd;
    std::string filename;
    
public:
    StdoutRedirect(const std::string& fname) : filename(fname) {
        // Flush all streams first
        fflush(stdout);
        fflush(stderr);
        
        // Save old file descriptors
        old_stdout_fd = dup(fileno(stdout));
        old_stderr_fd = dup(fileno(stderr));
        
        // Open new file
        file_fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (file_fd < 0) {
            throw std::runtime_error("Failed to open redirect file: " + filename);
        }
        
        // Redirect both stdout and stderr to file
        dup2(file_fd, fileno(stdout));
        dup2(file_fd, fileno(stderr));
        close(file_fd);
    }
    
    ~StdoutRedirect() {
        // Flush everything before restoring
        fflush(stdout);
        fflush(stderr);
        
        // Force filesystem sync
        sync();
        
        // Restore original file descriptors
        dup2(old_stdout_fd, fileno(stdout));
        dup2(old_stderr_fd, fileno(stderr));
        close(old_stdout_fd);
        close(old_stderr_fd);
        
        // One more flush after restoring
        fflush(stdout);
        fflush(stderr);
    }
};

// Function to run CUDA program via fork/exec (safer than system())
int run_cuda_program(int argc, char* argv[]) {
    log_message("Running CUDA program with arguments:\n");
    for (int i = 0; i < argc; ++i) {
        log_message("  argv[" + std::to_string(i) + "] = \"" + std::string(argv[i]) + "\"\n");
    }

    // Timeout in seconds for the CUDA program
    const int timeout_seconds = 60;

    pid_t pid = fork();
    
    if (pid < 0) {
        log_message("ERROR: fork() failed\n");
        return -1;
    }
    
    if (pid == 0) {
        // Child process
        
        // Redirect stdout and stderr to cuda.log
        int fd = open("cuda.log", O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) {
            exit(127);
        }
    	
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);
        
    	 // Force unbuffered output
    	setvbuf(stdout, NULL, _IONBF, 0);
    	setvbuf(stderr, NULL, _IONBF, 0);	
        
	// Set environment variable to force CUDA to flush output
    	setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
	
	// Build argument list
        std::vector<char*> exec_argv;
        exec_argv.push_back(const_cast<char*>("./cuda_program"));
        for (int i = 1; i < argc; ++i) {
            exec_argv.push_back(argv[i]);
        }
        exec_argv.push_back(nullptr);
        
        // Execute the CUDA program
        execv("./cuda_program", exec_argv.data());
        
        // If execv returns, it failed
        exit(127);
    }
    
    // Parent process - wait with timeout
    int status = 0;
    bool timeout_occurred = false;
    
    for (int i = 0; i < timeout_seconds * 10; ++i) {
        pid_t result = waitpid(pid, &status, WNOHANG);
        
        if (result == pid) {
            // Process finished
            break;
        } else if (result < 0) {
            log_message("ERROR: waitpid() failed\n");
            return -1;
        }
        
        // Sleep for 100ms
        usleep(100000);
        
        if (i == (timeout_seconds * 10 - 1)) {
            timeout_occurred = true;
        }
    }
    
    if (timeout_occurred) {
        log_message("CUDA program timed out after " + std::to_string(timeout_seconds) + " seconds\n");
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        return -1;
    }
    
    // Ensure output is written to disk
    sync();
    usleep(100000); // Wait 100ms for filesystem
    
    if (WIFEXITED(status)) {
        int exit_code = WEXITSTATUS(status);
        log_message("CUDA program exited with code: " + std::to_string(exit_code) + "\n");
        return exit_code;
    } else if (WIFSIGNALED(status)) {
        int signal = WTERMSIG(status);
        log_message("CUDA program killed by signal: " + std::to_string(signal) + "\n");
        return -1;
    }
    
    return -1;
}

// Improved comparison function with normalization
bool compare_pass_fail_results(const std::string& file1, const std::string& file2) {
    // Ensure files exist and are readable
    std::ifstream f1(file1), f2(file2);
    if (!f1 || !f2) {
        log_message("WARNING: Could not open one or both log files for comparison\n");
        log_message("  file1 (" + file1 + "): " + (f1.good() ? "OK" : "FAIL") + "\n");
        log_message("  file2 (" + file2 + "): " + (f2.good() ? "OK" : "FAIL") + "\n");
        return false;
    }

    std::vector<std::string> pass_fail_lines1, pass_fail_lines2;
    std::string line;

    // Helper function to normalize PASS/FAIL lines
    auto normalize = [](const std::string& s) -> std::string {
        std::string result;
        // Remove extra whitespace and convert to uppercase
        bool prev_space = false;
        for (char c : s) {
            if (std::isspace(c)) {
                if (!prev_space && !result.empty()) {
                    result += ' ';
                    prev_space = true;
                }
            } else {
                result += std::toupper(c);
                prev_space = false;
            }
        }
        // Trim trailing space
        if (!result.empty() && result.back() == ' ') {
            result.pop_back();
        }
        return result;
    };

    // Extract and normalize PASS/FAIL lines from file1
    while (std::getline(f1, line)) {
        if (line.find("PASS") != std::string::npos || line.find("FAIL") != std::string::npos) {
            std::string normalized = normalize(line);
            if (!normalized.empty()) {
                pass_fail_lines1.push_back(normalized);
            }
        }
    }

    // Extract and normalize PASS/FAIL lines from file2
    while (std::getline(f2, line)) {
        if (line.find("PASS") != std::string::npos || line.find("FAIL") != std::string::npos) {
            std::string normalized = normalize(line);
            if (!normalized.empty()) {
                pass_fail_lines2.push_back(normalized);
            }
        }
    }

    // Log what we found
    log_message("Found " + std::to_string(pass_fail_lines1.size()) + 
                " PASS/FAIL lines in " + file1 + "\n");
    log_message("Found " + std::to_string(pass_fail_lines2.size()) + 
                " PASS/FAIL lines in " + file2 + "\n");

    // Compare the number of PASS/FAIL lines
    if (pass_fail_lines1.size() != pass_fail_lines2.size()) {
        log_message("MISMATCH: Different number of PASS/FAIL lines\n");
        return false;
    }

    // If no PASS/FAIL lines found in either file, consider it a match
    // (both programs likely crashed before producing output)
    if (pass_fail_lines1.empty() && pass_fail_lines2.empty()) {
        log_message("WARNING: No PASS/FAIL lines found in either output\n");
        return true;
    }

    // Compare each PASS/FAIL line
    for (size_t i = 0; i < pass_fail_lines1.size(); ++i) {
        if (pass_fail_lines1[i] != pass_fail_lines2[i]) {
            log_message("MISMATCH at line " + std::to_string(i) + ":\n");
            log_message("  OMP:  \"" + pass_fail_lines1[i] + "\"\n");
            log_message("  CUDA: \"" + pass_fail_lines2[i] + "\"\n");
            return false;
        }
    }

    return true;
}

// This is the new entry point for the fuzzer.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    // Convert the fuzzer's raw data into a string
    std::string line(reinterpret_cast<const char*>(Data), Size);

    // Log the input being tested
    log_message("=== Testing input: \"" + line + "\" ===\n");

    // Ignore anything after the first occurrence of "&&" (combined seed/min/max format)
    std::size_t cut = line.find("&&");
    if (cut != std::string::npos) {
        line = line.substr(0, cut);
    }

    // Log the cut string
    log_message("Using input for arguments: \"" + line + "\"\n");

    // Tokenize the input string to create fake arguments
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    // Build argv array
    std::vector<char*> fake_argv;
    fake_argv.push_back(const_cast<char*>("fuzzed_driver"));
    for (std::string& arg : tokens) {
        fake_argv.push_back(const_cast<char*>(arg.c_str()));
    }
    int fake_argc = static_cast<int>(fake_argv.size());

    // Run CUDA program with proper error handling
    int cuda_result = -1;
    bool cuda_success = false;
    try {
        cuda_result = run_cuda_program(fake_argc, fake_argv.data());
        cuda_success = (cuda_result == 0);
    } catch (const std::exception& e) {
        log_message("Exception in CUDA execution: " + std::string(e.what()) + "\n");
    } catch (...) {
        log_message("Unknown exception in CUDA execution\n");
    }

    // Run OpenMP program with proper error handling
    int omp_result = -1;
    bool omp_success = false;
    
    log_message("Running OMP program with arguments:\n");
    for (int i = 0; i < fake_argc; ++i) {
        log_message("  argv[" + std::to_string(i) + "] = \"" + std::string(fake_argv[i]) + "\"\n");
    }

    try {
        StdoutRedirect redirect("omp.log");
        omp_result = omp_program::omp_main(fake_argc, fake_argv.data());
        omp_success = (omp_result == 0);
    } catch (const std::exception& e) {
        log_message("Exception in OMP execution: " + std::string(e.what()) + "\n");
    } catch (...) {
        log_message("Unknown exception in OMP execution\n");
    }

    // CRITICAL: Wait a bit more for filesystem to flush
    sync();
    usleep(200000); // 200ms delay

    // Verify both log files exist and have content
    bool cuda_log_exists = (access("cuda.log", F_OK) == 0);
    bool omp_log_exists = (access("omp.log", F_OK) == 0);
    
    if (!cuda_log_exists) {
        log_message("WARNING: cuda.log does not exist!\n");
    }
    if (!omp_log_exists) {
        log_message("WARNING: omp.log does not exist!\n");
    }

    // Compare log files
    bool outputs_match = compare_pass_fail_results("omp.log", "cuda.log");
    
    if (!outputs_match) {
        log_message("=== TRANSLATION BUG DETECTED ===\n");
        log_message("PASS/FAIL results don't match!\n");

        // Log the input that caused the bug
        std::string input_str(reinterpret_cast<const char*>(Data), Size);
        log_message("Input that triggered bug: \"" + input_str + "\"\n");

        log_message("Exit codes - CUDA: " + std::to_string(cuda_result) + 
                   " (success: " + (cuda_success ? "true" : "false") + 
                   "), OMP: " + std::to_string(omp_result) + 
                   " (success: " + (omp_success ? "true" : "false") + ")\n");

        // Capture and log diff
        log_message("=== Diff output ===\n");
        std::string diff_cmd = "diff -u omp.log cuda.log >> " + get_log_file_path() + " 2>&1";
        std::system(diff_cmd.c_str());
        log_message("===================\n");

        // Classify the type of mismatch
        if (!cuda_success && !omp_success) {
            log_message("Both programs crashed but with different outputs\n");
        } else if (!cuda_success) {
            log_message("CUDA crashed but OMP succeeded\n");
        } else if (!omp_success) {
            log_message("OMP crashed but CUDA succeeded\n");
        } else {
            log_message("Both programs succeeded but produced different results\n");
        }

        // Trigger AFL crash
        __builtin_trap();
    }

    log_message("=== Test passed - outputs match ===\n\n");
    return 0;
}


#ifndef STANDALONE
int main() {
    std::vector<uint8_t> input;
    int c;
    
    // Read all input from stdin
    while ((c = getchar()) != EOF) {
        input.push_back(static_cast<uint8_t>(c));
    }
    
    if (input.empty()) {
        return 0;
    }
    
    // Call the fuzzer function
    return LLVMFuzzerTestOneInput(input.data(), input.size());
}
#endif

// Main function for standalone execution
#ifdef STANDALONE
int main(int argc, char* argv[]) {
    if (argc != 2) {
        log_message("Usage: " + std::string(argv[0]) + " <input_file>\n");
        return 1;
    }

    const char* input_path = argv[1];

    std::ifstream input_file(input_path, std::ios::binary);
    if (!input_file) {
        log_message("Error: Cannot open input file " + std::string(input_path) + "\n");
        return 1;
    }

    // Read the file content
    input_file.seekg(0, std::ios::end);
    size_t size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (size > 0) {
        input_file.read(reinterpret_cast<char*>(data.data()), size);
    }
    input_file.close();

    std::cout << "Running test with input from: " << input_path << std::endl;
	
    LLVMFuzzerTestOneInput(data.data(), data.size());

    std::cout << "Test finished. See omp.log and cuda.log for outputs." << std::endl;

    // After running, compare the results
    if (compare_pass_fail_results("omp.log", "cuda.log")) {
        std::cout << "✓ PASS/FAIL results match." << std::endl;
        return 0;
    } else {
        std::cout << "✗ Translation Bug Detected - PASS/FAIL results don't match!" << std::endl;
        std::cout << "Differences:" << std::endl;
        std::system("diff -u omp.log cuda.log");
        return 1;
    }
}
#endif // STANDALONE
