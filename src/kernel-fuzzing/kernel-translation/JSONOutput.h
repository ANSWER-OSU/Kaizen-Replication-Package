#ifndef JSON_OUTPUT_H
#define JSON_OUTPUT_H

#include "../kernel-identification/KernelExtractor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>
#include <filesystem>

/**
 * Escapes special characters in a string for JSON output.
 * 
 * @param S The input string
 * @return The JSON-escaped string
 */
std::string escapeJSON(llvm::StringRef S);

/**
 * Removes all lines containing triple slash comments from content.
 * 
 * @param content The input content string
 * @return The content with triple slash comment lines removed
 */
std::string removeTripleSlashComments(const std::string &content);

/**
 * Prints the extraction results as a formatted JSON array.
 * 
 * @param results The extraction results to output
 * @param OS The output stream to write to
 */
void printResultsAsJSON(const std::vector<FileResult> &results, llvm::raw_ostream &OS);

/**
 * Creates the output directory if it doesn't exist.
 * 
 * @param outputPath The output file path
 * @return true if directory exists or was created successfully, false otherwise
 */
bool ensureOutputDirectory(const std::string &outputPath);

#endif // JSON_OUTPUT_H
