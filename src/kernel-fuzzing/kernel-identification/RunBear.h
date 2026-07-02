#ifndef RUN_BEAR_H
#define RUN_BEAR_H

#include <string>
#include <vector>

/*
* Runs 'make' with verbose flags in the given directory to generate compile_commands.json.
* Returns true on success, else false. Exits in same directory as start.
*/
bool generateCompileCommands(const std::string &build_dir);

/*
* Parses make verbose output to extract compilation commands.
* Returns a vector of compilation command strings.
*/
std::vector<std::string> parseMakeOutput(const std::string &make_output, const std::string &build_dir);

/*
* Generates compile_commands.json from a vector of compilation commands.
* Returns true on success, else false.
*/
bool writeCompileCommandsJson(const std::vector<std::string> &commands, const std::string &build_dir);

#endif
