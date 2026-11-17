cmake_minimum_required(VERSION 3.10)
project(KernelExtractor)

set(CMAKE_C_COMPILER "clang-18")
set(CMAKE_CXX_COMPILER "clang++-18")

# ignore optional missing messages for find_package
macro(message)
endmacro()

# Find LLVM and Clang
find_package(LLVM 18.1 REQUIRED CONFIG)
find_package(Clang 18.1 REQUIRED CONFIG)
find_package(Threads REQUIRED)

# restore messages
macro(message)
  _message(${ARGN})
endmacro()

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

# Add the executable
add_executable(kernel-extractor main.cpp)

# Link to the Clang and LLVM libraries
target_link_libraries(kernel-extractor PRIVATE
    clangTooling
    clangASTMatchers
    clangFrontend
    clangSema
    clangDriver
    clangSerialization
    clangCodeGen
    clangParse
    clangAnalysis
    clangRewrite
    clangEdit
    clangLex
    clangAST
    clangBasic
    clangAPINotes
    clangSupport
    Threads::Threads
)
