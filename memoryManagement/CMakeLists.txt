cmake_minimum_required(VERSION 3.18)
project(memoryManagement LANGUAGES CXX CUDA)

include_directories(include)

file(GLOB_RECURSE SOURCES "include/*.cuh")

add_library(memoryManagement STATIC ${SOURCES})

set_target_properties(memoryManagement PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(memoryManagement PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

