cmake_minimum_required(VERSION 3.18)
project(memoryManagement CUDA)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cu" "include/*.cuh")

add_library(memoryManagement STATIC ${SOURCES})
set_target_properties(memoryManagement PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(memoryManagement PUBLIC include)
