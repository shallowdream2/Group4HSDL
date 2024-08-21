cmake_minimum_required(VERSION 3.18)
project(Group4HSDL LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA architecture (modify according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add subdirectory for tests
add_subdirectory(test)

# Set global CUDA compile options
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Uncomment and set your CUDA version if needed (usually auto-detected)
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")

# Set CUDA compile options
set(CUDA_NVCC_FLAGS "")
list(APPEND CUDA_NVCC_FLAGS
    "-std=c++17"
    "--relocatable-device-code=true"
    "-g"
    "-G"
    "--expt-relaxed-constexpr"
    "--extended-lambda"
)
string(JOIN " " CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C++ compile flags
set(CMAKE_CXX_FLAGS "-g -O3")

# Gather source files
file(GLOB_RECURSE SRC_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/sources/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sources/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/includes/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/includes/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/includes/.hpp"
)

# Create the library
add_library(${PROJECT_NAME} ${SRC_FILES})

# Include directories for the target
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes ${CMAKE_CURRENT_SOURCE_DIR}/sources)

# Apply CUDA compile options to the target
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# Debugging message
message("Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/includes ${CMAKE_CURRENT_SOURCE_DIR}/sources")
