cmake_minimum_required(VERSION 3.18)
project(Group4HSDL LANGUAGES CXX CUDA)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 75)  # 根据您的 GPU 修改

# 添加子目录
add_subdirectory(test)

# 设置全局 CUDA 编译选项
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 设置 CUDA 版本
set(CMAKE_CUDA_COMPILER_VERSION "11.8")
# 其他 CUDA 编译选项
set(CUDA_NVCC_FLAGS 
    "-std=c++17"
    "-Xcompiler=-fPIC"
    "--relocatable-device-code=true"
    "-g"
    "-G"
)
# 将 CUDA 编译选项应用到 CMake CUDA FLAGS
string(JOIN " " CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})

#设置C++版本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#设置C++ FALGS
set(CMAKE_CXX_FLAGS "-O2")

# 设置 源文件目录
set(SRC_DIRS 
    "sources"
    "sources/utilities"
    "sources/vgroup"
    "includes/graph"
    "includes/memoryManagement"
    "includes/text_mining"
    "includes/tool_functions"
    "includes/vgroup"
)
file(GLOB_RECURSE SRC_FILES 
    "sources/*.cpp"
    "sources/*.cu"
    # "includes/*.h"
    # "includes/*.cuh"
)

add_library(${PROJECT_NAME} ${SRC_FILES})

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes ${CMAKE_CURRENT_SOURCE_DIR}/sources)
message("Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/includes ${CMAKE_CURRENT_SOURCE_DIR}/sources")
