cmake_minimum_required(VERSION 3.18)
project(Group4HSDL LANGUAGES CXX CUDA)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 75)  # 可以修改为适合你的 GPU

# 添加子目录
add_subdirectory(memoryManagement)
add_subdirectory(test)

# 设置全局 CUDA 编译选项
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
list(APPEND CUDA_NVCC_FLAGS "--relocatable-device-code=true")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS}")
