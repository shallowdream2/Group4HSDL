cmake_minimum_required(VERSION 3.18)
project(Test LANGUAGES CXX CUDA)

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(test_cuda_vector test_cuda_vector.cu)
add_executable(test_readGraph test_readGraph.cpp)
add_executable(test_dijkstra test_dijkstra.cpp)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/includes)

# 如果 memoryManagement 中有源代码文件，这里需要添加它们
file(GLOB MEMORY_MANAGEMENT_SOURCES "${CMAKE_SOURCE_DIR}/includes/memoryManagement/*.cuh")
target_sources(test_cuda_vector PRIVATE ${MEMORY_MANAGEMENT_SOURCES})

# 链接库
target_link_libraries(test_cuda_vector PRIVATE CUDA::cudart)
target_link_libraries(test_dijkstra PRIVATE Group4HSDL)

# 设置CUDA可分离编译
set_target_properties(test_cuda_vector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
