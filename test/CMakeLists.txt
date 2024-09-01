cmake_minimum_required(VERSION 3.18)
project(Test LANGUAGES CXX CUDA)

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/includes)

# 定义可执行文件和源文件列表
set(CUDA_EXECUTABLES
    test_cuda_vector
    test_globla_labels
    test_query_cuda
    test_generation
    test_cuda_queue
    test_dijkstra
    test_gen_groups
    test_ct_kmeans
    test_louvain
    test_kmeans
)

set(CPP_EXECUTABLES
    test_readGraph
)

# 添加可执行文件
foreach(exec ${CUDA_EXECUTABLES})
    add_executable(${exec} ${exec}.cu)
    target_link_libraries(${exec} PRIVATE CUDA::cudart Group4HSDL)
    target_link_libraries(${exec} ${CUDA_RUNTIME_LIBRARY})
    set_target_properties(${exec} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

foreach(exec ${CPP_EXECUTABLES})
    add_executable(${exec} ${exec}.cpp)
    target_link_libraries(${exec} PRIVATE Group4HSDL)
endforeach()

# 如果有特定的编译选项，可以在这里设置
# target_compile_options(test_generation PRIVATE ${CUDA_NVCC_FLAGS})