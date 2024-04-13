cmake_minimum_required(VERSION 3.18)
project(Test CUDA)

add_executable(test_cuda_vector test_cuda_vector.cu)

target_link_libraries(test_cuda_vector PRIVATE memoryManagement)
target_link_libraries(test_cuda_vector PRIVATE cudart)

set_target_properties(memoryManagement PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(test_cuda_vector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(test_cuda_vector PRIVATE ../memoryManagement/include)
