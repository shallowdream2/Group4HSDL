cmake_minimum_required(VERSION 3.18)
project(Test LANGUAGES CXX CUDA)

add_executable(test_cuda_vector test_cuda_vector.cu)

target_link_libraries(test_cuda_vector PRIVATE memoryManagement)
include_directories(../memoryManagement/include)
set_target_properties(test_cuda_vector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
