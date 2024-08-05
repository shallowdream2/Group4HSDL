#ifndef CUDA_SORT_CUH
#define CUDA_SORT_CUH

#include "memoryManagement/cuda_vector.cuh"

void cuda_sort(cuda_vector<hub_type>* L_cuda, int vertex_num);

#endif