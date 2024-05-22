#ifndef CUDA_QUERY_H
#define CUDA_QUERY_H
#include "definition/hub_def.h"
#include "memoryManagement/cuda_vector.cuh"

// 此函数用于在两个hop向量中找到最小的距离，返回对应的hop
__global__ void find_mindis_hops(int hop_cst, cuda_vector<hub_type> *vec1,
                                 cuda_vector<hub_type> *vec2,
                                 hub_type *result_vec1, hub_type *result_vec2);
__device__ double query_distance(int hop_cst, cuda_vector<hub_type> *vec1,
                                 cuda_vector<hub_type> *vec2);
#endif