#ifndef CUDA_QUERY_H
#define CUDA_QUERY_H
#include "definition/hub_def.h"
#include "memoryManagement/cuda_vector.cuh"
#include "utilities/dijkstra.h"


__device__ float atomicMin(float* address, float val);


/**
 * @brief 此函数用于在两个hop向量中找到最小的距离，返回对应的hop
 * 
 * @param hop_cst 
 * @param vec1 
 * @param vec2 
 * @param result_vec1 
 * @param result_vec2 
 * @return __global__ 
 */
__global__ void query_mindis_with_hub(int hop_cst, cuda_vector<hub_type> *vec1,
                                 cuda_vector<hub_type> *vec2,
                                 hub_type *result_vec1, hub_type *result_vec2,disType* distance);


#endif