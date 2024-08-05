#ifndef CUDA_QUERY_H
#define CUDA_QUERY_H
#include "definition/hub_def.h"
#include "memoryManagement/cuda_vector.cuh"
#include "memoryManagement/cuda_label.cuh"


__device__ float atomicMin(float* address, float val);


/**
 * @brief 此函数在host中调用，是一个并行查询最短距离的函数，还未优化
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


/**
 * @brief 此函数在generation中调用，效率低
 * 
 * @param hop_cst 
 * @param vec1 
 * @param vec2 
 * @param result_vec1 
 * @param result_vec2 
 * @param distance 
 * @return __device__ 
 */
__device__ void
query_mindis_with_hub_device(int hop_cst, cuda_vector<hub_type> *vec1,
                             cuda_vector<hub_type> *vec2, hub_type *result_vec1,
                             hub_type *result_vec2, disType *distance);

/**
 * @brief 此函数用于sort之后的clean，由于有序，可以更快的找到无效元素
 * 
 * @param hop_cst 
 * @param vec1 
 * @param vec2 
 * @param result_vec1 
 * @param result_vec2 
 * @param distance 
 * @return __device__ 
 */
__device__ __host__ void
query_mindis_final(int hop_cst, cuda_label<hub_type> *vec1,
                   cuda_label<hub_type> *vec2, int *result_vec1_index,
                   int *result_vec2_index, disType *distance);
#endif