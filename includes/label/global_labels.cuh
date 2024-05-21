#ifndef GLOBAL_LABELS_CUH
#define GLOBAL_LABELS_CUH

#include "label/hop_constrained_two_hop_labels.h"
#include "memoryManagement/cuda_vector.cuh"
#include "memoryManagement/mmpool.cuh"
#include "unordered_map"
#include <cstddef>

#define hop_type hop_constrained_two_hop_label
class hop_constrained_case_info {
public:
  /*labels*/
  mmpool<hop_type>*mmpool_labels;
  cuda_vector<hop_type>*L_cuda; //待修改,以便于在cuda上使用
  size_t L_size;
  __device__ __host__ void init(int n);
  __device__ __host__ void destroy_L_cuda();
  
};

#endif