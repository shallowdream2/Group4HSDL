#ifndef GLOBAL_LABELS_CUH
#define GLOBAL_LABELS_CUH

#include "da/hub_def.h"
#include "label/hop_constrained_two_hop_labels.cuh"
#include "memoryManagement/cuda_vector.cuh"
#include "memoryManagement/mmpool.cuh"
#include "unordered_map"
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

class hop_constrained_case_info {
public:
  /*labels*/
  mmpool<hub_type> *mmpool_labels;
  cuda_vector<hub_type> **L_cuda; // gpu res
  vector<vector<hub_type>> L_cpu; // cpu res
  size_t L_size;
  __host__ void init(int n);
  __host__ void destroy_L_cuda();
  inline size_t cuda_vector_size() { return L_size; }
  __host__ void vector_gpu_to_cpu();

};

#endif