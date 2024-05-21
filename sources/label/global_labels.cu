#include "label/global_labels.cuh"
#include "label/hop_constrained_two_hop_labels.h"
#include "memoryManagement/cuda_vector.cuh"

__device__ __host__ void hop_constrained_case_info::init(int n) {

  cudaMallocManaged(&mmpool_labels,
                    sizeof(mmpool<hop_type>));  // 创建统一内存的对象
  new (mmpool_labels) mmpool<hop_type>(n, 100); // 调用构造函数

  cudaMallocManaged(&L_cuda,
                    n * sizeof(cuda_vector<hop_type>)); // 分配n个cuda_vector
  for (int i = 0; i < n; i++) {
    new (&L_cuda[i]) cuda_vector<hop_type>(mmpool_labels); // 调用构造函数
  }
}