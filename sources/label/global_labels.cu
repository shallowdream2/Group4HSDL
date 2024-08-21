#include "definition/mmpool_size.h"
#include "label/global_labels.cuh"
#include "memoryManagement/cuda_vector.cuh"

void hop_constrained_case_info::init() {
  int vertex_nums = L_size;
  int mmpool_size_block = vertex_nums *20;

  cudaMallocManaged(&mmpool_labels,
                    sizeof(mmpool<hub_type>)); // 创建统一内存的对象
  new (mmpool_labels) mmpool<hub_type>(mmpool_size_block); // 调用构造函数

  cudaMallocManaged(
      &L_cuda,
      vertex_nums * sizeof(cuda_vector<hub_type>)); // 分配n个cuda_vector指针
  for (int i = 0; i < vertex_nums; i++) {

    new (L_cuda + i) cuda_vector<hub_type>(
        mmpool_labels, vertex_nums / nodes_per_block + 1); // 调用构造函数
  }
  cudaDeviceSynchronize();
}

void hop_constrained_case_info::destroy_L_cuda() {
  for (int i = 0; i < L_size; i++) {
    L_cuda[i].~cuda_vector<hub_type>(); // 调用析构函数
  }
  cudaFree(L_cuda);
}

