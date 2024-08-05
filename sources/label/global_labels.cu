#include "definition/mmpool_size.h"
#include "label/global_labels.cuh"
#include "memoryManagement/cuda_vector.cuh"

__host__ void hop_constrained_case_info::init(int vertex_nums,int mmpool_size_block) {
  L_size = vertex_nums;

  cudaMallocManaged(&mmpool_labels,
                    sizeof(mmpool<hub_type>)); // 创建统一内存的对象
  new (mmpool_labels) mmpool<hub_type>(mmpool_size_block); // 调用构造函数

  cudaMallocManaged(
      &L_cuda, vertex_nums * sizeof(cuda_vector<hub_type> )); // 分配n个cuda_vector指针
  for (int i = 0; i < vertex_nums; i++) {

    new (L_cuda+i) cuda_vector<hub_type>(mmpool_labels,vertex_nums/nodes_per_block+1); // 调用构造函数
  }
   cudaDeviceSynchronize();
}

__host__ void hop_constrained_case_info::destroy_L_cuda() {
  for (int i = 0; i < L_size; i++) {
    L_cuda[i].~cuda_vector<hub_type>(); // 调用析构函数
  }
  cudaFree(L_cuda);
}



// __host__ void hop_constrained_case_info::vector_gpu_to_cpu() {
//   //将gpu的vector转移到cpu，先从gpu拷贝结果回来，然后再转移到cpu
//   L_cpu.clear();
//   L_cpu.resize(L_size, vector<hub_type>());
//   for (int i = 0; i < L_size; i++) {
//     for (int j = 0; j < L_cuda[i]->size(); j++) {
//       hub_type tmp;
//       L_cuda[i]->copy_to_cpu(j, &tmp);
//       // printf("tmp: %lf %d %d %d\n", tmp.distance, tmp.hop, tmp.hub_vertex,
//       //        tmp.parent_vertex);
//       L_cpu[i].push_back(tmp);
//     }
//   }
// }