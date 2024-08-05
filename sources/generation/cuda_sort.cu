#include "generation/cuda_sort.cuh"
#include <omp.h> // 包含OpenMP头文件

void cuda_sort(cuda_vector<hub_type> *L_cuda, int vertex_num)
{
  //#pragma omp parallel for // 指示OpenMP并行化这个循环
  for (int i = 0; i < vertex_num; i++) {
    L_cuda[i].sort_label();
  }
}