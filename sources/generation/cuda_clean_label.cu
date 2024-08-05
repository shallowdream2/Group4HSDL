#include "definition/hub_def.h"
#include "generation/cuda_clean_label.cuh"
#include "generation/cuda_query.cuh"

__global__ void cuda_clean_label(cuda_label<hub_type> *L_cuda, int upper_bound,
                                 int vertex_num) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vertex_num ) {
    return;
  }
  // L_cuda[idx].init_index_table();

  //printf("thread: %d starting\n", idx);
  // printf("vertex: %d\n", );
  // L_cuda[idx].print_L();
  for (int i = 0; i < L_cuda[idx].length; i++) {
    int u = L_cuda[idx].data[i].hub_vertex;
    //printf("thread: %d loop: %d\n, stop: 1", idx, i);
    int res1, res2;
    disType distance = 1e9;
    //printf("u: %d\n", u);
    query_mindis_final(upper_bound, L_cuda + u, L_cuda + idx, &res1, &res2,
                       &distance);
    //printf("thread: %d loop: %d\n, stop: 2", idx, i);

    //检查res1和res2是否是当前的label
    if (distance == L_cuda[idx].data[i].distance) {
      //printf("thread: %d finished loop: %d\n", idx, i);
      continue;
    } else {
      L_cuda[idx].data[i].hub_vertex = -1; //将label中的无效元素标记为-1
    }
    //printf("thread: %d finished loop: %d\n", idx, i);
  }
  //printf("thread: %d finished\n", idx);
}