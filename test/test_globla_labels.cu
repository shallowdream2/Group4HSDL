#include "label/global_labels.cuh"
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void test(hop_constrained_case_info *hcci) {
  // push something to L_cuda
  hcci->L_cuda[0].push_back(hop_constrained_two_hop_label(1, 1, 1, 1));
  hcci->L_cuda[0].push_back(hop_constrained_two_hop_label(2, 2, 2, 2));
  // hcci->L_cuda[9]->push_back(hop_constrained_two_hop_label(3, 3, 3, 3));
  printf("push success\n");
  printf("size: %lu\n", hcci->L_cuda[0].size());
}

int main() {
  // hop_constrained_case_info *hcci;
  // cudaMallocManaged(&hcci, sizeof(hop_constrained_case_info));
  // hcci->init(1,1);
  // printf("init success\n");
  // test<<<1, 1>>>(hcci);
  // cudaDeviceSynchronize();

  // hcci->vector_gpu_to_cpu();
  // for (auto &l : hcci->L_cpu) {
  //   for (auto &ll : l) {
  //     printf("%lf %d %d %d\n", ll.distance, ll.hop, ll.hub_vertex,
  //            ll.parent_vertex);
  //   }
  // }
}