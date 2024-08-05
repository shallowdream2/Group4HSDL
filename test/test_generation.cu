#include "definition/hub_def.h"
#include "generation/gen_label.cuh"
#include <graph/graph_v_of_v.h>
#include <memoryManagement/cuda_label.cuh>
#define DATASET_PATH "/mnt/f/linux/rucgraph-HBPLL-GPU/data/euroroad2.txt"

int main() {

  // int deviceCount = 0;
  // cudaError_t err = cudaGetDeviceCount(&deviceCount);

  // if (err != cudaSuccess) {
  //   printf("CUDA error: %s\n", cudaGetErrorString(err));
  //   return -1;
  // }

  // printf("Detected %d CUDA Capable device(s)\n", deviceCount);

  // for (int i = 0; i < deviceCount; i++) {
  //   cudaDeviceProp deviceProp;
  //   cudaGetDeviceProperties(&deviceProp, i);
  //   printf("Device %d: %s\n", i, deviceProp.name);
  //   // 在这里可以打印出更多的设备属性
  // }
  // cudaStream_t stream;
  // cudaStreamCreate(&stream);
  // int *test1;
  // cudaMallocManaged(&test1, sizeof(int));
  // *test1 = 1;
  // int *test2;
  // cudaMallocManaged(&test2, sizeof(int));
  // size_t i = 0;
  // for (; i < 1;) {
  //   int start_idx = i;
  //   int end_idx = i;
  //   while (end_idx + 1 < 1 && 1 == 1) {
  //     end_idx++;
  //   }
  //   size_t continuous_block_size = (end_idx - start_idx + 1) * 1;
  //   if (end_idx == 1 - 1) {
  //     continuous_block_size = (end_idx - start_idx) * 1 + 1 % 1;
  //   }
  //   int *block_start_ptr = test1;
  //   cudaError_t err = cudaMemcpyAsync(test2, block_start_ptr,
  //                                     continuous_block_size * sizeof(int),
  //                                     cudaMemcpyDefault, stream);
  //   i = end_idx + 1;
  // }

  // cudaStreamSynchronize(stream);

  // printf("%d\n", *test2);

  graph_v_of_v<weight_type> instance_graph;
  instance_graph.txt_read(DATASET_PATH);
  printf("Graph read from %s\n", DATASET_PATH);
  printf("Number of vertices: %d\n", instance_graph.size());

  hop_constrained_case_info *info = NULL;
  //cuda_label<hub_type> *Labels = NULL;
  gen_labels_gpu(&instance_graph, info,  5);

  // printf
  //  //打印
  // printf("dis , hop , hub , parent \n");
  // for (int i = 0; i < vertex_num; i++) {
  //   info->L_cuda[i].sort_label();
  //   printf("vertex %d\n", i);
  //   for (int j = 0; j < info->L_cuda[i].size(); j++) {
  //     printf("{%d, %d, %d,%d},", (info->L_cuda[i]).get(j)->distance,
  //            (info->L_cuda[i]).get(j)->hop,
  //            (info->L_cuda[i]).get(j)->hub_vertex,
  //            (info->L_cuda[i]).get(j)->parent_vertex);
  //   }
  //   printf("\n");
  // }
}