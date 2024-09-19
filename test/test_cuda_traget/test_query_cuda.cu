#include "definition/hub_def.h"
#include "generation/cuda_query.cuh"
#include "iostream"
__global__ void init_data(cuda_vector<hub_type> &vec1,
                          cuda_vector<hub_type> &vec2) {
  // 向量1初始化
  vec1.push_back(hub_type(1, 0, 1, 2.0f));
  vec1.push_back(hub_type(2, 0, 2, 2.0f));
  vec1.push_back(hub_type(3, 0, 3, 3.0f));

  // 向量2初始化
  vec2.push_back(hub_type(1, 0, 1, 15.5f));
  vec2.push_back(hub_type(2, 0, 2, 2.5f));
  vec2.push_back(hub_type(3, 0, 1, 1.2f));
  // printf("vec1 %d %d %d %f\n", vec1[0].hub_vertex, vec1[0].parent_vertex,
  //        vec1[0].hop, vec1[0].distance);
  // printf("vec2 %d %d %d %f\n", vec2[0].hub_vertex, vec2[0].parent_vertex,
  //         vec2[0].hop, vec2[0].distance);
}

void test_find_mindis_hops() {

  mmpool<hub_type> *pool;
  cudaMallocManaged(&pool, sizeof(mmpool<hub_type>)); // 创建统一内存的对象
  new (pool) mmpool<hub_type>(10);
  // 初始化数据

  // 分配设备内存并复制数据
  cuda_vector<hub_type> *d_vec1, *d_vec2;
  cudaMallocManaged(&d_vec1, sizeof(cuda_vector<hub_type>));
  cudaMallocManaged(&d_vec2, sizeof(cuda_vector<hub_type>));
  new (d_vec1) cuda_vector<hub_type>(pool);
  new (d_vec2) cuda_vector<hub_type>(pool);
  init_data<<<1, 1>>>(*d_vec1, *d_vec2);

  // 分配结果向量的设备内存
  hub_type *d_result_vec1, *d_result_vec2;
  cudaMalloc(&d_result_vec1, sizeof(hub_type));
  cudaMalloc(&d_result_vec2, sizeof(hub_type));

  // 设置hop约束并启动内核
  int hop_cst = 3;
  //计时
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  query_mindis_with_hub<<<1, 256>>>(hop_cst, d_vec1, d_vec2, d_result_vec1,
                               d_result_vec2, 0);
  cudaDeviceSynchronize();
  cudaEventCreate(&stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Elapsed time: " << elapsedTime << "ms" << std::endl;

  // 从设备内存复制结果到主机内存
  hub_type h_result_vec1, h_result_vec2;
  cudaMemcpy(&h_result_vec1, d_result_vec1, sizeof(hub_type),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_result_vec2, d_result_vec2, sizeof(hub_type),
             cudaMemcpyDeviceToHost);

  // 打印结果
  std::cout << "Result Vector 1: " << h_result_vec1.hub_vertex << ", "
            << h_result_vec1.parent_vertex << ", " << h_result_vec1.hop << ", "
            << h_result_vec1.distance << std::endl;
  std::cout << "Result Vector 2: " << h_result_vec2.hub_vertex << ", "
            << h_result_vec2.parent_vertex << ", " << h_result_vec2.hop << ", "
            << h_result_vec2.distance << std::endl;

  // 释放设备内存
  cudaFree(d_vec1);
  cudaFree(d_vec2);
  cudaFree(d_result_vec1);
  cudaFree(d_result_vec2);
}

int main() {
  test_find_mindis_hops();
  return 0;
}
