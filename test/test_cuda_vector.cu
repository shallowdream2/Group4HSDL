#include "memoryManagement/cuda_vector.cuh" // 确保包含正确的头文件路径
#include "memoryManagement/mmpool.cuh"
#include <cuda_runtime.h>
#include <iostream>

// 核函数，用于测试 cuda_vector 功能
__global__ void test_vector(cuda_vector<int> *vec) {
  //  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // 在向量中添加一些元素
  printf("vec->size() = %lu\n", vec->size());
  vec->push_back(2);
  if (vec->push_back(1)) {
    printf("push_back(1) success\n");
  }
  printf("vec->size() = %lu\n", vec->size());

  // 确保所有线程都已完成写操作
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0){
    printf("vec->size() = %lu\n", vec->size());
    for(int i = 0; i < vec->size(); i++){
      printf("vec[%d] = %d\n", i, *(vec->get(i)));
    }
  }
  // if (idx < vec->size()) {
  //     printf("vec[%d] = %d\n", idx, (*vec)[idx]);
  // }
}

__global__ void test_pool(mmpool<int> *pool) {
  //  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // 在向量中添加一些元素
  printf("pool->size() = %lu\n", pool->size());
  if (pool->push_node(2, 1)) {
    printf("push_node(2, 1) success\n");
  }

  // 确保所有线程都已完成写操作
  //__syncthreads();

  // if (idx < vec->size()) {
  //     printf("vec[%d] = %d\n", idx, (*vec)[idx]);
  // }
}

int main() {
  mmpool<int> *pool;
  cudaMallocManaged(&pool, sizeof(mmpool<int>)); // 在cuda上分配内存
  new (pool) mmpool<int>(10);               // 调用构造函数，在cuda分配的内存上构造对象

  test_pool<<<1, 1>>>(pool);
  cudaDeviceSynchronize(); // 等待核函数完成

  printf("%lu\n", pool->size());

  // // 分配和初始化 cuda_vector
  cuda_vector<int> *d_vector, *d_vector2;
  cudaMallocManaged(&d_vector, sizeof(cuda_vector<int>));
  cudaMallocManaged(&d_vector2, sizeof(cuda_vector<int>));
  new (d_vector2) cuda_vector<int>(pool); // 调用构造函数
  new (d_vector) cuda_vector<int>(pool);  // 调用构造函数

  // 启动核函数
  //test_vector<<<1, 1>>>(d_vector);
  test_vector<<<1, 2>>>(d_vector2);

  // 等待 GPU 完成
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    return 1;
  }

  // 销毁 vector 和释放资源

  cudaFree(d_vector);
  cudaFree(d_vector2);
  // 清理
  pool->~mmpool<int>(); // 调用析构函数
  cudaFree(pool);       // 释放内存

  return 0;
}
