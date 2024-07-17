#include "definition/hub_def.h"
#include "memoryManagement/cuda_queue.cuh"

// 核函数，用于测试 cuda_queue 功能
__global__ void test(cuda_queue<hub_type> *q) {
  printf("test\n");
  hub_type item = {1, 2, 3, 4};
  q->enqueue(item);
  printf("enqueue success\n");
  hub_type item2;
  q->dequeue(&item2);  // 确保传递的是指针
  printf("item2: %d %d %d %d\n", item2.distance, item2.hop, item2.hub_vertex, item2.parent_vertex);
}

int main() {
  mmpool<hub_type> *pool;
  cudaError_t err;

  err = cudaMallocManaged(&pool, sizeof(mmpool<hub_type>)); // 在cuda上分配内存
  if (err != cudaSuccess) {
    printf("cudaMallocManaged failed for pool: %s\n", cudaGetErrorString(err));
    return -1;
  }

  new (pool) mmpool<hub_type>(10); // 调用构造函数，在cuda分配的内存上构造对象

  cuda_queue<hub_type> *q;
  err = cudaMallocManaged(&q, sizeof(cuda_queue<hub_type>)); // 在cuda上分配内存
  if (err != cudaSuccess) {
    printf("cudaMallocManaged failed for queue: %s\n", cudaGetErrorString(err));
    cudaFree(pool);
    return -1;
  }

  new (q) cuda_queue<hub_type>(10, pool); // 调用构造函数，在cuda分配的内存上构造对象
  cudaDeviceSynchronize(); // 确保设备和主机同步
  printf("After resize: data->blocks: %d\n", q->data->blocks);


  
  test<<<1, 1>>>(q);
  err = cudaDeviceSynchronize(); // 等待核函数完成

  if (err != cudaSuccess) {
    printf("cudaDeviceSynchronize returned error code %d after launching the kernel!\n", err);
    return -1;
  }

  // 销毁对象并释放内存
  q->~cuda_queue<hub_type>();
  pool->~mmpool<hub_type>();
  cudaFree(q);
  cudaFree(pool);


  return 0;
}
