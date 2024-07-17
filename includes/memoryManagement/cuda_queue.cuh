#ifndef CUDA_QUEUE_CUH
#define CUDA_QUEUE_CUH

#include "definition/mmpool_size.h"
#include "memoryManagement/cuda_vector.cuh"
#include "memoryManagement/mmpool.cuh"
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

template <typename T> class cuda_queue {
public:
  mmpool<T> *pool;
  cuda_vector<T> *data;
  int *front; // 队首索引
  int *rear;  // 队尾索引
  size_t max_size;


  cuda_queue(size_t size, mmpool<T> *pool) : pool(pool) {
    cudaMallocManaged(&front, sizeof(int));
    cudaMallocManaged(&rear, sizeof(int));
    cudaMallocManaged(&data, sizeof(cuda_vector<T>));
    new (data)cuda_vector<T>(pool, size);

    *front = 0;
    *rear = 0;
    // cudaDeviceSynchronize();
    // printf("data->blocks: %d\n", data->blocks);
    if (!data->resize(size)) {
      printf("resize failed\n");
      printf("data->blocks: %d\n", data->blocks);
      assert(false);
    }
    max_size = size * nodes_per_block; // size是block的个数
  }

  ~cuda_queue() {
    cudaFree(front);
    cudaFree(rear);
    data->~cuda_vector();
    cudaFree(data);
  }

  __device__ bool enqueue(const T item) {
    int next_rear = (*rear + 1) % max_size;
    if (next_rear == *front) { // 检查队列是否已满
      return false;
    }
    *(this->data->get(*rear)) = item;
    *rear = next_rear;
    return true;
  }

  __device__ bool dequeue(T *item) {
    if (*front == *rear) { // 检查队列是否为空
      return false;
    }
    *item = *(this->data->get(*front));
    *front = (*front + 1) % max_size;
    return true;
  }

  __device__ bool is_empty() const { return *front == *rear; }

  __device__ bool is_full() const { return (*rear + 1) % max_size == *front; }
};

//显式声明模板类
template class cuda_queue<hub_type>;

#endif