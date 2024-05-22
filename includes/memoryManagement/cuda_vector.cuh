#ifndef CUDA_VECTOR_CUH
#define CUDA_VECTOR_CUH

#include "memoryManagement/mmpool.cuh"
#include <cassert>
#include <cuda_runtime.h>
// include log
#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels.cuh"
#include <iostream>
// log function which can be used in device code
inline __host__ __device__ void mylog(const char *message) {
  printf("%s\n", message);
}
//#define data_type hop_constrained_two_hop_label

template <typename T> class cuda_vector {
private:
  mmpool<T> *pool;
  size_t current_size;
  size_t capacity; // 当前vector的容量, 以块为单位
  int *block_idx_array;
  int blocks;
  int lock = false;

public:
  __host__ cuda_vector(mmpool<T> *pool); // 初始块大小可以根据需求调整

  __host__ __device__ ~cuda_vector();

  __device__ bool push_back(const T &value);
  __device__ __host__ T &operator[](size_t index);
  // const T& operator[](size_t index) const;
  __host__ __device__ void clear();
  __host__ __device__ size_t size() const { return current_size; }
  __device__ bool empty() const { return current_size == 0; }
  __host__ void copy_to_cpu(size_t index, T *cpu_ptr);
  __device__ __host__ T *get_device_ptr(size_t index);
  __device__ __host__ T at(size_t index) { return (*this)[index]; }
};

template <typename T>
__host__ cuda_vector<T>::cuda_vector(mmpool<T> *pool) : pool(pool) {
  this->blocks = 0;
  this->current_size = 0;
  this->capacity = 100;

  //申请空行
  int block_idx = pool->find_available_block();
  // printf("block_idx:%d\n\n", block_idx);
  if (block_idx == -1) {
    //没有空行，申请失败
    mylog("No available block in mmpool");
    return;
  }

  // copy to cuda
  cudaMallocManaged(&this->block_idx_array, sizeof(int) * capacity);
  // this->block_idx_array[this->blocks] = block_idx;
  // this->blocks += 1;
  cudaMemcpy(this->block_idx_array, &block_idx, sizeof(int),
             cudaMemcpyHostToDevice);
  this->blocks += 1;
};

template <typename T>
__device__ bool cuda_vector<T>::push_back(const T &value) {
  // 将此操作放在一个事务中，以确保线程安全
  // 事务开始
  while (atomicCAS(&this->lock, 0, 1) != 0)
    ;

  //找到当前vector的最后一个节点
  int last_block_idx = this->block_idx_array[this->blocks - 1];
  // printf("last_block_idx:%d\n", last_block_idx);
  //判断当前块是否已满
  if (pool->is_full_block(last_block_idx)) {
    //当前块已满，申请新块
    int block_idx = pool->find_available_block();
    if (block_idx == -1) {
      //没有空行，申请失败
      mylog("No available block in mmpool");
      atomicExch(&this->lock, 0);
      return false;
    }
    this->block_idx_array[this->blocks++] = block_idx;
    last_block_idx = block_idx;
  }
  //添加节点
  if (this->pool->push_node(last_block_idx, value)) {
    this->current_size++;
    atomicExch(&this->lock, 0);
    return true;
  }

  atomicExch(&this->lock, 0);
  return false;
};

template <typename T>
__device__ __host__ T &cuda_vector<T>::operator[](size_t index) {
  if (index >= this->current_size) {
    mylog("Index out of range");
    // error
    cudaError_t error = cudaGetLastError();

    assert(error == cudaErrorMemoryAllocation);
    // return nullptr;
  }
  //找到对应的块
  int block_idx = this->block_idx_array[index / pool->get_nodes_per_block()];
  //找到对应的节点
  int node_idx = index % pool->get_nodes_per_block();
  //返回节点
  // printf("block_idx:%d, node_idx:%d\n", block_idx, node_idx);
  return pool->get_node(block_idx, node_idx)->data;
};

template <typename T> __host__ __device__ void cuda_vector<T>::clear() {
  //释放所有块
  for (int i = 0; i < this->blocks; i++) {
    pool->remove_block(this->block_idx_array[i]);
  }
  //释放block_idx_array
  delete[] this->block_idx_array;
  this->blocks = 0;
  this->current_size = 0;
};

template <typename T> __host__ __device__ cuda_vector<T>::~cuda_vector() {
  clear();
};

template <typename T>
__device__ __host__ T *cuda_vector<T>::get_device_ptr(size_t index) {
  return &((*this)[index]);
}
template <typename T>
__host__ void cuda_vector<T>::copy_to_cpu(size_t index, T *cpu_ptr) {
  // 获取cuda_vector[i]的GPU地址
  // 将cuda_vector[i]的值复制到CPU的某个地址
  T *device_ptr = get_device_ptr(index);
  // 将 cuda_vector[i] 的值复制到 CPU 的某个地址
  cudaMemcpy(cpu_ptr, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
}

//显式声明模板类
template class cuda_vector<int>;
template class cuda_vector<float>;
template class cuda_vector<hub_type>;

#endif