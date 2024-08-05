#ifndef CUDA_VECTOR_CUH
#define CUDA_VECTOR_CUH

#include "memoryManagement/mmpool.cuh"
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
// include log
#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
// log function which can be used in device code
inline __host__ __device__ void mylog(const char *message) {
  printf("%s\n", message);
}
//#define data_type hop_constrained_two_hop_label

template <typename T> class cuda_vector {
public:
  mmpool<T> *pool;
  size_t current_size;
  size_t capacity;      // 当前vector的容量, 以块为单位
  int *block_idx_array; // unified memory
  size_t blocks;
  int lock = false;
  T *first_elements; // unified memory
                     // ptr,在thrust中排序后，将数据拷贝到这里，以便在device中使用

  __host__ cuda_vector(mmpool<T> *pool,
                       size_t capacity = 100); // 初始块大小可以根据需求调整

  __host__ ~cuda_vector();
  //__host__ void resize(size_t new_size); //用于在pool中提前申请指定个数的块

  __device__ bool push_back(const T &value);
  __device__ __host__ T *get(size_t index);
  // const T& operator[](size_t index) const;
  __host__ void clear();
  __host__ __device__ size_t size() const { return current_size; }
  __device__ bool empty() const { return current_size == 0; }
  // __host__ void copy_to_cpu(size_t index, T *cpu_ptr);
  __host__ void sort_label(); //从host使用
  __host__ bool resize(size_t new_size);
  __host__ void prefetch() {
    for (int i = 0; i < blocks; i++) {
      pool->prefetch(block_idx_array[i]);
    }
  }
};

template <typename T>
__host__ cuda_vector<T>::cuda_vector(mmpool<T> *pool, size_t capacity)
    : pool(pool) {
  this->blocks = 0;
  this->current_size = 0;
  this->capacity = capacity;

  //申请空行
  int block_idx = pool->find_available_block();
  // printf("block_idx:%d\n\n", block_idx);
  if (block_idx == -1) {
    //没有空行，申请失败
    mylog("No available block in mmpool");
    assert(false);
    return;
  }

  // copy to cuda
  cudaMallocManaged(&this->block_idx_array, sizeof(int) * capacity);
  this->block_idx_array[this->blocks] = block_idx;
  this->blocks += 1;
  // printf("Constructor: blocks: %d\n", this->blocks);
  // printf("block_idx:%d\n", block_idx);
  //  cudaMemcpy(this->block_idx_array, &block_idx, sizeof(int),
  //             cudaMemcpyHostToDevice);
  //  this->blocks += 1;
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

template <typename T> __device__ __host__ T *cuda_vector<T>::get(size_t index) {
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
  return pool->get_node(block_idx, node_idx);
};

template <typename T> __host__ void cuda_vector<T>::clear() {
  //释放所有块
  for (int i = 0; i < this->blocks; i++) {
    pool->remove_block(this->block_idx_array[i]);
  }
  //释放block_idx_array
  // delete[] this->block_idx_array;
  this->blocks = 0;
  this->current_size = 0;
};

template <typename T> __host__ cuda_vector<T>::~cuda_vector() {
  clear();
  cudaFree(this->block_idx_array);
  // first_elements->clear();，
  // free(this->first_elements);，数据在cuda label中释放
};

template <typename T> __host__ bool cuda_vector<T>::resize(size_t new_size) {
  //在初始化后立即调用resize()，因此我们不需要检查是否有足够的块
  if (this->blocks == 0) {
    return false;
  }

  if (this->blocks == 1) {
    //刚刚初始化完，标为满
    pool->set_block_user_nodes(this->block_idx_array[0], nodes_per_block);
  }
  if (new_size <= this->blocks) {
    this->blocks = new_size;
    this->current_size = new_size * nodes_per_block;
    return true;
  }
  while (this->blocks < new_size) {
    int block_idx = pool->find_available_block();
    if (block_idx == -1) {
      //没有空行，申请失败
      mylog("No available block in mmpool");
      assert(false);
      return false;
    }
    this->block_idx_array[this->blocks++] = block_idx;
    // cudaMemcpy(this->block_idx_array + this->blocks, &block_idx, sizeof(int),
    //            cudaMemcpyHostToDevice);
    // this->blocks += 1;
    pool->set_block_user_nodes(block_idx, nodes_per_block);
  }
  this->current_size = new_size * nodes_per_block;
  return true;
}

template <typename T> __host__ void cuda_vector<T>::sort_label() {
  if (this->blocks == 0) {
    return;
  }

  // Calculate the total number of elements in the device_vector
  size_t total_elements = this->current_size;

  // 分配统一内存
  cudaError_t err =
      cudaMallocManaged(&first_elements, total_elements * sizeof(T));
  if (err != cudaSuccess) {
    printf("cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
    return;
  }

  // T *block_start_ptr;
  // cudaMallocManaged(&block_start_ptr, total_elements * sizeof(T));
  // block_start_ptr = new T(1, 1, 1, 1);

  // cudaStream_t stream;
  // cudaStreamCreate(&stream);

  size_t offset = 0;
  T *block_start_ptr;
  size_t i = 0;
  size_t continuous_block_size;
  for (; i < blocks;) {
    // 检测连续的索引块
    size_t start_idx = i;
    size_t end_idx = i;
    while (end_idx + 1 < blocks && this->block_idx_array[end_idx + 1] ==
                                       this->block_idx_array[end_idx] + 1) {
      end_idx++;
    }

    // 计算连续块的大小
    continuous_block_size = (end_idx - start_idx + 1) * nodes_per_block;
    if (end_idx == blocks - 1) {
      if (current_size % nodes_per_block == 0) {
        continuous_block_size = (end_idx - start_idx + 1) * nodes_per_block;
      } else {
        continuous_block_size = (end_idx - start_idx) * nodes_per_block +
                                current_size % nodes_per_block;
      }
    }

    // 获取连续块的起始指针
    block_start_ptr = pool->get_block_head(this->block_idx_array[start_idx]);

    // 合并拷贝操作到统一内存

    err = cudaMemcpy(first_elements + offset, block_start_ptr,
                     continuous_block_size * sizeof(T), cudaMemcpyDefault);

    // cudaStreamSynchronize(stream);
    offset += continuous_block_size;
    i = end_idx + 1;
  }

  // cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    printf("cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    assert(false);
  }
  // 使用thrust对统一内存中的数据进行排序
  thrust::sort(thrust::device_ptr<T>(first_elements),
               thrust::device_ptr<T>(first_elements + total_elements));
}

//显式声明模板类
// template class cuda_vector<int>;
// template class cuda_vector<float>;
template <typename hub_type> class cuda_vector;

#endif