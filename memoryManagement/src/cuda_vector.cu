#include "cuda_vector.cuh"
#include <cuda_runtime.h>
#include <cassert>
// include log
#include <iostream>


template <typename T> __host__ __device__ cuda_vector<T>::cuda_vector(mmpool<T> &pool) {
  this->pool = pool;
  this->current_size = 0;
  this->capacity = 100;

  //申请空行
  int block_idx = pool.find_available_block();
  if (block_idx == -1) {
    //没有空行，申请失败
    log("No available block in mmpool");
    return;
  }
  this->block_idx_array = new int[capacity];
  this->block_idx_array[this->blocks++] = block_idx;
};

template <typename T> __device__ bool cuda_vector<T>::push_back(const T &value) {
  //找到当前vector的最后一个节点
  int last_block_idx = this->block_idx_array[this->blocks - 1];
  //判断当前块是否已满
  if (pool.is_full_block(last_block_idx)) {
    //当前块已满，申请新块
    int block_idx = pool.find_available_block();
    if (block_idx == -1) {
      //没有空行，申请失败
      log("No available block in mmpool");
      return false;
    }
    this->block_idx_array[this->blocks++] = block_idx;
    last_block_idx = block_idx;
  }
  //添加节点
  if (this->pool.push_node(last_block_idx, value)) {
    this->current_size++;
    return true;
  }
  return false;
};

template <typename T> __device__ T &cuda_vector<T>::operator[](size_t index) {
    if(index >= this->current_size){
      log("Index out of range");
      // error
      cudaError_t error = cudaGetLastError();
      
      assert(error == cudaErrorMemoryAllocation);
      return nullptr;
    }
    //找到对应的块
    int block_idx = this->block_idx_array + index / pool.nodes_per_block;
    //找到对应的节点
    int node_idx = index % pool.nodes_per_block;
    //返回节点
    return pool.get_node(block_idx, node_idx)->data;
};

template <typename T> __device__ void cuda_vector<T>::clear() {
    //释放所有块
    for (int i = 0; i < this->blocks; i++) {
        pool.remove_block(this->block_idx_array[i]);
    }
    //释放block_idx_array
    delete[] this->block_idx_array;
    this->blocks = 0;
    this->current_size = 0;
};

template <typename T> __host__ __device__ cuda_vector<T>::~cuda_vector() {
    clear();
};

