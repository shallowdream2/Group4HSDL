#ifndef MMPOOL_CUH
#define MMPOOL_CUH
#include <cuda_runtime.h>
#include <iostream>


template <typename T> class mmpool {
private:
  struct node {
    T data;
    bool is_used;
    bool is_end;
    __host__ __device__ node() : is_used(false), is_end(false) {}
    __host__ __device__ node(T data)
        : data(data), is_used(true), is_end(false) {}
  };
  node *nodes_pool;      // 指向预分配的所有节点的指针
  int *block_used_nodes; // 每个块中已使用节点的数量
  int *block_next_index; // 下一个块的索引
  int num_blocks;        // 块的数量
  int nodes_per_block;   // 每个块的节点数量
  int now_empty_block_idx;

public:
  // 构造函数
  __host__ mmpool(int num_blocks = 100, int nodes_per_block = 100);
  __host__ __device__ size_t size();
  // 析构函数
  __host__ __device__ ~mmpool();

  __host__ __device__ bool is_full_block(int block_idx);

  __host__ __device__ bool is_valid_block(int block_idx);

  // 添加节点到内存池中指定块
  __device__ bool push_node(int block_idx, const T &node_data);

  // 查找内存池中指定行的指定下标的块
  __host__ __device__ node *get_node(int block_idx, int node_idx);

  // 查找空块
  __host__ __device__ int find_available_block();

  // 删除块（逻辑删除）
  __host__ __device__ bool remove_block(int block_idx);

  // 获取块的数量
  __host__ __device__ int get_num_blocks() { return num_blocks; }

  // 获取每个块的节点数量
  __host__ __device__ int get_nodes_per_block() { return nodes_per_block; }
};

template <typename T>
__host__ mmpool<T>::mmpool(int num_blocks, int nodes_per_block)
    : num_blocks(num_blocks), nodes_per_block(nodes_per_block) {
  cudaError_t error =
      cudaMalloc(&nodes_pool, sizeof(node) * nodes_per_block * num_blocks);
  if (error != cudaSuccess) {
    printf("CUDA malloc failed: %s\n", cudaGetErrorString(error));
    // 处理错误，如退出程序
  }

  cudaMallocManaged(&block_used_nodes, sizeof(int) * num_blocks);
  cudaMallocManaged(&block_next_index, sizeof(int) * num_blocks);

  // 初始化每个块
  for (int i = 0; i < num_blocks; ++i) {
    block_used_nodes[i] = 0;
    block_next_index[i] = (i == num_blocks - 1) ? -1 : i + 1;
  }
  now_empty_block_idx = 0;
}

template<typename T> __host__ __device__ size_t mmpool<T>::size() {
  size_t size = 0;
  for (int i = 0; i < num_blocks; ++i) {
    size += block_used_nodes[i];
  }
  return size;
}

// 析构函数
template <typename T> __host__ __device__ mmpool<T>::~mmpool() {
  cudaFree(nodes_pool);
  cudaFree(block_used_nodes);
  cudaFree(block_next_index);
}

template <typename T>
__host__ __device__ bool mmpool<T>::is_full_block(int block_idx) {
  return block_used_nodes[block_idx] == nodes_per_block;
}

template <typename T>
__host__ __device__ bool mmpool<T>::is_valid_block(int block_idx) {
  if (block_idx >= 0 && block_idx < num_blocks) {
    return true; // 无效块索引
  }
  return false;
}

// 添加节点到内存池
template <typename T>
__device__ bool mmpool<T>::push_node(int block_idx,
                                     const T &node_data) {
  if (!is_valid_block(block_idx)) {
    return false; // 无效块索引
  }
  if (is_full_block(block_idx)) {
    return false; // 块已满
  }
  node new_node(node_data);

  // 使用直接的设备端内存访问，而不是 cudaMemcpy

  int index = atomicAdd(&block_used_nodes[block_idx], 1);
  nodes_pool[block_idx * nodes_per_block + index] = new_node;

  // 直接增加 block_used_nodes[block_idx]
  block_used_nodes[block_idx]++;

  return true;
}

// 查找空块
template <typename T>
__host__ __device__ int mmpool<T>::find_available_block() {
  for (int i = now_empty_block_idx; i < num_blocks; ++i) {
    if (block_used_nodes[i] == 0) {
      return i;
    }
  }
  return -1; // 没有可用块
}

// 查找内存池中指定行的指定下标的块
template <typename T>
__host__ __device__ mmpool<T>::node *mmpool<T>::get_node(int block_idx,
                                                         int node_idx) {
  if (!is_valid_block(block_idx)) {
    return NULL; // 无效块索引
  }
  if (node_idx < 0 || node_idx >= block_used_nodes[block_idx]) {
    return NULL; // 无效节点索引
  }
  return &nodes_pool[block_idx * nodes_per_block + node_idx];
}

// 删除块（逻辑删除）
template <typename T>
__host__ __device__ bool mmpool<T>::remove_block(int block_idx) {
  if (!is_valid_block(block_idx)) {
    return false; // 无效块索引
  }

  block_used_nodes[block_idx] = 0; // 逻辑删除，标记为未使用
  return true;
}

#endif