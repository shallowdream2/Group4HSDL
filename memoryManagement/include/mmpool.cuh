#ifndef MMPOOL_CUH
#define MMPOOL_CUH
#include <cuda_runtime.h>

template <typename T> class mmpool<T> {
private:
  node *nodes_pool;      // 指向预分配的所有节点的指针
  int *block_used_nodes; // 每个块中已使用节点的数量
  int *block_next_index; // 下一个块的索引
  int num_blocks;        // 块的数量
  int nodes_per_block;   // 每个块的节点数量
  int now_empty_block_idx;

public:
  struct node {
    T data;
    bool is_used;
    bool is_end;
  };
  // 构造函数
  mmpool(int num_blocks, int nodes_per_block = 100);

  // 析构函数
  ~mmpool();

  bool is_full_block(int block_idx);
  
  bool is_valid_block(int block_idx);

  // 添加节点到内存池中指定块
  bool push_node(int block_idx, node new_node);

  // 查找内存池中指定行的指定下标的块
  node *get_node(int block_idx, int node_idx);

  // 查找空块
  int find_available_block();

  // 删除块（逻辑删除）
  bool remove_block(int block_idx);
};

#endif