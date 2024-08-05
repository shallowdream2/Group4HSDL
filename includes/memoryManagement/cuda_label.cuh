#ifndef CUDA_LABEL_CUH
#define CUDA_LABEL_CUH
#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels.cuh"
#include "memoryManagement/cuda_hashTable.cuh"
//#include <thrust/device_vector.h>

template <typename T> class cuda_label {
public:
  T *data; // device ptr
  int vertex;
  int length;
  int current_length; //当前label中有效元素的个数
  cuda_hashTable<int, label_index> *index_table;

  cuda_label(T *other_data, int length,int vertex)
      : length(length), current_length(length),vertex(vertex) {

    if (length == 0) {
      data = NULL;
      index_table = NULL;
      throw "Empty label";
    }
    //将数据拷贝到device
    data = other_data;
    //初始化hashTable
    cudaMallocManaged(&index_table, sizeof(cuda_hashTable<int, label_index>));
    new (index_table) cuda_hashTable<int, label_index>(length);
    init_index_table();
  }

  __device__ __host__ void init_index_table() {
    if (length == 0) {
      return;
    }
    // data中，将连续的label放在一起，因此我们需要标识出连续的段
    int start = 0;
    int end = 0;
    int current_label = data[0].hub_vertex;
    for (int i = 1; i < length; i++) {
      if (data[i].hub_vertex != current_label) {
        end = i - 1;
        label_index index = {start, end};
        index_table->insert(current_label, index);
        start = i;
        current_label = data[i].hub_vertex;
      }
    }
    end = length - 1;
    label_index index = {start, end};
    index_table->insert(current_label, index);
  }

  __host__ __device__ void query_hub(int hub_vertex, int *start, int *end) {
    if (length == 0) {
      *start = -1;
      *end = -1;
      return;
    }
    label_index *index = index_table->find(hub_vertex);
    if (index == NULL) {
      *start = -1;
      *end = -1;
      return;
    }
    *start = index->start;
    *end = index->end;
  }

  void minimize() {
    if (length == 0) {
      return;
    }
    // label中无效元素已经被标记为1，现在我们需要将这些元素删除，并正确更新index_table
    T *new_data;
    cudaMallocManaged(&new_data, current_length * sizeof(T));
    int now = 0;
    for (int i = 0; i < length; i++) {
      if (data[i].hub_vertex != -1) {
        new_data[now++] = data[i];
      }
    }

    //释放原来的data
    cudaFree(data);
    data = new_data;
    //更新index_table
    cuda_hashTable<int, label_index> *new_index_table;
    cudaMallocManaged(&new_index_table,
                      sizeof(cuda_hashTable<int, label_index>));
    new (new_index_table) cuda_hashTable<int, label_index>(current_length);

    //释放原来的index_table
    index_table->~cuda_hashTable();
    cudaFree(index_table);

    index_table = new_index_table;
    length = current_length;
    init_index_table();
  }

  __device__ __host__ void print_L() {
    for (int i = 0; i < length; i++) {
      printf("{%d, %d, %d,%d},", data[i].hub_vertex,data[i].distance, data[i].hop,
              data[i].parent_vertex);
    }
    printf("\n");
  }
  void print_index_table() {
    printf("index table:\n");
    for (int i = 0; i < length; i++) {
      label_index index = *index_table->find(data[i].hub_vertex);
      printf("hub: %d, start: %d, end: %d\n", data[i].hub_vertex, index.start, index.end);
    }
  }
};

template <typename hub_type> class cuda_label;
#endif