#include "definition/hub_def.h"
#include "definition/mmpool_size.h"
#include "generation/cuda_clean_label.cuh"
#include "generation/gen_label.cuh"
#include "graph/graph_v_of_v.h"
#include "memoryManagement/cuda_hashTable.cuh"
#include "utilities/dijkstra.h"
#include <cassert>
#include <cuda_runtime.h>

#define cuda_block_dim 256

void correctness_check(hop_constrained_case_info *info,
                       graph_v_of_v<disType> *G, int upper_bound) {
  dijkstra_table d_t(*G, false, upper_bound);
  //随机选择100个起点，100个终点，计算最短路径
  vector<int> start_points;
  vector<int> end_points;

  for (int i = 0; i < 100; i++) {
    start_points.push_back(rand() % G->size());
    end_points.push_back(rand() % G->size());
  }

  for (int i = 0; i < 100; i++) {
    int start = start_points[i];
    int end = end_points[i];
    // printf("start: %d, end: %d\n", start, end);
    //计算最短路径
    d_t.runDijkstra(start);
    disType dis1 = d_t.query_distance(start, end);
    disType dis2 = info->query_distance(start, end, upper_bound);
    if (abs(dis1 - dis2) > 1e-6) {
      info->print_final_label(start);
      info->print_final_label(end);
      vector<int> path = d_t.query_path(start, end);
      printf("path: ");
      for (int i = 0; i < path.size(); i++) {
        printf("%d ", path[i]);
      }
      printf("start: %d, end: %d\n", start, end);
      printf("dis1: %d, dis2: %d\n", dis1, dis2);
      assert(0);
    }
  }
}

__global__ void gen_labels_kernel_vertex_level(
    gpu_Graph *d_g, cuda_vector<hub_type> *L_gpu,
    cuda_hashTable<int, int> *L_hashes, cuda_queue<hub_type> *Qs,
    int upper_bound, int *group, cuda_hashTable<int, int> *reflect_group,
    int vertex_num) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= d_g->num_nodes || tid >= vertex_num)
    return;

  int g_id = group[tid];
  // printf("Thread %d starting\n", tid);
  cuda_queue<hub_type> *Q1 = Qs + tid;
  cuda_hashTable<int, int> *L_hash = L_hashes + tid;

  Q1->enqueue({tid, tid, 0, 0}); // group id
  L_gpu[tid].push_back({g_id, g_id, 0, 0});

  int degree_v_id = d_g->d_offsets[g_id + 1] - d_g->d_offsets[g_id];

  while (!Q1->is_empty()) {
    hub_type temp;
    Q1->dequeue(&temp);
    int u = temp.hub_vertex; //代表在graph中的id
    int g_u = group[u];

    if (L_hash->find(u) != NULL)
      continue;
    L_hash->insert(u, 1);
    weight_type du = temp.distance;
    int hu = temp.hop;

    int degree_u = d_g->d_offsets[g_u + 1] -
                   d_g->d_offsets[g_u]; //与图交互，需要使用图中的id

    if (degree_v_id > degree_u || (degree_v_id == degree_u && g_id >= g_u)) {
      hub_type t1, t2;
      weight_type q_dis;
      if (tid == u) {
        q_dis = 0.0;
      } else {
        query_mindis_with_hub_device(upper_bound, L_gpu + tid, L_gpu + u, &t1,
                                     &t2, &q_dis);
      }
      if (q_dis > du) {
        // 插入标签到L_gpu
        L_gpu[u].push_back(
            {g_id, temp.parent_vertex, hu, du}); //插入时，hub_vertex是图中
      }

      int h1 = hu + 1;
      if (h1 <= upper_bound) {
        for (int i = d_g->d_offsets[g_u]; i < d_g->d_offsets[g_u + 1]; ++i) {
          int v = d_g->d_edges[i].target;          //图中的id
          int *reflect_v = reflect_group->find(v); //转换为group中的id
          if (reflect_v == NULL) { //说明超过了group中的点，直接去除
            continue;
          }

          weight_type dv = du + d_g->d_edges[i].weight;
          hub_type t1, t2;
          weight_type q_dis_v;

          if (tid == v) {
            q_dis_v = 0.0;
          } else {
            query_mindis_with_hub_device(upper_bound, L_gpu + tid,
                                         L_gpu + *reflect_v, &t1, &t2,
                                         &q_dis_v);
          }
          if (q_dis_v > dv && L_hash->find(*reflect_v) == NULL) {
            Q1->enqueue({*reflect_v, g_u, h1, dv});
          }
        }
      }
    }
  }
}

void gen_labels_gpu(graph_v_of_v<weight_type> *G,
                    hop_constrained_case_info *info, int upper_bound) {

  cudaError_t err;
  // 1. Initiation
  // 包括初始化gpu上的图结构，初始化case_info，初始化queues和hashTable

  int vertex_num = info->L_size;

  // int edge_num = G->edge_number();

  //生成gpu上的graph
  // 将邻接表转换为一维数组表示的图结构

  gpu_Graph *d_g;
  cudaMallocManaged(&d_g, sizeof(gpu_Graph));
  new (d_g) gpu_Graph(G->ADJs);

  // printf("init case_info success\n");

  //准备queues
  cuda_queue<hub_type> *queues;
  cudaMallocManaged(&queues, vertex_num * sizeof(cuda_queue<hub_type>));

  int queue_size_blocks = 100;
  // printf("queue_size: %d\n", queue_size_blocks);

  for (int i = 0; i < vertex_num; i++) {

    new (&queues[i])
        cuda_queue<hub_type>(queue_size_blocks, info->mmpool_labels);
  }

  //准备hashTable
  // printf("max_degree: %d\n", d_g->max_degree);
  cuda_hashTable<int, int> *L_hash;
  cudaMallocManaged(&L_hash, vertex_num * sizeof(cuda_hashTable<int, int>));
  for (int i = 0; i < vertex_num; i++) {
    new (L_hash + i) cuda_hashTable<int, int>(d_g->max_degree * 10);
  }

  // 1. Initiation success
  // printf("initation success\n");

  // 2. Task，生成label

  int grid_dim = (vertex_num + cuda_block_dim - 1) / cuda_block_dim;
  // 时间测量
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  gen_labels_kernel_vertex_level<<<grid_dim, cuda_block_dim>>>(
      d_g, info->L_cuda, L_hash, queues, upper_bound, info->group,
      info->reflect_group, vertex_num);

  err = cudaGetLastError(); // 检查内核启动错误
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
              << std::endl;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    // 在这里添加更多调试信息
  }
  // 时间测量结束
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("generate label time: %f ms\n", milliseconds);

  //拷贝结果到cpu

  // 2. 生成label结束

  // 3. sort label,对标签进行清洗
  // 时间测量
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  //   cudaEventRecord(start, 0);

  //   cuda_sort(info->L_cuda, vertex_num);
  //   err = cudaGetLastError(); // 检查内核启动错误
  //   if (err != cudaSuccess) {
  //     std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
  //               << std::endl;
  //   }

  //   err = cudaDeviceSynchronize();
  //   if (err != cudaSuccess) {
  //     std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
  //               << std::endl;
  //   }

  //   // 时间测量结束
  //   cudaEventRecord(stop, 0);
  //   cudaEventSynchronize(stop);
  //   cudaEventElapsedTime(&milliseconds, start, stop);

  //   printf("sort label time: %f ms\n", milliseconds);

  //   // 3. sort label结束

  //   // // 4. clean label
  //   // clean之前，我们要先生成最后的cuda_label

  //   cuda_label<hub_type> *Labels;
  //   cudaMallocManaged(&Labels, vertex_num * sizeof(cuda_label<hub_type>));
  //   for (int i = 0; i < vertex_num; i++) {
  //     new (Labels + i) cuda_label<hub_type>(info->L_cuda[i].first_elements,
  //                                           info->L_cuda[i].size(), i);
  //   }

  // //  时间测量
  //   cudaEventCreate(&start);
  //   cudaEventCreate(&stop);
  //   cudaEventRecord(start, 0);

  //   cuda_clean_label<<<grid_dim, cuda_block_dim>>>(Labels, upper_bound,
  //   vertex_num);

  //   err = cudaDeviceSynchronize();
  //   if (err != cudaSuccess) {
  //     std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
  //               << std::endl;
  //   }

  //   // 时间测量结束
  //   cudaEventRecord(stop, 0);
  //   cudaEventSynchronize(stop);
  //   cudaEventElapsedTime(&milliseconds, start, stop);

  //   for (int i = 0; i < vertex_num; i++) {
  //     Labels[i].minimize();
  //   }

  //   printf("clean label time: %f ms\n", milliseconds);

  //   // 4. clean label结束

  //   //  打印

  //   correctness_check(Labels, G, upper_bound);
  //   printf("hub ,dis , hop ,  parent \n");
  //   for (int i = 0; i < vertex_num; i++) {
  //     printf("label %d\n", i);
  //     Labels[i].print_L();
  //     Labels[i].print_index_table();
  //     printf("\n");
  //   }

  // clean
}