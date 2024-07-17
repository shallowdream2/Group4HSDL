#ifndef GEN_LABEL_CUH
#define GEN_LABEL_CUH
#include "generation/cuda_query.cuh"
#include "graph/gpu_graph.cuh"
#include "graph/graph_v_of_v.h"
#include "label/global_labels.cuh"
#include "label/hop_constrained_two_hop_labels.cuh"
#include "memoryManagement/cuda_queue.cuh"
#include "memoryManagement/cuda_vector.cuh"
#include "vgroup/kmeans.h"
#include "definition/hub_def.h"
#include "memoryManagement/cuda_hashTable.cuh"

void gen_labels_gpu(graph_v_of_v<weight_type> *G, hop_constrained_case_info *info,
                    int upper_bound);
__global__ void gen_labels_kernel_vertex_level(gpu_Graph *d_g,
                                               cuda_vector<hub_type> **L_gpu,
                                               cuda_hashTable<int,int>* L_hash,
                                               cuda_queue<hub_type> *Qs,
                                               int upper_bound);
#endif