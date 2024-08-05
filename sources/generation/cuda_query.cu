#include "definition/hub_def.h"
#include "generation/cuda_query.cuh"
#include <cfloat>
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

__device__ float atomicMin(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void query_mindis_with_hub(int hop_cst, cuda_vector<hub_type> *vec1,
                                      cuda_vector<hub_type> *vec2,
                                      hub_type *result_vec1,
                                      hub_type *result_vec2,
                                      disType *distance) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ float min_dis;
  __shared__ int min_hub1_vertex, min_hub1_hop, min_hub1_parent;
  __shared__ float min_hub1_dis;
  __shared__ int min_hub2_vertex, min_hub2_hop, min_hub2_parent;
  __shared__ float min_hub2_dis;

  if (threadIdx.x == 0) {
    min_dis = FLT_MAX;
  }
  __syncthreads();

  float local_min_dis = FLT_MAX;
  int local_hub1_vertex = 0, local_hub1_hop = 0, local_hub1_parent = 0;
  float local_hub1_dis = 0.0f;
  int local_hub2_vertex = 0, local_hub2_hop = 0, local_hub2_parent = 0;
  float local_hub2_dis = 0.0f;

  if (idx < vec1->size()) {
    for (int j = 0; j < vec2->size(); j++) {
      if (vec1->get(idx)->hub_vertex == vec2->get(j)->hub_vertex &&
          vec1->get(idx)->hop + vec2->get(j)->hop <= hop_cst) {
        float dis = vec1->get(idx)->distance + vec2->get(j)->distance;
        if (dis < local_min_dis) {
          local_min_dis = dis;
          local_hub1_vertex = vec1->get(idx)->hub_vertex;
          local_hub1_hop = vec1->get(idx)->hop;
          local_hub1_dis = vec1->get(idx)->distance;
          local_hub2_vertex = vec2->get(j)->hub_vertex;
          local_hub2_hop = vec2->get(j)->hop;
          local_hub2_dis = vec2->get(j)->distance;
          local_hub1_parent = vec1->get(idx)->parent_vertex;
          local_hub2_parent = vec2->get(j)->parent_vertex;
          // printf("j:%d,  idx:%d\n",j,idx);
        }
      }
    }
  }

  atomicMin(&min_dis, local_min_dis);
  __syncthreads();

  if (min_dis == local_min_dis) {
    atomicExch(&min_hub1_vertex, local_hub1_vertex);
    atomicExch(&min_hub1_hop, local_hub1_hop);
    atomicExch(&min_hub1_dis, local_hub1_dis);
    atomicExch(&min_hub2_vertex, local_hub2_vertex);
    atomicExch(&min_hub2_hop, local_hub2_hop);
    atomicExch(&min_hub2_dis, local_hub2_dis);
    atomicExch(&min_hub1_parent, local_hub1_parent);
    atomicExch(&min_hub2_parent, local_hub2_parent);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    *result_vec1 =
        hub_type(min_hub1_vertex, min_hub1_parent, min_hub1_hop, min_hub1_dis);
    *result_vec2 =
        hub_type(min_hub2_vertex, min_hub2_parent, min_hub2_hop, min_hub2_dis);
    *distance = min_dis;
  }
}

__device__ void
query_mindis_with_hub_device(int hop_cst, cuda_vector<hub_type> *vec1,
                             cuda_vector<hub_type> *vec2, hub_type *result_vec1,
                             hub_type *result_vec2, disType *distance) {
  disType min_dis = (int)(1e9);
  int min_hub1_vertex = 0, min_hub1_hop = 0, min_hub1_parent = 0;
  disType min_hub1_dis = min_dis;
  int min_hub2_vertex = 0, min_hub2_hop = 0, min_hub2_parent = 0;
  disType min_hub2_dis = min_dis;

  for (int i = 0; i < vec1->size(); i++) {
    for (int j = 0; j < vec2->size(); j++) {
      auto element1 = vec1->get(i);
      auto element2 = vec2->get(j);
      if (element1 && element2 &&
          element1->hub_vertex == element2->hub_vertex &&
          element1->hop + element2->hop <= hop_cst) {
        disType dis = vec1->get(i)->distance + vec2->get(j)->distance;
        if (dis < min_dis) {
          min_dis = dis;
          min_hub1_vertex = vec1->get(i)->hub_vertex;
          min_hub1_hop = vec1->get(i)->hop;
          min_hub1_dis = vec1->get(i)->distance;
          min_hub2_vertex = vec2->get(j)->hub_vertex;
          min_hub2_hop = vec2->get(j)->hop;
          min_hub2_dis = vec2->get(j)->distance;
          min_hub1_parent = vec1->get(i)->parent_vertex;
          min_hub2_parent = vec2->get(j)->parent_vertex;
        }
      }
    }
  }

  *result_vec1 =
      hub_type(min_hub1_vertex, min_hub1_parent, min_hub1_hop, min_hub1_dis);
  *result_vec2 =
      hub_type(min_hub2_vertex, min_hub2_parent, min_hub2_hop, min_hub2_dis);
  *distance = min_dis;
}

__device__ __host__ void
query_mindis_final(int hop_cst, cuda_label<hub_type> *vec1,
                   cuda_label<hub_type> *vec2, int *result_vec1_index,
                   int *result_vec2_index, disType *distance) {

  if (vec1->vertex == vec2->vertex) {
    *distance = 0;
    *result_vec1_index = -1;
    *result_vec2_index = -1;
    return;
  }
  int i, j; //使用双指针
  i = 0;
  j = 0;
  int size1 = vec1->length;
  int size2 = vec2->length;
  int min_dis = 1e9;

  while (i < size1 && j < size2) {
    //从hash表中找到对应的start和end

    int hub1 = vec1->data[i].hub_vertex;
    int hub2 = vec2->data[j].hub_vertex;
    if (hub1 == -1) { //已经被删除
      i++;
      continue;
    }
    if (hub2 == -1) {
      j++;
      continue;
    }

   // printf("hub1: %d, hub2: %d\n", hub1, hub2);
    int start1 = 0, end1 = i;
    vec1->query_hub(hub1, &start1, &end1);
    if (start1 == -1) {
      // error，没有找到对应的hub，直接跳过
      assert(false);
      break;
    }
    if (hub1 < hub2) {
      i = end1 + 1;
      continue;
    }

    int start2 = 0, end2 = j;
    vec2->query_hub(hub2, &start2, &end2);
    if (start2 == -1) {
      // error，没有找到对应的hub，直接跳过
      assert(false);
      break;
    }

    if (hub1 > hub2) {
      j = end2 + 1;
      continue;
    }
    //printf("start1: %d, end1: %d, start2: %d, end2: %d\n", start1, end1, start2,
    //       end2);

    int k = start1, l = start2;

    while (k <= end1 && l <= end2) {
      int hop1 = vec1->data[k].hop;
      int hop2 = vec2->data[l].hop;

      if (hop1 + hop2 <= hop_cst) {
        weight_type dis = vec1->data[k].distance + vec2->data[l].distance;

        if (dis < min_dis) {
          min_dis = dis;
          *result_vec1_index = k;
          *result_vec2_index = l;
        }
        l++; // Try to find a better match by increasing the hop in the second
             // vector
      } else {
        k++; // If the combined hop exceeds the limit, move the first pointer
      }
    }
    if (i == end1 + 1 && j == end2 + 1) {
      // 如果 i 和 j 都没有增加，说明可能陷入了死循环
      assert(false);
      break;
    }
    i = end1 + 1;
    j = end2 + 1;
  }
  *distance = min_dis;
}
