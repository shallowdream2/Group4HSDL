#ifndef HOP_CONSTRAINED_TWO_HOP_LABELS_H
#define HOP_CONSTRAINED_TWO_HOP_LABELS_H
#include "definition/hub_def.h"
#pragma once

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using namespace std;
/* label format */
struct hop_constrained_two_hop_label {
  int hub_vertex, parent_vertex, hop;
  weight_type distance;

  __device__ __host__ hop_constrained_two_hop_label(int hv, int pv, int h,
                                                    weight_type d)
      : hub_vertex(hv), parent_vertex(pv), hop(h), distance(d) {}
  __device__ __host__ hop_constrained_two_hop_label() {}
  // copy
  __device__ __host__
  hop_constrained_two_hop_label(const hop_constrained_two_hop_label &other) {
    hub_vertex = other.hub_vertex;
    parent_vertex = other.parent_vertex;
    hop = other.hop;
    distance = other.distance;
  }
  __device__ __host__ bool operator<(const hop_constrained_two_hop_label &y) const {
    //congregate the same hub_vertex into a continuous block, so we can use segment to find the label
    if(hub_vertex != y.hub_vertex){
      return hub_vertex < y.hub_vertex;
    }
    if (distance != y.distance) {
      return distance < y.distance; // < is the max-heap; > is the min heap
    } else {
      return hop < y.hop; // < is the max-heap; > is the min heap
    }
  }

  __device__ __host__ bool operator==(const hop_constrained_two_hop_label &rhs) const{
  return hub_vertex == rhs.hub_vertex &&
         parent_vertex == rhs.parent_vertex && hop == rhs.hop &&
         distance == rhs.distance;
}
};

// 定义哈希函数
struct DeviceHash {
  __device__ size_t operator()(int key) const {
    // 这是一个简单的 hash 函数，你可能需要根据你的需求来修改它
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key;
  }
};

namespace std {
template <> struct hash<hop_constrained_two_hop_label> {
  __device__ __host__ size_t
  operator()(const hop_constrained_two_hop_label &label) const {
#ifdef __CUDA_ARCH__
    DeviceHash hash;
#else
    std::hash<int> hash;
#endif
    return hash(label.hub_vertex) ^ (hash(label.parent_vertex) << 1) ^
           (hash(label.hop) << 2) ^ (hash(label.distance) << 3);
  }
};
} // namespace std



//  __device__ __host__ int hop_constrained_extract_distance(
//     vector<vector<hop_constrained_two_hop_label>> &L, int source, int
//     terminal, int hop_cst);

//  __device__ __host__ vector<pair<int, int>>
//  hop_constrained_extract_shortest_path(
//     vector<vector<hop_constrained_two_hop_label>> &L, int source, int
//     terminal, int hop_cst);
#endif // HOP_CONSTRAINED_TWO_HOP_LABELS_H
