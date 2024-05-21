#ifndef HOP_CONSTRAINED_TWO_HOP_LABELS_H
#define HOP_CONSTRAINED_TWO_HOP_LABELS_H
#pragma once

#include <iostream>
#include <limits>
#include <vector>
#include <fstream>
using namespace std;
/* label format */
class hop_constrained_two_hop_label {
public:
  int hub_vertex, parent_vertex, hop, distance;

  hop_constrained_two_hop_label(int hv, int pv, int h, int d)
      : hub_vertex(hv), parent_vertex(pv), hop(h), distance(d) {}
  hop_constrained_two_hop_label() {}
  // copy
  hop_constrained_two_hop_label(const hop_constrained_two_hop_label &other) {
    hub_vertex = other.hub_vertex;
    parent_vertex = other.parent_vertex;
    hop = other.hop;
    distance = other.distance;
  }
};

// 定义哈希函数
namespace std {
template <> struct hash<hop_constrained_two_hop_label> {
  size_t operator()(const hop_constrained_two_hop_label &label) const {
    return hash<int>()(label.hub_vertex) ^
           (hash<int>()(label.parent_vertex) << 1) ^
           (hash<int>()(label.hop) << 2) ^ (hash<int>()(label.distance) << 3);
  }
};
} // namespace std

// 定义等价性比较操作符
inline bool operator==(const hop_constrained_two_hop_label &lhs,
                const hop_constrained_two_hop_label &rhs) {
  return lhs.hub_vertex == rhs.hub_vertex &&
         lhs.parent_vertex == rhs.parent_vertex && lhs.hop == rhs.hop &&
         lhs.distance == rhs.distance;
}

inline bool operator<(hop_constrained_two_hop_label const &x,
               hop_constrained_two_hop_label const &y) {
  if (x.distance != y.distance) {
    return x.distance > y.distance; // < is the max-heap; > is the min heap
  } else {
    return x.hop > y.hop; // < is the max-heap; > is the min heap
  }
}



int hop_constrained_extract_distance(
    vector<vector<hop_constrained_two_hop_label>> &L, int source, int terminal,
    int hop_cst);

vector<pair<int, int>> hop_constrained_extract_shortest_path(
    vector<vector<hop_constrained_two_hop_label>> &L, int source, int terminal,
    int hop_cst); 
#endif // HOP_CONSTRAINED_TWO_HOP_LABELS_H
