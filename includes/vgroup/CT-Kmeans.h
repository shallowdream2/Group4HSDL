#ifndef CT_KMEANS_H
#define CT_KMEANS_H

#include "graph/graph_v_of_v.h"
#include <vector>
#include <unordered_set>
#include "vgroup/CT/CT.hpp"
#include "utilities/dijkstra.h"
using namespace std;


static void generate_Group_CT_cores(
    graph_v_of_v<int> &instance_graph,
    int hop_cst, 
    std::unordered_map<int, std::vector<int>> &groups) {

  int N = instance_graph.size();
  vector<int> labels(N, -1);
  unordered_set<int> centers;
  // generate CT cores
  CT_case_info mm;
  dijkstra_table dt(instance_graph,false, hop_cst);
  mm.d = 10;
  mm.use_P2H_pruning = 1;
  mm.two_hop_info.use_2M_prune = 1;
  mm.two_hop_info.use_canonical_repair = 1;
  mm.thread_num = 10;

  CT_cores(instance_graph, mm);
  printf("CT_cores finished\n");



  for (int i = 0; i < N; i++) {
    if (mm.isIntree[i] == 0) {
      centers.insert(i);
      dt.add_source(i);
      dt.runDijkstra(i);
    }
  }

  bool changed=true;
  while (changed) {
    changed = false;
    // 2.
    // 对于每个点，计算它到每个聚类中心的距离，将它划分到距离最近的聚类中心所在的类中
    for (int i = 0; i < N; ++i) {
      int nearest_center = -1;
      int min_distance = std::numeric_limits<int>::max();
      for (auto j : centers) {
        int distance = dt.query_distance(i, j);
        //printf("distance %d %d %d\n", i, j, distance);
        if (distance < min_distance) {
          nearest_center = j;
          min_distance = distance;
        }
      }
      if (labels[i] != nearest_center) {
        labels[i] = nearest_center;
        changed = true;
      }
    }
  }

  // 根据最终的labels数组构建groups输出
  for (int i = 0; i < N; ++i) {
    groups[labels[i]].push_back(i);
  }
}
#endif