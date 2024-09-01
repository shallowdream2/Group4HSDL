#ifndef CT_KMEANS_H
#define CT_KMEANS_H

#include "graph/graph_v_of_v.h"
#include "utilities/dijkstra.cuh"
#include "vgroup/CT/CT.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

static void
generate_Group_CT_cores(graph_v_of_v<disType> &instance_graph, int hop_cst,
                        std::vector<std::vector<int>> &groups) {

  int N = instance_graph.size();
  vector<int> labels(N, -1);
  std::unordered_map<int, int> group_size;
  vector<int> centers;
  // generate CT cores
  CT_case_info mm;
  dijkstra_table dt(instance_graph, false, hop_cst);
  dt.is_gpu = true;
  mm.d = 50;
  mm.use_P2H_pruning = 1;
  mm.two_hop_info.use_2M_prune = 1;
  mm.two_hop_info.use_canonical_repair = 1;
  mm.thread_num = 64;

  CT_cores(instance_graph, mm);
  printf("CT_cores finished\n");

  for (int i = 0; i < N; i++) {
    if (mm.isIntree[i] == 0) {
      centers.push_back(i);
      
    }
  }
  groups.resize(N);
  printf("centers size %d\n", centers.size());
  auto start = std::chrono::high_resolution_clock::now();
  dt.runDijkstra_gpu(centers);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("runDijkstra_gpu took %f seconds\n", duration.count());


    // 2.
    // 对于每个点，计算它到每个聚类中心的距离，将它划分到距离最近的聚类中心所在的类中
    for (int i = 0; i < N; ++i) {
      int nearest_center = -1;
      int min_distance = std::numeric_limits<int>::max();
      for (auto j : centers) {
        int distance = dt.query_distance(j, i);
        // printf("distance %d %d %d\n", i, j, distance);
        if (distance < min_distance && group_size[j] < MAX_GROUP_SIZE) {
          nearest_center = j;
          min_distance = distance;
        }
      }
      if (labels[i] != nearest_center) {
        group_size[nearest_center]++;
        if (labels[i] != -1) {
          group_size[labels[i]]--;
        }
        labels[i] = nearest_center;
      }
    }
  

  // 根据最终的labels数组构建groups输出
  for (int i = 0; i < N; ++i) {
    groups[labels[i]].push_back(i);
  }
}
#endif