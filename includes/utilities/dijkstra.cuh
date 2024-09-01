#pragma once

#include <graph/graph_v_of_v.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
// 优先队列
#include "utilities/GPU_shortest_paths.cuh"
#include <definition/hub_def.h>
#include <queue>
#include <cuda_runtime.h>

using namespace std;

//#define disType double
#define entry pair<disType, int> // <distance, previous node>

class dijkstra_table {
public:
  graph_v_of_v<disType> graph;
  CSR_graph<disType> input_graph;
  bool is_gpu = false;

  bool is_directed;
  int k; // k-hop
  unordered_set<int> source_set;
  unordered_map<int, unordered_map<int, entry>> query_table_cpu;
  unordered_map<int, vector<disType>> query_table_gpu;
  dijkstra_table(graph_v_of_v<disType> &g, bool is_directed = false, int k = 5,
                 vector<int> sources = {0});

  void runDijkstra(int s);
  void runDijkstra_gpu(vector<int> &sources);
  disType query_distance(int s, int t);
  vector<int> query_path(int s, int t);
  void add_source(int s) { source_set.insert(s); }
  void add_source(vector<int> &s) {
    for (auto it : s) {
      add_source(it);
      runDijkstra(it);
    }
  }
};
