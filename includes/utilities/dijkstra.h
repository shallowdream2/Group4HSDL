#pragma once

#include <graph/graph_v_of_v.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
// 优先队列
#include <queue>
#include <definition/hub_def.h>
using namespace std;

//#define disType double
#define entry pair<disType, int> // <distance, previous node>

class dijkstra_table {
public:
  graph_v_of_v<int> graph;
  bool is_directed;
  int k; // k-hop
  unordered_set<int> source_set;
  unordered_map<int, unordered_map<int, entry>> query_table;
  dijkstra_table(graph_v_of_v<int> &g, bool is_directed = false, int k = 5,
                 vector<int> sources = {0});

  void runDijkstra(int s);
  disType query_distance(int s, int t);
  vector<int> query_path(int s, int t);
  void add_source(int s) {
    source_set.insert(s);
  }
  void add_source(vector<int> &s) {
    for (auto it : s) {
      add_source(it);
    }
  }
};
