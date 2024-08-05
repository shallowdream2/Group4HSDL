#include "utilities/dijkstra.h"
#include "definition/hub_def.h"
#include <unordered_map>
#include <unordered_set>

dijkstra_table::dijkstra_table(graph_v_of_v<int> &g, bool is_directed, int k,
                               vector<int> sources)
    : graph(g), is_directed(is_directed), k(k),
      source_set(unordered_set<int>(sources.begin(), sources.end())) {}

void dijkstra_table::runDijkstra(int s) {
  priority_queue<pair<disType, pair<int, int>>,
                 vector<pair<disType, pair<int, int>>>,
                 greater<pair<disType, pair<int, int>>>>
      pq;
  unordered_map<int, bool> visited;

  distance.clear();
  distance.resize(graph.size());
  //初始化
  for (int i = 0; i < graph.size(); i++) {
    distance[i] = numeric_limits<disType>::max();
  }

  pq.push({0, {s, 0}}); // <距离, <parent节点, 跳数>>
  distance[s] = 0;
  while (!pq.empty()) {
    disType dist = pq.top().first;
    int u = pq.top().second.first;
    int hops = pq.top().second.second;
    pq.pop();

    if (visited.find(u) != visited.end())
      continue;
    visited[u] = true;
    for (auto &[v, weight] : graph[u]) {
      if (hops + 1 > this->k)
        continue; // 跳数限制
      disType new_dist = dist + weight;
      if (distance[v] >= new_dist) {
        distance[v] = new_dist;
        pq.push({new_dist, {v, hops + 1}});
      }
    }
  }
}

disType dijkstra_table::query_distance(int s, int t) { return distance[t]; }
