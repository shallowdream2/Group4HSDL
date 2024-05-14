#include "utilities/dijkstra.h"

dijkstra_table::dijkstra_table(graph_v_of_v<int> &g, bool is_weighted, int k,
                 vector<int> sources)
      : graph(g), is_weighted(is_weighted), k(k) {
    for (auto &s : sources) {
      runDijkstra(s);
    }
  }



void dijkstra_table::runDijkstra(int s) {
  priority_queue<pair<disType, pair<int, int>>,
                 vector<pair<disType, pair<int, int>>>,
                 greater<pair<disType, pair<int, int>>>>
      pq;
  vector<bool> visited(graph.size(), false);
  pq.push({0, {s, 0}}); // <距离, <节点, 跳数>>

  while (!pq.empty()) {
    disType dist = pq.top().first;
    int u = pq.top().second.first;
    int hops = pq.top().second.second;
    pq.pop();

    if (visited[u])
      continue;
    visited[u] = true;

    for (auto &[v, weight] : graph[u]) {
      if (hops + 1 > this->k)
        continue; // 跳数限制
      disType new_dist = dist + weight;
      if (get<0>(this->query(s, v)) > new_dist) {
        query_table[s][v] = {new_dist, u, hops + 1};
        pq.push({new_dist, {v, hops + 1}});
      }
    }
  }
  source_set.insert(s); // 更新源点集合
}

entry dijkstra_table::query(int s, int t) {
  if (query_table[s].find(t) != query_table[s].end()) {
    return query_table[s][t];
  }

  if ((!this->is_weighted) && query_table[t].find(s) != query_table[t].end()) {
    return query_table[t][s];
  }

  return {numeric_limits<disType>::max(), -1, -1};
}

disType dijkstra_table::query_distance(int s, int t) {
  return get<0>(query(s, t));
}