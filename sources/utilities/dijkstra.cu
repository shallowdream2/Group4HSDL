#include "utilities/dijkstra.cuh"
#include "definition/hub_def.h"
#include "utilities/GPU_csr.hpp"
#include <unordered_map>
#include <unordered_set>

dijkstra_table::dijkstra_table(graph_v_of_v<disType> &g, bool is_directed,
                               int k, vector<int> sources)
    : graph(g), is_directed(is_directed), k(k),input_graph(graph_v_of_v_to_CSR<disType>(g)),
      source_set(unordered_set<int>(sources.begin(), sources.end())) {
  
  for (int source : sources) {
    runDijkstra(source);
  }
}

void dijkstra_table::runDijkstra_gpu(vector<int> &sources) {
  for (int source : sources) {
    if (source_set.find(source) == source_set.end()) {
      source_set.insert(source);
      query_table_gpu[source].resize(graph.size());
      gpu_shortest_paths(input_graph, source, query_table_gpu[source]);
    }
  }
}

void dijkstra_table::runDijkstra(int s) {
  if (query_table_cpu.find(s) != query_table_cpu.end()) {
    return;
  }
  priority_queue<pair<disType, pair<int, int>>, // <dis, <当前节点, hop>>
                 vector<pair<disType, pair<int, int>>>,
                 greater<pair<disType, pair<int, int>>>>
      pq;
  unordered_map<int, entry> distance;

  // 初始化所有节点的距离为无穷大，前驱为-1
  for (int i = 0; i < graph.size(); i++) {
    distance[i] = {numeric_limits<disType>::max(), -1};
  }

  // 将源节点加入优先队列
  pq.push({0, {s, 0}});
  distance[s] = {0, s};

  while (!pq.empty()) {

    int u = pq.top().second.first;
    disType dist = pq.top().first;
    int hop = pq.top().second.second;
    pq.pop();
    if (dist > distance[u].first) {
      //说明distance已经被更新，可以直接跳出
      continue;
    }
    if (hop + 1 > this->k)
      continue; // 跳数限制
    // 遍历邻接节点
    for (auto &[v, weight] : graph[u]) {
      disType new_dist = dist + weight;
      if (new_dist < distance[v].first) {
        distance[v].first = new_dist;
        distance[v].second = u;
        pq.push({new_dist, {v, hop + 1}});
      }
    }
  }

  // 保存计算结果到查询表
  query_table_cpu[s] = distance;
}

vector<int> dijkstra_table::query_path(int s, int t) {
  vector<int> path;
  if (query_table_cpu.find(s) != query_table_cpu.end()) {
    int cur = t;
    while (cur != s) {
      path.push_back(cur);
      cur = query_table_cpu[s][cur].second;
    }
    path.push_back(s);
    reverse(path.begin(), path.end());

  } else if (query_table_cpu.find(t) != query_table_cpu.end()) {
    int cur = s;
    while (cur != t) {
      path.push_back(cur);
      cur = query_table_cpu[t][cur].second;
    }
    path.push_back(t);
  }
  return path;
}

disType dijkstra_table::query_distance(int s, int t) {
  if (is_gpu) {
    if (source_set.find(s) != source_set.end()) {
      return query_table_gpu[s][t];
    }
    if (source_set.find(t) != source_set.end()) {
      return query_table_gpu[t][s];
    }
    return numeric_limits<disType>::max();
  }
  disType min_distance = numeric_limits<disType>::max();

  if (query_table_cpu.find(s) != query_table_cpu.end()) {
    if (query_table_cpu[s].find(t) != query_table_cpu[s].end()) {
      min_distance = query_table_cpu[s][t].first;
    }
  } else if (query_table_cpu.find(t) != query_table_cpu.end()) {
    if (query_table_cpu[t].find(s) != query_table_cpu[t].end()) {
      min_distance = query_table_cpu[t][s].first;
    }
  }

  return min_distance;
}
