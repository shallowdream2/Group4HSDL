#include "utilities/dijkstra.h"
#include "definition/hub_def.h"
#include <unordered_map>
#include <unordered_set>

dijkstra_table::dijkstra_table(graph_v_of_v<int> &g, bool is_directed, int k,
                               vector<int> sources)
    : graph(g), is_directed(is_directed), k(k),
      source_set(unordered_set<int>(sources.begin(), sources.end())) {
  for (int source : sources) {
    runDijkstra(source);
  }
}

void dijkstra_table::runDijkstra(int s) {
  priority_queue<pair<disType, pair<int, int>>, // <距离, <当前节点, 前驱节点>>
                 vector<pair<disType, pair<int, int>>>,
                 greater<pair<disType, pair<int, int>>>>
      pq;
  unordered_map<int, bool> visited;
  unordered_map<int, entry> distance;

  // 初始化所有节点的距离为无穷大，前驱为-1
  for (int i = 0; i < graph.size(); i++) {
    distance[i] = {numeric_limits<disType>::max(), -1};
  }

  // 将源节点加入优先队列
  pq.push({0, {s, s}});
  distance[s] = {0, s};

  while (!pq.empty()) {
    disType dist = pq.top().first;
    int u = pq.top().second.first;
    int parent = pq.top().second.second;
    pq.pop();

    // 如果节点已经被访问，跳过
    if (visited.find(u) != visited.end())
      continue;

    visited[u] = true;

    // 遍历邻接节点
    for (auto &[v, weight] : graph[u]) {
      if (distance[u].second == parent) {
        int hops = distance[u].second; // 当前节点的跳数
        if (hops + 1 > this->k)
          continue; // 跳数限制
      }

      disType new_dist = dist + weight;
      if (new_dist < distance[v].first) {
        distance[v].first = new_dist;
        distance[v].second = u;
        pq.push({new_dist, {v, u}});
      }
    }
  }

  // 保存计算结果到查询表
  query_table[s] = distance;
}

vector<int> dijkstra_table::query_path(int s, int t) {
  vector<int> path;
  if (query_table.find(s) != query_table.end()) {
    int cur = t;
    while (cur != s) {
      path.push_back(cur);
      cur = query_table[s][cur].second;
    }
    path.push_back(s);
    reverse(path.begin(), path.end());

  } else if (query_table.find(t) != query_table.end()) {
    int cur = s;
    while (cur != t) {
      path.push_back(cur);
      cur = query_table[t][cur].second;
    }
    path.push_back(t);
  }
    return path;
  }

  disType dijkstra_table::query_distance(int s, int t) {
    disType min_distance = numeric_limits<disType>::max();

    if (query_table.find(s) != query_table.end()) {
      if (query_table[s].find(t) != query_table[s].end()) {
        min_distance = query_table[s][t].first;
      }
    } else if (query_table.find(t) != query_table.end()) {
      if (query_table[t].find(s) != query_table[t].end()) {
        min_distance = query_table[t][s].first;
      }
    }

    return min_distance;
  }
