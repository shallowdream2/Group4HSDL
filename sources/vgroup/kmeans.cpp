#include "graph/graph_v_of_v.h"
#include "utilities/dijkstra.h"
#include "vgroup/Groups.h"
#include <algorithm>

using namespace std;

vector<int> get_centers(vector<pair<int, int>> &graph, int &start,
                        vector<bool> &chosen, int nums) {
  vector<int> temp_v;
  for (; temp_v.size() < nums && start < graph.size(); ++start) {
    if (!chosen[start]) {
      temp_v.push_back(start);
      chosen[start] = 1;
    }
  }
  return temp_v;
}

void generate_Group_kmeans(graph_v_of_v<int> &instance_graph, int group_num,
                           int hop_cst,
                           std::unordered_map<int, std::vector<int>> &groups) {

  // initialization
  int n = instance_graph.size();
  // vector<int> centers(group_num, -1);
  vector<bool> chosen(n, false);
  vector<int> labels(n, -1);

  //将每个点和其度数构成pair，然后按度数从大到小排序
  vector<pair<int, int>> degree;
  for (int i = 0; i < instance_graph.size(); ++i) {
    degree.push_back(make_pair(i, instance_graph[i].size()));
  }
  sort(degree.begin(), degree.end(),
       [&](pair<int, int> a, pair<int, int> b) { return a.second > b.second; });

  //初始化dijkstra table
  dijkstra_table dt(instance_graph, false, hop_cst);

  // 选取group_num个点作为初始的centers
  int start = 0;

  bool changed = true;
  while (changed) {
    changed = false;

    vector<int> temp_centers = get_centers(degree, start, chosen, group_num);
    dt.add_source(temp_centers);
    // Assign nodes to the nearest cluster center
    for (int i = 0; i < n; ++i) {
      int nearest_center = -1;
      double min_distance = numeric_limits<double>::max();
      for (auto it : temp_centers) {
        double distance = dt.query_distance(it, i);
        if (distance <= min_distance) {
          nearest_center = it;
          min_distance = distance;
          chosen[i] = true;
        }
      }
      if (nearest_center != -1 && nearest_center != labels[i]) {
        labels[i] = nearest_center;
        changed = true;
      }
      if (labels[i] == -1) {
        changed = true;
      }
    }
  }

  // 根据最终的labels数组构建groups输出
  for (int i = 0; i < n; ++i) {
    groups[labels[i]].push_back(i);
  }

  //对groups中的每个group进行排序，按点的度数从大到小排序
  for (auto it : groups) {
    sort(it.second.begin(), it.second.end(), [&](int a, int b) {
      return instance_graph[a].size() > instance_graph[b].size();
    });
  }
}
