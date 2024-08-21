
#include "vgroup/Kmeans_group.h"
#include <unordered_map>

using namespace std;

void print_groups(std::unordered_map<int, std::vector<int>> &groups) {
  for (auto it : groups) {
    printf("center: %d\n", it.first);
    for (auto it2 : it.second) {
      printf("%d ", it2);
    }
    printf("\n\n");
  }
}

// vector<int> get_centers(vector<pair<int, int>> &graph, int &start,
//                         vector<bool> &chosen, int nums) {
//   vector<int> temp_v;
//   for (; temp_v.size() < (size_t)nums && (size_t)start < graph.size();
//        ++start) {
//     if (!chosen[start]) {
//       temp_v.push_back(start);
//       chosen[start] = 1;
//     }
//   }
//   return temp_v;
// }

// void generate_Group_kmeans(graph_v_of_v<int> &instance_graph, int hop_cst,
//                            std::unordered_map<int, std::vector<int>> &groups)
//                            {

//   // initialization
//   int n = instance_graph.size();
//   // vector<int> centers(group_num, -1);
//   vector<bool> chosen(n, false);
//   vector<int> labels(n, -1);

//   //将每个点和其度数构成pair，然后按度数从大到小排序
//   vector<pair<int, int>> degree;
//   for (int i = 0; i < instance_graph.size(); ++i) {
//     degree.push_back(make_pair(i, instance_graph[i].size()));
//   }
//   sort(degree.begin(), degree.end(),
//        [&](pair<int, int> a, pair<int, int> b) { return a.second > b.second;
//        });

//   //初始化dijkstra table
//   dijkstra_table dt(instance_graph, false, hop_cst);

//   // 选取group_num个点作为初始的centers
//   int start = 0;

//   bool changed = true;
//   while (changed) {
//     changed = false;

//     vector<int> temp_centers = get_centers(degree, start, chosen, group_num);
//     dt.add_source(temp_centers);
//     // Assign nodes to the nearest cluster center
//     for (int i = 0; i < n; ++i) {
//       int nearest_center = -1;
//       double min_distance = numeric_limits<double>::max();
//       for (auto it : temp_centers) {
//         double distance = dt.query_distance(it, i);
//         if (distance <= min_distance) {
//           nearest_center = it;
//           min_distance = distance;
//           chosen[i] = true;
//         }
//       }
//       if (nearest_center != -1 && nearest_center != labels[i]) {
//         labels[i] = nearest_center;
//         changed = true;
//       }
//       if (labels[i] == -1) {
//         changed = true;
//       }
//     }
//   }

//   // 根据最终的labels数组构建groups输出
//   for (int i = 0; i < n; ++i) {
//     groups[labels[i]].push_back(i);
//   }

//   //对groups中的每个group进行排序，按点的度数从大到小排序
//   for (auto it : groups) {
//     sort(it.second.begin(), it.second.end(), [&](int a, int b) {
//       return instance_graph[a].size() > instance_graph[b].size();
//     });
//   }
// }

void generate_Group_kmeans(graph_v_of_v<int> &instance_graph, int hop_cst,
                           std::unordered_map<int, std::vector<int>> &groups) {
  int n = instance_graph.size();
  vector<int> labels(n, -1);
  vector<int> centers;
  unordered_map<int, bool> chosen;
  srand(static_cast<unsigned int>(time(0)));
  //初始化dijkstra table
  dijkstra_table dt(instance_graph, false, 10000);
  int center = rand() % n;
  // 随机选择初始聚类中心
  for (int i = 0; i < group_num; ++i) {
    do {
      center = rand() % n;
    } while (chosen.find(center) != chosen.end());
    centers.push_back(center);
    chosen[center] = true;
  }

  bool changed = true;
  while (changed) {
    changed = false;
    dt.add_source(centers);


    // Assign nodes to the nearest cluster center
    for (int i = 0; i < n; ++i) {
      int nearest_center = -1;
      double min_distance = numeric_limits<double>::max();
      for (int center : centers) {
        int distance = dt.query_distance(center, i);
        //printf("source: %d,target: %d,dis: %d\n",center,i,distance );
        if (distance < min_distance) {
          nearest_center = center;
          min_distance = distance;
        }
      }
      if (nearest_center != labels[i] && nearest_center!=-1) {
        labels[i] = nearest_center;
        changed = true;
      }
    }

    // 更新聚类中心
    if (changed) {
      for (int i = 0; i < group_num; ++i) {
        vector<int> cluster;
        // printf("center %d\n", centers[i]);
        // for (int j = 0; j < n; ++j) {
        //   if (labels[j] == centers[i]) {
        //     cluster.push_back(j);
        //     printf("%d ", j);
        //   }
        // }
        // printf("\n");
        dt.add_source(cluster);
        // 计算簇的重心（中心）并选择新的聚类中心
        int new_center = find_new_center(cluster, instance_graph, dt);
        centers[i] = new_center;
      }
    }
  }

  // 构建最终的groups
  for (int i = 0; i < n; ++i) {
    groups[labels[i]].push_back(i);
  }
}

int find_new_center(vector<int> &cluster, graph_v_of_v<int> &graph,
                    dijkstra_table &dt) {
  // 找到离簇重心最近的节点作为新的中心
  double min_sum_distance = numeric_limits<double>::max();
  int new_center = -1;
  for (int node : cluster) {
    double sum_distance = 0;
    for (int other : cluster) {
      sum_distance += dt.query_distance(node, other);
    }
    if (sum_distance < min_sum_distance) {
      min_sum_distance = sum_distance;
      new_center = node;
    }
  }
  return new_center;
}
