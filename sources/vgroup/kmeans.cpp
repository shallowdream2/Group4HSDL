#include "utilities/dijkstra.h"
#include "vgroup/Groups.h"
#include <algorithm>
using namespace std;

void generate_Group_kmeans(graph_v_of_v<int> &instance_graph, int group_num,
                           int hop_cst,
                           std::unordered_map<int, std::vector<int>> &groups) {
  int n = instance_graph.size();
  vector<int> centers(group_num);
  vector<int> labels(n, -1);

  // 1. 无重复的随机选择group_num个点作为初始的聚类中心
  for (int i = 0; i < group_num; ++i) {
    do {
      centers[i] = rand() % n;
    } while (find(centers.begin(), centers.end(), centers[i]) !=
             centers.begin() + i);
  }

  dijkstra_table dt(instance_graph, false, hop_cst, centers);

  bool changed = 1;
  while (changed) {
    changed = false;
    // 2.
    // 对于每个点，计算它到每个聚类中心的距离，将它划分到距离最近的聚类中心所在的类中
    for (int i = 0; i < n; ++i) {
      int nearest_center = -1;
      double min_distance = numeric_limits<int>::max();
      for (int j = 0; j < group_num; ++j) {
        double distance = dt.query_distance(centers[j], i);
        //printf("source:%d,dst:%d,dis:%lf\n", i, j, distance);
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

    // 3.
    // 重新计算聚类中心（这个步骤在图聚类中略有不同，因为我们不能简单地取平均值）
    // 这里我们留空，因为重新计算聚类中心在图中意义不大，通常保持选择的中心不变

  } // 如果在一轮迭代中聚类结果没有变化，则算法结束


  
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
