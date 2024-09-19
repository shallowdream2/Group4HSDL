
#include "vgroup/Kmeans_group.cuh"
#include <vector>
#include <unordered_map>
using namespace std;

void print_groups(std::vector<std::vector<int>> &groups) {
  for (int i=0;i<groups.size();++i) {
    printf("center: %d\n", i);
    for (auto it2 : groups[i]) {
      printf("%d ", it2);
    }
    printf("\n\n");
  }
}


void generate_Group_kmeans(graph_v_of_v<disType> &instance_graph, int hop_cst,
                           std::vector< std::vector<int>> &groups) {
  int n = instance_graph.size();
  vector<int> labels(n, -1);
  vector<int> centers;
  std::unordered_map<int, int> group_size;

  unordered_map<int, bool> chosen;
  srand(static_cast<unsigned int>(time(0)));
  //初始化dijkstra table
  dijkstra_table dt(instance_graph, false, 10000);


  int center = rand() % n;
  int group_num  = n/MAX_GROUP_SIZE+1;
  // 随机选择初始聚类中心
  for (int i = 0; i < group_num; ++i) {
    do {
      center = rand() % n;
    } while (chosen.find(center) != chosen.end());
    centers.push_back(center);
    chosen[center] = true;
  }

  bool changed = true;
  int max_iteration = 5;
  int iter = 0;
  while (changed && iter++ < max_iteration) {
    changed = false;
    dt.add_source(centers);

    // Assign nodes to the nearest cluster center
    for (int i = 0; i < n; ++i) {
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

    // 更新聚类中心
    if (changed) {
      for (int i = 0; i < group_num; ++i) {
        vector<int> cluster;
        dt.add_source(cluster);
        // 计算簇的重心（中心）并选择新的聚类中心
        int new_center = find_new_center(cluster, instance_graph, dt);
        centers[i] = new_center;
      }
    }
  }
  groups.resize(n);

  // 构建最终的groups
  for (int i = 0; i < n; ++i) {
    groups[labels[i]].push_back(i);
  }
}

int find_new_center(vector<int> &cluster, graph_v_of_v<disType> &graph,
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
