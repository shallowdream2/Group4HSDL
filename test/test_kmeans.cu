#include "vgroup/Kmeans_group.cuh"

int main() {
  graph_v_of_v<disType> g;
  g.txt_read("/mnt/f/linux/rucgraph-HBPLL-GPU/data/euroroad2.txt");
  std::unordered_map<int, std::vector<int>> groups;

  generate_Group_kmeans(g, 10, groups);

  for (auto it : groups) {
    printf("group %d: ", it.first);
    for (auto i : it.second) {
      printf("%d ", i);
    }
    printf("\n");
  }
}