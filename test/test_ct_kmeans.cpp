#include "graph/graph_v_of_v.h"
#include "vgroup/CT-Kmeans.h"

int main() {
  graph_v_of_v<int> g;
  g.txt_read("/mnt/f/linux/rucgraph-HBPLL-GPU/data/euroroad2.txt");
  std::unordered_map<int, std::vector<int>> groups;
  generate_Group_CT_cores(g, 10,groups);
  for (auto it : groups) {
    printf("group %d: ", it.first);
    for (auto i : it.second) {
      printf("%d ", i);
    }
    printf("\n");
    }
}