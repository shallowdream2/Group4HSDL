#include "chrono"
#include "graph/graph_v_of_v.h"
#include "vgroup/CT-Kmeans.cuh"

int main() {
  graph_v_of_v<disType> g;
  auto start = std::chrono::high_resolution_clock::now();
  g.txt_read(
      "/mnt/f/linux/rucgraph-HBPLL-GPU/data/cit-patents/cit-Patents2.txt");

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "read graph took " << duration.count() << " seconds."
            << std::endl;
  std::vector<std::vector<int>> groups;

  start = std::chrono::high_resolution_clock::now();
  generate_Group_CT_cores(g, 10, groups);
  end = std::chrono::high_resolution_clock::now();

  duration = end - start;
  std::cout << "generate_Group_CT_cores function took " << duration.count()
            << " seconds." << std::endl;

  // for (auto it : groups) {
  //   printf("group %d: ", it.first);
  //   for (auto i : it.second) {
  //     printf("%d ", i);
  //   }
  //   printf("\n");
  //   }
}