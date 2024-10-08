#include "vgroup/Kmeans_group.cuh"
#include "chrono"
int main() {
  graph_v_of_v<disType> g;
  g.txt_read("/home/pengchang/data/cit-Patents2.txt");
  std::vector<std::vector<int>> groups;
  auto start = std::chrono::high_resolution_clock::now();
  generate_Group_kmeans(g, 10, groups);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "generate_Group_kmeans took " << duration.count() << " seconds."
            << std::endl;
  // for (auto it : groups) {
  //   printf("group %d: ", it.first);
  //   for (auto i : it.second) {
  //     printf("%d ", i);
  //   }
  //   printf("\n");
  // }
}