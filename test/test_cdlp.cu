#include "chrono"
#include "graph/graph_v_of_v.h"
#include "vgroup/CDLP_group.cuh"

int main() {
  graph_v_of_v<disType> g;
  auto start = std::chrono::high_resolution_clock::now();
  g.txt_read(
      "/home/pengchang/data/wiki-RfA2.txt");

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "read graph took " << duration.count() << " seconds."
            << std::endl;
  std::vector<std::vector<int>> groups;

  start = std::chrono::high_resolution_clock::now();
  generate_Group_CDLP(g, groups);
  end = std::chrono::high_resolution_clock::now();

  duration = end - start;
  std::cout << "generate_Group_CDLP function took " << duration.count()
            << " seconds." << std::endl;

  for (auto it : groups) {
    printf("group size: %d\n", it.size());
    }
}