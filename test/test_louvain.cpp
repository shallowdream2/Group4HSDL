#include "vgroup/Louvain_group.cuh"
#include "chrono"
int main() {
  graph_v_of_v<disType> g;
  g.txt_read("/home/pengchang/data/cit-Patents2.txt");
  std::vector<std::vector<int>> groups;
  auto start = std::chrono::high_resolution_clock::now();
  generate_Group_louvain(g, 10, groups);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "generate_Group_louvain took " << duration.count() << " seconds."
            << std::endl;
  //打印每个group的大小和前十个元素
  // for(auto it:groups)
  // {
  //   printf("group_size: %d\n",it.size());
  //   for(int i=0;i<10&&i<it.size();++i)
  //   {
  //     printf("%d\n",it.at(i));
  //   }
  //   printf("\n");
  // }
}