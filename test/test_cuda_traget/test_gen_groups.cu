//#include "graph/graph_v_of_v.h"
#include "vgroup/Kmeans_group.cuh"

#include <iostream>
#define DATASET_PATH "/mnt/f/linux/rucgraph-HBPLL-GPU/data/euroroad2.txt"
using namespace std;

int main() {

  // init graph
  graph_v_of_v<disType> graph;
  graph.txt_read(DATASET_PATH);
  std::unordered_map<int, std::vector<int>> groups;
  generate_Group_kmeans(graph,  5, groups);
  print_groups(groups);
  

}