#include <graph/graph_v_of_v.h>
#define DATASET_PATH "/mnt/f/linux/rucgraph-HBPLL-GPU/data/euroroad2.txt"
int main() {
  graph_v_of_v<int> instance_graph;

  instance_graph.txt_read(DATASET_PATH);
  printf("Graph read from %s\n", DATASET_PATH);
  printf("Number of vertices: %d\n", instance_graph.size());
}