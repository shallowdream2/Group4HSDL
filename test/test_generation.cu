#include "definition/hub_def.h"
#include "generation/gen_label.cuh"


#include <graph/graph_v_of_v.h>
#define DATASET_PATH "/mnt/f/linux/rucgraph-HBPLL-GPU/data/euroroad2.txt"

int main() {
  graph_v_of_v<weight_type> instance_graph;

  instance_graph.txt_read(DATASET_PATH);
  printf("Graph read from %s\n", DATASET_PATH);
  printf("Number of vertices: %d\n", instance_graph.size());

  hop_constrained_case_info *info = NULL;
  gen_labels_gpu(&instance_graph, info, 10);
  
}