//#include "graph/graph_v_of_v.h"
#include "utilities/dijkstra.h"

#include <iostream>
#define DATASET_PATH "/mnt/f/linux/rucgraph-HBPLL-GPU/data/euroroad2.txt"
using namespace std;

int main() {

  // init graph
  graph_v_of_v<int> graph;
  graph.txt_read(DATASET_PATH);

  // init dijkstra
  dijkstra_table dijkstra(graph);
  dijkstra.add_source(2);
  cout<<dijkstra.query_distance(2,3);
  

}