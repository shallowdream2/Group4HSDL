#include "definition/hub_def.h"
#include "generation/gen_label.cuh"
#include "vgroup/CT-Kmeans.h"
#include <graph/graph_v_of_v.h>
#include <memoryManagement/cuda_label.cuh>
#include <unordered_set>
#include <boost/random/mersenne_twister.hpp> // Include the appropriate header for mt19937
#include <ctime> // Include the appropriate header for std::time
#include "graph/graph_v_of_v_generate_random_graph.h"
using namespace std;
#define DATASET_PATH "/mnt/f/linux/rucgraph-HBPLL-GPU/data/euroroad2.txt"
boost::random::mt19937 boost_random_time_seed{0};
int main() {


  // parameters
  long long V = 100;
  long long E = 300;
  double ec_min = 1;
  double ec_max = 10;

  graph_v_of_v<weight_type> instance_graph =
      graph_v_of_v_generate_random_graph<weight_type>(V, E, ec_min, ec_max, 1,
                                                      boost_random_time_seed);
  instance_graph.txt_save("/mnt/f/linux/rucgraph-HBPLL-GPU/Group4HSDL/build/graph6.txt");
  //instance_graph.txt_read(DATASET_PATH);


  int hop_cst = 2;
  printf("Graph read from %s\n", DATASET_PATH);
  printf("Number of vertices: %d\n", instance_graph.size());

  std::unordered_map<int, std::vector<int>> groups;
  generate_Group_CT_cores(instance_graph, hop_cst, groups);
  // cuda_label<hub_type> *Labels = NULL;

  //auto group = groups.begin()->second;

  hop_constrained_case_info *info;
  cudaMallocManaged(&info, sizeof(hop_constrained_case_info) * groups.size());

  int num = 0;
  for (auto it : groups) {
    new (info + num) hop_constrained_case_info();
    (info + num)->init_group(it.second, instance_graph,hop_cst);
    (info + num)->init();
    

    gen_labels_gpu(&instance_graph, info + num, hop_cst);
    num++;
    printf("Group %d done\n\n", num);
  }

  // for (int i = 0; i < num; i++) {
  //   printf("Group %d\n", i);
  //   (info + i)->print_L();
  // }
  hop_constrained_case_info final;
  final.final_label = new std::vector<std::unordered_set<hub_type>>(instance_graph.size());
  for (int i = 0; i < num; ++i) {
    final.merge_instance(*(info + i));
  }
  correctness_check(&final, &instance_graph,hop_cst);

  // for (int i = 0; i < instance_graph.size(); i++) {
  //   printf("Vertex %d: ", i);
  //   unordered_set<hub_type>&temp = final.final_label->at(i);
  //   for (const auto& it : temp) {
  //     printf("{%d,%d,%d,%d}, ", it.hub_vertex,it.parent_vertex,it.hop,it.distance);
  //   }
  //   printf("\n");
  // }
}