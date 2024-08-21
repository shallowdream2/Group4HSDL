#ifndef LOUVAIN_GROUP_H
#define LOUVAIN_GROUP_H
#include "graph/graph_v_of_v.h"
#include "vgroup/Louvain/louvain.h"
#include <unordered_map>
#include <vector>

static void
generate_Group_louvain(graph_v_of_v<int> &instance_graph, int hop_cst,
                       std::unordered_map<int, std::vector<int>> &groups) {
  Louvain *lv = mycreate_louvain(instance_graph);
  learn_louvain(lv);
  printf("community number: %d\n", lv->clen);

  //找到每个点的clsid
  for (int i = 0; i < instance_graph.size(); i++) {
    int clsid = lv->nodes[i].clsid;
    groups[clsid].push_back(i);
  }
}
#endif