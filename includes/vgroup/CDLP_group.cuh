#pragma once
#include "definition/hub_def.h"
#include "graph/graph_v_of_v.h"
#include "vgroup/CDLP/GPU_Community_Detection.cuh"
#include <unordered_map>
#include <vector>

static void
generate_Group_CDLP(graph_v_of_v<disType> &instance_graph,
                    std::vector<std::vector<int>> &groups) {

  // 将图转换为CSR格式
  CSR_graph<disType> csr = graph_v_of_v_to_CSR<disType>(instance_graph);
  
  // 初始化标签向量
  std::vector<int> labels(instance_graph.size(), 0);

  // 执行CDLP算法
  CDLP(instance_graph.size(), csr, labels, 10000);

  // 确保groups的大小足够
  groups.resize(*max_element(labels.begin(), labels.end()) + 1);

  // 根据标签将节点分组
  for (int node_id = 0; node_id < labels.size(); node_id++) {
    groups[labels[node_id]].push_back(node_id);
  }
}