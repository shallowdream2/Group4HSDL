#pragma once
#include "definition/hub_def.h"
#include "graph/graph_v_of_v.h"
#include "vgroup/CDLP/GPU_Community_Detection.cuh"
#include <unordered_map>
#include <vector>
#include "chrono"
static void
generate_Group_CDLP(graph_v_of_v<disType> &instance_graph,
                    std::vector<std::vector<int>> &groups) {

  auto start = std::chrono::high_resolution_clock::now();
  // 将图转换为CSR格式
  CSR_graph<disType> csr = graph_v_of_v_to_CSR<disType>(instance_graph);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "transform graph took " << duration.count() << " seconds."
            << std::endl;
  
  // 初始化标签向量
  std::vector<int> labels(instance_graph.size(), 0);

  // 执行CDLP算法
  start = std::chrono::high_resolution_clock::now();
  CDLP_GPU(instance_graph.size(),csr,labels,1000);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "CDLP took " << duration.count() << " seconds."
            << std::endl;

  // 确保groups的大小足够
  groups.resize(*max_element(labels.begin(), labels.end()) + 1);

  start = std::chrono::high_resolution_clock::now();
  // 根据标签将节点分组
  for (int node_id = 0; node_id < labels.size(); node_id++) {
    groups[labels[node_id]].push_back(node_id);
  }
  //去除大小为0的group
  groups.erase(std::remove_if(groups.begin(), groups.end(),
                            [](const std::vector<int>& group) { return group.size() == 0; }),
             groups.end());
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "gen groups took " << duration.count() << " seconds."
            << std::endl;
}