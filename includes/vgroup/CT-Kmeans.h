void generate_Group_CT_cores(
    graph_v_of_v<int> &instance_graph,
    hop_constrained_case_info &case_info, //暂时用作最短距离查询
    int hop_cst, int group_num,
    std::unordered_map<int, std::vector<int>> &groups) {

  int N = instance_graph.size();
  vector<int> labels(N, -1);
  unordered_set<int> centers;
  // generate CT cores
  CT_case_info mm;
  mm.d = 10;
  mm.use_P2H_pruning = 1;
  mm.two_hop_info.use_2M_prune = 1;
  mm.two_hop_info.use_canonical_repair = 1;
  mm.thread_num = 10;

  CT_cores(instance_graph, mm);
  printf("CT_cores finished\n");
  for (int i = 0; i < N; i++) {
    if (mm.isIntree[i] == 0) {
      centers.insert(i);
    }
  }

  bool changed;
  while (changed) {
    changed = false;
    // 2.
    // 对于每个点，计算它到每个聚类中心的距离，将它划分到距离最近的聚类中心所在的类中
    for (int i = 0; i < N; ++i) {
      int nearest_center = -1;
      int min_distance = numeric_limits<int>::max();
      for (auto j : centers) {
        int distance =
            hop_constrained_extract_distance(case_info.L, i, j, hop_cst);
        if (distance < min_distance) {
          nearest_center = j;
          min_distance = distance;
        }
      }
      if (labels[i] != nearest_center) {
        labels[i] = nearest_center;
        changed = true;
      }
    }
  }

  // 根据最终的labels数组构建groups输出
  for (int i = 0; i < N; ++i) {
    groups[labels[i]].push_back(i);
  }
}
