#pragma once
//#include <boost/heap/fibonacci_heap.hpp>
// #include <build_in_progress/HL/Group4HBPLL/hop_constrained_two_hop_labels.h>
// #include <build_in_progress/HL/Group4HBPLL/two_hop_labels.h>
#include "utilities/dijkstra.h"
#include <cstdio>
#include <graph/graph_v_of_v.h>
#include <shared_mutex>
#include <stdio.h>
#include <tool_functions/ThreadPool.h>
#include <unordered_map>
#include <unordered_set>

//#include <cpp_redis/cpp_redis>

#define vertex_groups std::unordered_map<int, std::vector<int>>

// algo
void generate_Group_kmeans(graph_v_of_v<int> &instance_graph, int group_num,
                           int hop_cst,
                           std::unordered_map<int, std::vector<int>> &groups);

// func
inline void print_groups(std::unordered_map<int, std::vector<int>> &groups) {
  for (auto it : groups) {
    printf("center: %d\n", it.first);
    for (auto it2 : it.second) {
      printf("%d ",it2);
    }
    printf("\n\n");
    
    }
}
using namespace std;
