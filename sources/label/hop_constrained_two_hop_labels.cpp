#include "label/hop_constrained_two_hop_labels.h"

int hop_constrained_extract_distance(
    vector<vector<hop_constrained_two_hop_label>> &L, int source, int terminal,
    int hop_cst) {

  /*return std::numeric_limits<int>::max() is not connected*/

  if (hop_cst < 0) {
    return std::numeric_limits<int>::max();
  }
  if (source == terminal) {
    return 0;
  } else if (hop_cst == 0) {
    return std::numeric_limits<int>::max();
  }

  long long int distance = std::numeric_limits<int>::max();
  auto vector1_check_pointer = L[source].begin();
  auto vector2_check_pointer = L[terminal].begin();
  auto pointer_L_s_end = L[source].end(), pointer_L_t_end = L[terminal].end();

  while (vector1_check_pointer != pointer_L_s_end &&
         vector2_check_pointer != pointer_L_t_end) {
    if (vector1_check_pointer->hub_vertex ==
        vector2_check_pointer->hub_vertex) {

      auto vector1_end = vector1_check_pointer;
      while (vector1_check_pointer->hub_vertex == vector1_end->hub_vertex &&
             vector1_end != pointer_L_s_end) {
        vector1_end++;
      }
      auto vector2_end = vector2_check_pointer;
      while (vector2_check_pointer->hub_vertex == vector2_end->hub_vertex &&
             vector2_end != pointer_L_t_end) {
        vector2_end++;
      }

      for (auto vector1_begin = vector1_check_pointer;
           vector1_begin != vector1_end; vector1_begin++) {
        // cout << "x (" << vector1_begin->hub_vertex << "," <<
        // vector1_begin->hop << "," << vector1_begin->distance << "," <<
        // vector1_begin->parent_vertex << ") " << endl;
        for (auto vector2_begin = vector2_check_pointer;
             vector2_begin != vector2_end; vector2_begin++) {
          // cout << "y (" << vector2_begin->hub_vertex << "," <<
          // vector2_begin->hop << "," << vector2_begin->distance << "," <<
          // vector2_begin->parent_vertex << ") " << endl;
          if (vector1_begin->hop + vector2_begin->hop <= hop_cst) {
            long long int dis = (long long int)vector1_begin->distance +
                                vector2_begin->distance;
            if (distance > dis) {
              distance = dis;
            }
          } else {
            break;
          }
        }
      }

      vector1_check_pointer = vector1_end;
      vector2_check_pointer = vector2_end;
    } else if (vector1_check_pointer->hub_vertex >
               vector2_check_pointer->hub_vertex) {
      vector2_check_pointer++;
    } else {
      vector1_check_pointer++;
    }
  }

  return distance;
}

vector<pair<int, int>> hop_constrained_extract_shortest_path(
    vector<vector<hop_constrained_two_hop_label>> &L, int source, int terminal,
    int hop_cst) {

  vector<pair<int, int>> paths;

  if (source == terminal) {
    return paths;
  }

  /* Nothing happened */
  /* In this case, the problem that the removed vertices appear in the path
   * needs to be solved */
  int vector1_capped_v_parent, vector2_capped_v_parent;
  long long int distance = std::numeric_limits<int>::max();
  auto vector1_check_pointer = L[source].begin();
  auto vector2_check_pointer = L[terminal].begin();
  auto pointer_L_s_end = L[source].end(), pointer_L_t_end = L[terminal].end();

  while (vector1_check_pointer != pointer_L_s_end &&
         vector2_check_pointer != pointer_L_t_end) {
    if (vector1_check_pointer->hub_vertex ==
        vector2_check_pointer->hub_vertex) {

      auto vector1_end = vector1_check_pointer;
      while (vector1_check_pointer->hub_vertex == vector1_end->hub_vertex &&
             vector1_end != pointer_L_s_end) {
        vector1_end++;
      }
      auto vector2_end = vector2_check_pointer;
      while (vector2_check_pointer->hub_vertex == vector2_end->hub_vertex &&
             vector2_end != pointer_L_t_end) {
        vector2_end++;
      }

      for (auto vector1_begin = vector1_check_pointer;
           vector1_begin != vector1_end; vector1_begin++) {
        for (auto vector2_begin = vector2_check_pointer;
             vector2_begin != vector2_end; vector2_begin++) {
          if (vector2_begin->hop + vector1_begin->hop <= hop_cst) {
            long long int dis = (long long int)vector1_begin->distance +
                                vector2_begin->distance;
            if (distance > dis) {
              distance = dis;
              vector1_capped_v_parent = vector1_begin->parent_vertex;
              vector2_capped_v_parent = vector2_begin->parent_vertex;
            }
          } else {
            break;
          }
        }
      }

      vector1_check_pointer = vector1_end;
      vector2_check_pointer = vector2_end;
    } else if (vector1_check_pointer->hub_vertex >
               vector2_check_pointer->hub_vertex) {
      vector2_check_pointer++;
    } else {
      vector1_check_pointer++;
    }
  }

  if (distance < std::numeric_limits<int>::max()) { // connected
    if (source != vector1_capped_v_parent) {
      paths.push_back({source, vector1_capped_v_parent});
      source = vector1_capped_v_parent;
      hop_cst--;
    }
    if (terminal != vector2_capped_v_parent) {
      paths.push_back({terminal, vector2_capped_v_parent});
      terminal = vector2_capped_v_parent;
      hop_cst--;
    }
  } else {
    return paths;
  }

  // find new edges
  vector<pair<int, int>> new_edges =
      hop_constrained_extract_shortest_path(L, source, terminal, hop_cst);

  paths.insert(paths.end(), new_edges.begin(), new_edges.end());

  return paths;
}