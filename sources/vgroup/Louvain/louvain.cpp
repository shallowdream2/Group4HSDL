
#include "vgroup/Louvain/louvain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "str.h"
#include <cassert>
#include <sys/timeb.h>

static void malloc_louvain(Louvain *lv) {
  int i;
  lv->cindex = new int[lv->clen];
  lv->nodes = new Node[lv->nlen];
  lv->edges = new Edge[lv->elen];
  for (i = 0; i < lv->clen; i++) {
    lv->nodes[i].eindex = -1;
    lv->cindex[i] = -1;
  }
  // memset(lv->cindex, -1, lv->clen * sizeof(int));
}

void init_node(_louvain *lv, int I, int weight) {
  if (lv->cindex[I] == -1) {
    lv->cindex[I] = I;
    lv->nodes[I].count = 1;
    lv->nodes[I].kin = 0;
    lv->nodes[I].clskin = 0;
    lv->nodes[I].clsid = I;
    lv->nodes[I].first = -1;
    lv->nodes[I].prev = -1;
    lv->nodes[I].next = -1;
  }
  lv->nodes[I].kout += weight;
  lv->nodes[I].clstot += weight;
}

void link_edge(_louvain *lv, int l, int r, int &ei, int weight) {
  lv->edges[ei].left = l;
  lv->edges[ei].right = r;
  lv->edges[ei].weight = weight;
  lv->edges[ei].next = lv->nodes[l].eindex;
  lv->nodes[l].eindex = ei;
  ei += 1;
}

Louvain *mycreate_louvain(graph_v_of_v<int> &G) {

  std::vector<std::vector<std::pair<int, int>>> &graph = G.ADJs;
  Louvain *lv = new Louvain;
  // Hash *hs = hash_create(1 << 23, INT); // 创建哈希表来存储节点
  int ei = 0;
  double weight = 1.0;

  // 初始化 Louvain 结构体
  lv->clen = (int)graph.size();
  lv->elen = (int)(G.edge_number() * 2);
  lv->nlen = lv->clen;
  lv->olen = lv->elen;
  malloc_louvain(lv);

  // 重新遍历图，将边信息加入 Louvain 结构体
  for (size_t u = 0; u < graph.size(); u++) {
    for (auto &v_weight_pair : graph[u]) {
      int v = v_weight_pair.first;
      weight = static_cast<double>(v_weight_pair.second);

      lv->sumw += weight;

      init_node(lv, u, weight);
      // init_node(lv, v, weight);
      link_edge(lv, u, v, ei, weight);
      // link_edge(lv, v, u, ei, weight);
      if (ei > lv->elen) {
        assert(false);
      }
    }
  }

  return lv;
}

static void add_node_to_comm(Louvain *lv, int id, int cid, double weight) {
  lv->nodes[id].clsid = cid;
  lv->nodes[id].next = lv->nodes[cid].next;
  lv->nodes[cid].next = id;
  lv->nodes[id].prev = cid;
  if (lv->nodes[id].next != -1) {
    lv->nodes[lv->nodes[id].next].prev = id;
  }
  lv->nodes[cid].count += lv->nodes[id].count;
  lv->nodes[cid].clstot += lv->nodes[id].clstot;
  lv->nodes[cid].clskin += lv->nodes[id].kin + 2 * weight;
}

static void remove_node_from_comm(Louvain *lv, int id, double weight) {
  int cid = lv->nodes[id].clsid;
  int prev, next;
  if (cid != id) {
    prev = lv->nodes[id].prev;
    next = lv->nodes[id].next;
    lv->nodes[prev].next = next;
    if (next != -1) {
      lv->nodes[next].prev = prev;
    }
    lv->nodes[cid].count -= lv->nodes[id].count;
    lv->nodes[cid].clstot -= lv->nodes[id].clstot;
    lv->nodes[cid].clskin -= lv->nodes[id].kin + 2 * weight;
  } else {
    next = lv->nodes[id].next; // the new center of the community
    cid = next;                // cid , new center
    if (-1 != next) {
      lv->nodes[next].prev = -1;
      lv->nodes[next].clsid = next;
      while (-1 != (next = lv->nodes[next].next)) {
        lv->nodes[cid].count += lv->nodes[next].count;
        lv->nodes[next].clsid = cid;
      }
      lv->nodes[cid].clstot =
          lv->nodes[id].clstot - lv->nodes[id].kin - lv->nodes[id].kout;
      lv->nodes[cid].clskin =
          lv->nodes[id].clskin - lv->nodes[id].kin - 2 * weight;
      lv->nodes[id].count -= lv->nodes[cid].count;
      lv->nodes[id].clskin = lv->nodes[id].kin;
      lv->nodes[id].clstot -= lv->nodes[cid].clstot;
    }
  }
}

static int first_stage(Louvain *lv) {
  int i, j, ci, cid, ei, wi, wci, cct, idc, maxId, stage_two;
  int *ids = NULL;
  double kv, wei, cwei, maxInWei, deltaQ, maxDeltaQ;
  double *weight = NULL;
  ids = new int[lv->clen];
  // weight = (double *)calloc(lv->nlen, sizeof(double));
  weight = new double[lv->nlen];
  memset(ids, -1, lv->clen * sizeof(int));
  stage_two = 0;
  while (1) {
    cct = 0;
    for (i = 0; i < lv->clen; i++) {
      ci = lv->cindex[i];
      kv = lv->nodes[ci].kin + lv->nodes[ci].kout;
      cid = lv->nodes[ci].clsid;
      ei = lv->nodes[ci].eindex;
      idc = 0;
      while (-1 != ei) {
        wi = lv->edges[ei].right;
        wei = lv->edges[ei].weight;
        wci = lv->nodes[wi].clsid;
        weight[wci] += wei;
        ids[idc++] = wci;
        ei = lv->edges[ei].next;
      }
      maxInWei = cwei = maxDeltaQ = 0.0;
      maxId = -1;
      for (j = 0; j < idc; j++)
        if (weight[ids[j]] > 0.0) {
          if (cid == ids[j]) {
            deltaQ = weight[ids[j]] -
                     kv * (lv->nodes[ids[j]].clstot - kv) / lv->sumw;
            cwei = weight[ids[j]];
          } else {
            deltaQ = weight[ids[j]] - kv * lv->nodes[ids[j]].clstot / lv->sumw;
          }
          if (deltaQ > maxDeltaQ) {
            maxDeltaQ = deltaQ;
            maxId = ids[j];
            maxInWei = weight[ids[j]];
          }
          weight[ids[j]] = 0.0;
        }
      if (maxDeltaQ > 0.0 && maxId != cid) {
        if (maxId == -1) {
          fprintf(stderr, "this can not be, something must be wrong\n");
          return 0;
        }
        remove_node_from_comm(lv, ci, cwei);
        add_node_to_comm(lv, ci, maxId, maxInWei);
        cct += 1;
        stage_two = 1;
      }
    }
    fprintf(stderr, "    one iteration inner first stage, changed nodes : %d\n",
            cct);
    if (cct == 0) {
      break;
    }
  }
  free(ids);
  ids = NULL;
  free(weight);
  weight = NULL;
  return stage_two;
}

static void second_stage(Louvain *lv) {
  int i, ci, next, first, tclen = 0;
  int l, r, telen = 0, lcid, rcid;
  double w;
  for (i = 0; i < lv->clen; i++) {
    ci = lv->cindex[i];
    if (lv->nodes[ci].clsid == ci) {
      lv->cindex[tclen++] = ci;
      next = lv->nodes[ci].next;
      first = lv->nodes[ci].first;
      if (first != -1) {
        while (-1 != (lv->nodes[first].next)) {
          first = lv->nodes[first].next;
        }
        lv->nodes[first].next = next;
      } else {
        lv->nodes[ci].first = next;
      }
      if (next != -1) {
        lv->nodes[next].prev = first;
      }
      lv->nodes[ci].next = -1;
      lv->nodes[ci].prev = -1;
    }
  }
  lv->clen = tclen;
  for (i = 0; i < lv->clen; i++) {
    ci = lv->cindex[i];
    lv->nodes[ci].kin = lv->nodes[ci].clskin;
    lv->nodes[ci].kout = lv->nodes[ci].clstot - lv->nodes[ci].kin;
    lv->nodes[ci].eindex = -1;
  }
  for (i = 0; i < lv->elen; i++) {
    l = lv->edges[i].left;
    r = lv->edges[i].right;
    w = lv->edges[i].weight;
    lcid = lv->nodes[l].clsid;
    rcid = lv->nodes[r].clsid;
    if (lcid != rcid) {
      lv->edges[telen].left = lcid;
      lv->edges[telen].right = rcid;
      lv->edges[telen].weight = w;
      lv->edges[telen].next = lv->nodes[lcid].eindex;
      lv->nodes[lcid].eindex = telen++;
    }
  }
  lv->elen = telen;
}

int learn_louvain(Louvain *lv) {
  int circle = 0;
  while (first_stage(lv) && circle <= MAX_CIRCLE) {
    second_stage(lv);
    fprintf(stderr, "community count: %d after one pass\n", lv->clen);
    circle++;
  }
  return 0;
}

void save_louvain(Louvain *lv) {}

void free_louvain(Louvain *lv) {
  delete[] lv->cindex;
  delete[] lv->nodes;
  delete[] lv->edges;
}
