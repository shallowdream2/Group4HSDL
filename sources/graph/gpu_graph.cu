#include "graph/gpu_graph.cuh"

// 构造函数
gpu_Graph::gpu_Graph()
    : num_nodes(0), num_edges(0), edges(nullptr), offsets(nullptr),
      d_edges(nullptr), d_offsets(nullptr) {}

// 析构函数
gpu_Graph::~gpu_Graph() { freeGraphOnGPU(); }

// 在GPU上分配图数据结构
void gpu_Graph::allocateGraphOnGPU(int num_nodes, int num_edges) {
  this->num_nodes = num_nodes;
  this->num_edges = num_edges;

  cudaMalloc(&d_edges, num_edges * sizeof(Edge));
  cudaMalloc(&d_offsets, (num_nodes + 1) * sizeof(int));
}

// 将图数据从CPU复制到GPU
void gpu_Graph::copyGraphToGPU() {
  cudaMemcpy(d_edges, edges, num_edges * sizeof(Edge), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, offsets, (num_nodes + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
}

// 从邻接列表构造gpu_Graph
gpu_Graph::gpu_Graph(
    const std::vector<std::vector<std::pair<int, weight_type>>> &ADJs) {
  max_degree = 0;
  num_nodes = ADJs.size();
  num_edges = 0;
  for (const auto &adj : ADJs) {
    num_edges += adj.size();
    max_degree = std::max(max_degree, (int)adj.size());
  }

  edges = new Edge[num_edges];
  offsets = new int[num_nodes + 1];

  int edgeIndex = 0;
  for (int i = 0; i < num_nodes; ++i) {
    offsets[i] = edgeIndex;

    for (const auto &edge : ADJs[i]) {
      edges[edgeIndex].target = edge.first;
      edges[edgeIndex].weight = edge.second;
      ++edgeIndex;
    }
  }
  offsets[num_nodes] = edgeIndex; // 最后一个偏移量是边的总数

  // 分配GPU资源并复制数据
  allocateGraphOnGPU(num_nodes, num_edges);
  copyGraphToGPU();
}

// 在GPU上释放图数据结构
void gpu_Graph::freeGraphOnGPU() {
  if (d_edges != nullptr) {
    cudaFree(d_edges);
    d_edges = nullptr;
  }
  if (d_offsets != nullptr) {
    cudaFree(d_offsets);
    d_offsets = nullptr;
  }

  if (edges != nullptr) {
    delete[] edges;
    edges = nullptr;
  }
  if (offsets != nullptr) {
    delete[] offsets;
    offsets = nullptr;
  }
}