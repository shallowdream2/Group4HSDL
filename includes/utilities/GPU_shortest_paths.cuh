#ifndef WS_SSSP_H
#define WS_SSSP_H

#include <stdio.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <device_atomic_functions.h>
#include <utilities/GPU_csr.hpp>


__device__ __forceinline__ double atomicMinDouble (double * addr, double value);

static __global__ void Relax(int* offsets, int* edges, double* weights, double* dis, int* queue, int* queue_size, int* visited);
static __global__ void CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited);
static void gpu_shortest_paths(CSR_graph<double>& input_graph, int source, std::vector<double>& distance,cudaStream_t stream = 0,double max_dis = 10000000000);
static void gpu_sssp_pre(CSR_graph<double>& input_graph, int source, std::vector<double>& distance, std::vector<int>& pre_v, double max_dis = 10000000000);

// std::vector<std::pair<std::string, double>> Cuda_SSSP(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, double max_dis = 10000000000);
// std::vector<std::pair<std::string, double>> Cuda_SSSP_pre(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, std::vector<int>& pre_v, double max_dis = 10000000000);

// this function is used to get the minimum value of double type atomically
__device__ __forceinline__ double atomicMinDouble (double * addr, double value) {
    double old;
    old = __longlong_as_double(atomicMin((long long *)addr, __double_as_longlong(value)));
    return old;
}

static void CUDART_CB free_memory_callback(cudaStream_t stream, cudaError_t status, void *userData) {
    // userData 是一个指向内存指针数组的指针，最后一个元素是指针数量
    void **data = (void**)userData;
    
    // 获取指针数量
    int num = *((int*)data[0]);

    for (int i = 1; i <= num; ++i) {
        if (data[i] != nullptr) {
            cudaError_t err = cudaFree(data[i]);
            if (err != cudaSuccess) {
                printf("CUDA free error: %s\n", cudaGetErrorString(err));
            }
            data[i] = nullptr;  // 防止重复释放
        }
    }

    // 释放 userData 内存
    free(data[0]);  // 释放保存指针数量的内存
    free(data);
}



static __global__ void Relax(int* out_pointer, int* out_edge, double* out_edge_weight, double* dis, int* queue, int* queue_size, int* visited) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *queue_size) {
        int v = queue[idx];

        // for all adjacent vertices
        for (int i = out_pointer[v]; i < out_pointer[v + 1]; i++) {
            int new_v = out_edge[i];
            double weight = out_edge_weight[i];

            double new_w = dis[v] + weight;

            // try doing relaxation
            double old = atomicMinDouble(&dis[new_v], new_w);

            if (old <= new_w)
				continue;

            // if the distance is updated, set the vertex as visited
            atomicExch(&visited[new_v], 1);
        }
    }
}

static __global__ void CompactQueue(int V, int* next_queue, int* next_queue_size, int* visited) {
    // this function is used to ensure that each necessary vertex is only pushed into the queue once
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V && visited[idx]) {
        int pos = atomicAdd(next_queue_size, 1);
        next_queue[pos] = idx;
        // reset the visited flag
        visited[idx] = 0;
    }
}

static void gpu_shortest_paths(CSR_graph<double>& input_graph, int source, std::vector<double>& distance,cudaStream_t stream,double max_dis ) {
    int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();

    double* dis;
    int* out_edge = input_graph.out_edge;
    double* out_edge_weight = input_graph.out_edge_weight;
    int* out_pointer = input_graph.out_pointer;
    int* visited;
    
    int* queue, * next_queue;
    int* queue_size, * next_queue_size;

    // allocate memory on GPU
    cudaMallocManaged((void**)&dis, V * sizeof(double));
    cudaMallocManaged((void**)&visited, V * sizeof(int));
    cudaMallocManaged((void**)&queue, V * sizeof(int));
    cudaMallocManaged((void**)&next_queue, V * sizeof(int));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&next_queue_size, sizeof(int));

    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // initialize the distance array and visited array
    for (int i = 0; i < V; i++) {
        dis[i] = max_dis;
        visited[i] = 0;
    }
    dis[source] = 0;

    *queue_size = 1;
    queue[0] = source;
    *next_queue_size = 0;

    int threadsPerBlock = 1024;
    int numBlocks = 0;

    while (*queue_size > 0) {
        numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
        // launch the kernel function to relax the edges
        Relax <<< numBlocks, threadsPerBlock, 0, stream >>> (out_pointer, out_edge, out_edge_weight, dis, queue, queue_size, visited);

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Relax kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
        // do the compact operation
        CompactQueue <<< numBlocks, threadsPerBlock, 0, stream >>> (V, next_queue, next_queue_size, visited);

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "CompactQueue kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        // swap the queue and next_queue
        std::swap(queue, next_queue);
        *queue_size = *next_queue_size;
        *next_queue_size = 0;
    }

    // Use cudaMemcpyAsync to copy data back to host without blocking, and synchronize with stream
    cudaMemcpyAsync(distance.data(), dis, V * sizeof(double), cudaMemcpyDeviceToHost, stream);
    

    // int num_ptrs = 5;
    // void **userData = (void**)malloc((num_ptrs + 1) * sizeof(void*));
    // userData[0] = (void*)malloc(sizeof(int));
    // *((int*)userData[0]) = num_ptrs;  // 将指针数量存储在第一个元素中
    // //userData[1] = (void*)dis;
    // userData[1] = (void*)visited;
    // userData[2] = (void*)queue;
    // userData[3] = (void*)next_queue;
    // userData[4] = (void*)queue_size;
    // userData[5] = (void*)next_queue_size;

    // 在流完成后调用回调函数，释放内存
    //cudaStreamAddCallback(stream, free_memory_callback, userData, 0);
    

    // Free memory
    // cudaFree(dis);
    // cudaFree(visited);
    // cudaFree(queue);
    // cudaFree(next_queue);
    // cudaFree(queue_size);
    // cudaFree(next_queue_size);

    return;
}

static void gpu_shortest_paths_new(CSR_graph<double>& input_graph, int source, std::vector<double>& distance, disType* d_distance,cudaStream_t stream,double max_dis ) {
    int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();

    double* dis = d_distance;
    int* out_edge = input_graph.out_edge;
    double* out_edge_weight = input_graph.out_edge_weight;
    int* out_pointer = input_graph.out_pointer;
    int* visited;
    
    int* queue, * next_queue;
    int* queue_size, * next_queue_size;

    // allocate memory on GPU
    //cudaMallocManaged((void**)&dis, V * sizeof(double));
    cudaMallocManaged((void**)&visited, V * sizeof(int));
    cudaMallocManaged((void**)&queue, V * sizeof(int));
    cudaMallocManaged((void**)&next_queue, V * sizeof(int));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&next_queue_size, sizeof(int));

    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    // initialize the distance array and visited array
    for (int i = 0; i < V; i++) {
        dis[i] = max_dis;
        visited[i] = 0;
    }
    dis[source] = 0;

    *queue_size = 1;
    queue[0] = source;
    *next_queue_size = 0;

    int threadsPerBlock = 1024;
    int numBlocks = 0;

    while (*queue_size > 0) {
        numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
        // launch the kernel function to relax the edges
        Relax <<< numBlocks, threadsPerBlock, 0, stream >>> (out_pointer, out_edge, out_edge_weight, dis, queue, queue_size, visited);

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Relax kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
        // do the compact operation
        CompactQueue <<< numBlocks, threadsPerBlock, 0, stream >>> (V, next_queue, next_queue_size, visited);

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "CompactQueue kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        // swap the queue and next_queue
        std::swap(queue, next_queue);
        *queue_size = *next_queue_size;
        *next_queue_size = 0;
    }

    // Use cudaMemcpyAsync to copy data back to host without blocking, and synchronize with stream
    cudaMemcpyAsync(distance.data(), dis, V * sizeof(double), cudaMemcpyDeviceToHost, stream);
    

    // int num_ptrs = 5;
    // void **userData = (void**)malloc((num_ptrs + 1) * sizeof(void*));
    // userData[0] = (void*)malloc(sizeof(int));
    // *((int*)userData[0]) = num_ptrs;  // 将指针数量存储在第一个元素中
    // //userData[1] = (void*)dis;
    // userData[1] = (void*)visited;
    // userData[2] = (void*)queue;
    // userData[3] = (void*)next_queue;
    // userData[4] = (void*)queue_size;
    // userData[5] = (void*)next_queue_size;

    // 在流完成后调用回调函数，释放内存
    //cudaStreamAddCallback(stream, free_memory_callback, userData, 0);
    

    // Free memory
    // cudaFree(dis);
    // cudaFree(visited);
    // cudaFree(queue);
    // cudaFree(next_queue);
    // cudaFree(queue_size);
    // cudaFree(next_queue_size);

    return;
}




// std::vector<std::pair<std::string, double>> Cuda_SSSP(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, double max_dis) {
//     int src_v_id = graph.vertex_str_to_id[src_v];
//     std::vector<double> gpuSSSPvec(graph.V, 0);
//     gpu_shortest_paths(csr_graph, src_v_id, gpuSSSPvec, max_dis);

//     // transfer the vertex id to vertex name
//     return graph.res_trans_id_val(gpuSSSPvec);
// }

#endif