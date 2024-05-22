#include "generation/cuda_query.cuh"
#include <cfloat>
#include <cuda_runtime.h>

__device__ float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void find_mindis_hops(int hop_cst, cuda_vector<hub_type> *vec1, cuda_vector<hub_type> *vec2, hub_type *result_vec1, hub_type *result_vec2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float min_dis;
    __shared__ int min_hub1_vertex, min_hub1_hop, min_hub1_parent;
    __shared__ float min_hub1_dis;
    __shared__ int min_hub2_vertex, min_hub2_hop, min_hub2_parent;
    __shared__ float min_hub2_dis;

    if (threadIdx.x == 0) {
        min_dis = FLT_MAX;
    }
    __syncthreads();

    float local_min_dis = FLT_MAX;
    int local_hub1_vertex = 0, local_hub1_hop = 0, local_hub1_parent = 0;
    float local_hub1_dis = 0.0f;
    int local_hub2_vertex = 0, local_hub2_hop = 0, local_hub2_parent = 0;
    float local_hub2_dis = 0.0f;

    if (idx < vec1->size()) {
        for (int j = 0; j < vec2->size(); j++) {
            if (vec1->at(idx).hub_vertex == vec2->at(j).hub_vertex && vec1->at(idx).hop + vec2->at(j).hop <= hop_cst) {
                float dis = vec1->at(idx).distance + vec2->at(j).distance;
                if (dis < local_min_dis) {
                    local_min_dis = dis;
                    local_hub1_vertex = vec1->at(idx).hub_vertex;
                    local_hub1_hop = vec1->at(idx).hop;
                    local_hub1_dis = vec1->at(idx).distance;
                    local_hub2_vertex = vec2->at(j).hub_vertex;
                    local_hub2_hop = vec2->at(j).hop;
                    local_hub2_dis = vec2->at(j).distance;
                    local_hub1_parent = vec1->at(idx).parent_vertex;
                    local_hub2_parent = vec2->at(j).parent_vertex;
                    //printf("j:%d,  idx:%d\n",j,idx);
                }
            }
        }
    }

    atomicMin(&min_dis, local_min_dis);
    __syncthreads();

    if (min_dis == local_min_dis) {
        atomicExch(&min_hub1_vertex, local_hub1_vertex);
        atomicExch(&min_hub1_hop, local_hub1_hop);
        atomicExch(&min_hub1_dis, local_hub1_dis);
        atomicExch(&min_hub2_vertex, local_hub2_vertex);
        atomicExch(&min_hub2_hop, local_hub2_hop);
        atomicExch(&min_hub2_dis, local_hub2_dis);
        atomicExch(&min_hub1_parent, local_hub1_parent);
        atomicExch(&min_hub2_parent, local_hub2_parent);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *result_vec1 = hub_type(min_hub1_vertex, min_hub1_parent, min_hub1_hop, min_hub1_dis);
        *result_vec2 = hub_type(min_hub2_vertex, min_hub2_parent, min_hub2_hop, min_hub2_dis);
    }
}
