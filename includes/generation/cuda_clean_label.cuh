#ifndef CUDA_CLEAN_LABEL_CUH
#define CUDA_CLEAN_LABEL_CUH

#include "definition/hub_def.h"
#include "memoryManagement/cuda_label.cuh"

__global__ void cuda_clean_label(cuda_label<hub_type>*L_cuda,int upper_bound,int vertex_num);

#endif