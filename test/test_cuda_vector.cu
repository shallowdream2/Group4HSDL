#include "mmpool.cuh"
#include "cuda_vector.cuh"  // 确保包含正确的头文件路径
#include <iostream>
#include <cuda_runtime.h>

// 核函数，用于测试 cuda_vector 功能
__global__ void test_vector(cuda_vector<int> *vec) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        // 在向量中添加一些元素
        for (int i = 0; i < 10; i++) {
            vec->push_back(i);
        }
    }

    // 确保所有线程都已完成写操作
    __syncthreads();

    if (idx < vec->size()) {
        printf("vec[%d] = %d\n", idx, (*vec)[idx]);
    }
}

int main() {
    mmpool<int> pool(10, 100);  // 假设每个块100个元素，共10个块

    // 分配和初始化 cuda_vector
    cuda_vector<int>* d_vector;
    cudaMallocManaged(&d_vector, sizeof(cuda_vector<int>));
    new (d_vector) cuda_vector<int>(pool);  // 调用构造函数

    // 启动核函数
    test_vector<<<1, 256>>>(d_vector);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // 销毁 vector 和释放资源
    d_vector->~cuda_vector<int>();
    cudaFree(d_vector);

    return 0;
}
