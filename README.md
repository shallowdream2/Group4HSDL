# Group4HSDL
This is a  repository for rucgraph GPU Project

## Environment
- CUDA 11.8
- gcc 11.4.0

## Build && Test
- clone the repository
```bash
git clone https://github.com/shallowdream2/Group4HSDL.git
```

- Using cmake to build the project
```bash
cd Group4HSDL
mkdir build
cd build
cmake ..
make
```

- Run the test_cuda_vector or other test-modes
```bash
./test/test_cuda_vector
```

## Mode Details

### Memory Management

由于在生成gpu上的索引时需要一个动态的数据结构来存储数据，我们设计了一个内存管理池mmpool。

基于mmpool，我们实现了如下application

- cuda_vector     (finished)
- cuda_queue     (ongoing)

###  mmpool

mmpool是一个cuda内存管理类，实现在cuda的global memory上。

基本设计思想是申请一大块内存，防止频繁的cudaMalloc造成巨大的开销。

application向mmpool申请内存的单位是block。

每个block有nodes_per_block个node。

node的类型在 `includes/definition/hub_def`中定义。

###  cuda_vector  

`cuda_vector`是一个基于mmpool的application，运行在cuda全局内存上。

初始化时，cuda_vector需要提供一个mmpool

- eg.1

  ```cpp
  // host env
  mmpool<int> *pool;
  cudaMallocManaged(&pool, sizeof(mmpool<int>)); // 在cuda上分配内存
  new (pool) mmpool<int>(10);       // 调用构造函数，在cuda分配的内存上构造对象
  
  cuda_vector<int> *d_vector;
  cudaMallocManaged(&d_vector, sizeof(cuda_vector<int>));
  new (d_vector) cuda_vector<int>(pool);  // 调用构造函数
  
  //call global function to push_back, pop_back or anything...
  ```

### graph

在 `includes/graph/graph_v_of_v.h`中我们定义了全局的graph数据结构。并实现了读取和写入功能

### label

定义了本项目使用到的核心数据结构`hop_constrained_two_hop_label`

同时定义了管理 `hop_constrained_two_hop_label`的case_info:

`hop_constrained_case_info`

### generation

通过graph生成label，并提供`query`等服务

实现进度:

- cuda_query	(finished)
- cuda_sort       (ongoing)
- gen_label       (ongoing)



## Test

本模块用于开发中测试各个mode的功能，目前已测试的mode见`test/`

