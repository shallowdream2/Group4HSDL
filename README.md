# Group4HSDL
This is a  repository for rucgraph GPU Project

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

- Run the test_cuda_vector
```bash
./test/test_cuda_vector
```

## 1. Memory Management
It is a basic facility to manage CUDA memory dynamically  
###  Memory pool  
We construct a consecutive array to store our Labels and Queues.  
Specifically, we divide the one dimension array into short slices, and length of each slice is up to 100.  


