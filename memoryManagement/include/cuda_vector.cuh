#ifndef CUDA_VECTOR_CUH
#define CUDA_VECTOR_CUH
#include "mmpool.cuh"


template <typename T>
class cuda_vector<T> {
private:
    mmpool<T>& pool;
    size_t current_size;
    size_t capacity;

public:
    cuda_vector(){} // 初始块大小可以根据需求调整

    ~cuda_vector() = default;

    void push_back(const T& value);
    T& operator[](size_t index);
    //const T& operator[](size_t index) const;
    void clear();
    size_t size() const { return current_size; }
    bool empty() const { return current_size == 0; }
};


#endif