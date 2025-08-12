#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
struct CudaPinnedAllocator {
    typedef T value_type;

    CudaPinnedAllocator() = default;
    template <class U> 
    constexpr CudaPinnedAllocator(const CudaPinnedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        cudaError_t err = cudaMallocHost((void**)&ptr, n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* ptr, std::size_t n) noexcept {
        cudaFreeHost(ptr);
    }
};

template <class T, class U>
bool operator==(const CudaPinnedAllocator<T>&, const CudaPinnedAllocator<U>&) { return true; }

template <class T, class U>
bool operator!=(const CudaPinnedAllocator<T>&, const CudaPinnedAllocator<U>&) { return false; }

template <class T>
using CudaPinnedVector = std::vector<T, CudaPinnedAllocator<T>>;