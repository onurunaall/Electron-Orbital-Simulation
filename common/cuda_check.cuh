#pragma once


#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>


#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                                \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)


inline void cudaCheckKernel() {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Grid size for a 1D launch with `threadsPerBlock` threads.
inline int blocksFor(int count, int threadsPerBlock) {
    return (count + threadsPerBlock - 1) / threadsPerBlock;
}
