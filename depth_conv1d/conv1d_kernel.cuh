#include <cuda_runtime.h>
#include "../cuda_error.cuh"

const int MAX_K = 5;

__global__ void conv1d_kernel(const float* u, const float* filter,
                              const float* bias, float* out,
                              int B, int L, int D, int K) {
}

inline void launch_kernel(const float* u, const float* filter,
                          const float* bias, float* out,
                          uint B, uint L, uint D, uint K) {
    dim3 blockDims;
    dim3 gridDims;

    cudaFuncSetCacheConfig(conv1d_kernel, cudaFuncCachePreferL1);
    conv1d_kernel<<<gridDims, blockDims>>>(u, filter, bias, out, B, L, D, K);
    cudaCheckError(cudaGetLastError());
}
