#ifndef KERNELS
#define KERNELS

#include <stdio.h>

// Kernel implementations of SGEMM (Single precision GEneral Matrix Multiply)
// Performs the operation C = α*(A@B)+β*C for matrices A, B, C and scalars α, β
// where A is MxK, B is KxN, and C is MxN

// -------------------------------------------------------------------------------------
// Sequential implementations

void sgemm_sequential(int M, int K, int N, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    // TODO: Implement!
}

// -------------------------------------------------------------------------------------
// Kernel 1 implementation

__global__ void sgemm_naive(int M, int K, int N, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // TODO: Implement!
}

// -------------------------------------------------------------------------------------
// Kernel 2 implementation

__global__ void sgemm_global_coalescing(int M, int K, int N, float alpha, const float *A,
                                       const float *B, float beta, float *C) {
    // TODO: Implement!
}

// -------------------------------------------------------------------------------------
// Kernel 3 implementation

template <const int SMCACHEDIM>
__global__ void sgemm_shared_mem_cache(int M, int K, int N, float alpha, const float *A,
                                       const float *B, float beta, float *C) {
    __shared__ float blockA[SMCACHEDIM * SMCACHEDIM];
    __shared__ float blockB[SMCACHEDIM * SMCACHEDIM];

    // TODO: Implement!
}

// -------------------------------------------------------------------------------------
// Kernel 4 implementation

template <const int BM, const int BN, const int BK, const int TM>
__global__ void __launch_bounds__(1024, 1)
sgemm_1D_thread_tiling(int M, int N, int K, float alpha, const float *A, 
                      const float *B, float beta, float *C) {
    __shared__ float blockA[BM * BK];
    __shared__ float blockB[BK * BN];

    // TODO: Implement!
}

// -------------------------------------------------------------------------------------
// Kernel 5 implementation

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(1024, 1)
sgemm_2D_thread_tiling(int M, int N, int K, float alpha, const float *A,
                      const float *B, float beta, float *C) {
    __shared__ float blockA[BM * BK];
    __shared__ float blockB[BK * BN];

    // TODO: Implement!
}

// -------------------------------------------------------------------------------------
// Kernel launcher

inline void launch_kernel(long kernel_num, int M, int K, int N, float alpha, 
                          const float *d_A, const float *d_B, float beta, float *d_C) {
    if(kernel_num == 1) {
        // TODO: Set execution configuration parameters
        dim3 thr_per_blk;
        dim3 blk_in_grid;
        
        sgemm_naive<<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 2) {
        // TODO: Set execution configuration parameters
        dim3 thr_per_blk;
        dim3 blk_in_grid;

        sgemm_global_coalescing<<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 3) {
        const uint SMCACHEDIM = 32;

        // TODO: Set execution configuration parameters
        dim3 thr_per_blk;
        dim3 blk_in_grid;

        sgemm_shared_mem_cache<SMCACHEDIM><<<blk_in_grid, thr_per_blk>>>(M, K, N, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 4) {
        const uint BM = 64;
        const uint BN = 64;
        const uint BK = 8;
        const uint TM = 8;

        // TODO: Set execution configuration parameters
        dim3 blk_in_grid;
        dim3 thr_per_blk;

        sgemm_1D_thread_tiling<BM, BN, BK, TM><<<blk_in_grid, thr_per_blk>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else if(kernel_num == 5) {
        const uint BM = 128;
        const uint BN = 128;
        const uint BK = 8;
        const uint TM = 8;
        const uint TN = 8;

        // TODO: Set execution configuration parameters
        dim3 blk_in_grid;
        dim3 thr_per_blk;

        sgemm_2D_thread_tiling<BM, BN, BK, TM, TN><<<blk_in_grid, thr_per_blk>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    } else {
        fprintf(stderr, "Error: %lu is not a valid kernel number.\n", kernel_num);
        exit(1);
    }
}

#endif // KERNELS
