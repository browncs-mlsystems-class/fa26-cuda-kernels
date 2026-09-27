#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

#include "cuda_error.cuh"

inline void cudaCheckOccErr(cudaOccError ret, cudaDeviceProp prop) {
    if(ret == CUDA_OCC_ERROR_UNKNOWN_DEVICE) {
        fprintf(stderr, "Error: device %s has unsupported or unknown compute capability %d.%d\n", prop.name, prop.major, prop.minor);
        exit(1);
    }
}

int main() {
    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, 0));

    cudaOccDeviceProp occProp = prop;

    int smemGran, regGran, maxRegPerThread, maxBlocksPerMP, warpAllocGran = 0;
    cudaCheckOccErr(cudaOccSMemAllocationGranularity(&smemGran, &occProp), prop);
    cudaCheckOccErr(cudaOccRegAllocationGranularity(&regGran, &occProp), prop);
    cudaCheckOccErr(cudaOccRegAllocationMaxPerThread(&maxRegPerThread, &occProp), prop);
    cudaCheckOccErr(cudaOccMaxBlocksPerMultiprocessor(&maxBlocksPerMP, &occProp), prop);
    cudaCheckOccErr(cudaOccSubPartitionsPerMultiprocessor(&warpAllocGran, &occProp), prop);

    printf("CUDA Configuration for %s with compute capability %d.%d:\n", prop.name, prop.major, prop.minor);
    printf("========================================\n");
    printf("Max Threads per Block: %d\n", occProp.maxThreadsPerBlock);
    printf("Threads per Warp: %d\n", occProp.warpSize);
    printf("Max Threads per Multiprocessor: %d\n", occProp.maxThreadsPerMultiprocessor);
    printf("Max Thread Blocks per Multiprocessor: %d\n", maxBlocksPerMP);
    printf("Max Shared Memory per Multiprocessor: %lu bytes\n", occProp.sharedMemPerMultiprocessor);
    printf("Max Shared Memory per Block: %lu bytes\n", occProp.sharedMemPerBlock);
    printf("Shared Memory Allocation Unit Size: %d bytes\n", smemGran);
    printf("Number of 32-bit Registers per Multiprocessor: %d\n", occProp.regsPerMultiprocessor);
    printf("Register Allocation Unit Size: %d\n", regGran);
    printf("Register Allocation Granularity: per warp\n");
    printf("Warp Allocation Granularity (for register allocation): %d\n", warpAllocGran);
    printf("Max Registers per Thread: %d\n", maxRegPerThread);
    printf("Max Registers per Block: %d\n", occProp.regsPerBlock);
}
