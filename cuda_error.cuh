#ifndef CUDA_ERROR
#define CUDA_ERROR

#include <stdexcept>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__)
inline void cudaAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      std::ostringstream oss;
      oss << "CUDA error in " << file << " at line " << line << ": " << cudaGetErrorString(code);
      throw std::runtime_error(oss.str());
   }
}

inline void checkCublasStatus(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            // No error
            break;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            throw std::runtime_error("cuBLAS error: Not initialized");
        case CUBLAS_STATUS_ALLOC_FAILED:
            throw std::runtime_error("cuBLAS error: Resource allocation failed");
        case CUBLAS_STATUS_INVALID_VALUE:
            throw std::runtime_error("cuBLAS error: Invalid value");
        case CUBLAS_STATUS_ARCH_MISMATCH:
            throw std::runtime_error("cuBLAS error: Architecture mismatch");
        case CUBLAS_STATUS_EXECUTION_FAILED:
            throw std::runtime_error("cuBLAS error: Execution failed");
        case CUBLAS_STATUS_INTERNAL_ERROR:
            throw std::runtime_error("cuBLAS error: Internal error");
        default:
            std::ostringstream oss;
            oss << "cuBLAS error: Unknown error code: " << status;
            throw std::runtime_error(oss.str());
    }
}

#endif // CUDA_ERROR
