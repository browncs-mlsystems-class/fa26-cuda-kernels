#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <getopt.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../cuda_error.cuh"
#include "kernels.cuh"

// -------------------------------------------------------------------------------------
// Driver and helpers

#define MAX_KERNEL 5
#define KERNEL_ERR_MSG "Error: %s is not a valid kernel number.\n"
#define DIM_ERR_MSG "Error: %s is not a valid matrix dimension.\n"

void print_usage(const char *prog_name) {
    printf("Usage: %s <kernel-num> [-M <M>] [-K <K>] [-N <N>] [-h]\n", prog_name);
    printf("Performs SGEMM, where C = α*(A@B)+β*C for matrices A, B, C and scalars α, β, "
           "where A is an MxK, B is a KxN, and C is an MxN matrix\n");
    printf("  <kernel-num>      The kernel number to run; a value of 0 runs the sequential SGEMM implementation\n");
    printf("  -M <M>            Matrix dimension M\n");
    printf("  -K <K>            Matrix dimension K\n");
    printf("  -N <N>            Matrix dimension N\n");
    printf("  -h                Show this help message and exit\n");
}

int parse_int(char* str, int* val_ptr, long min_val, long max_val, const char* bounds_err_msg) {
    char *endptr;
    long val = strtol(str, &endptr, 10);
    if(endptr == str || *endptr != '\0') {
        fprintf(stderr, "Error: %s is not a valid number.\n", str);
        return -1;
    }
    if (val < min_val || val > max_val) {
        fprintf(stderr, bounds_err_msg, str);
        return -1;
    }
    *val_ptr = (int) val;
    return 0;
}

void init_matrix(float *mat, int dim1, int dim2) {
    for (int i = 0; i < dim1 * dim2; i++){
        float tmp = (rand() % 5) + (float) (rand() % 1000) / 1000;
        if (rand() % 2 == 0) {
            tmp *= -1;
        }
        mat[i] = tmp;
    }
}

int main(int argc, char* argv[]) {
    // ----------------------------------------------------------------------
    // Parse arguments
    if(argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    int M = 4096, N = 4096, K = 4096;
    int kernel_num = 0;

    int opt;
    while ((opt = getopt(argc, argv, "hM:K:N:")) != -1) {
        switch (opt) {
            case 'M':
                if(parse_int(optarg, &M, 1, INT_MAX, DIM_ERR_MSG) != 0) {
                    exit(EXIT_FAILURE);
                }
                break;
            case 'K':
                if(parse_int(optarg, &K, 1, INT_MAX, DIM_ERR_MSG) != 0) {
                    exit(EXIT_FAILURE);
                }
                break;
            case 'N':
                if(parse_int(optarg, &N, 1, INT_MAX, DIM_ERR_MSG) != 0) {
                    exit(EXIT_FAILURE);
                }
                break;
            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
                break;
            default:
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
                break;
        }
    }

    // Handle positional arguments (<kernel-num>)
    if(optind != argc - 1) {
        fprintf(stderr, "Error: expected one positional argument, <kernel-num>\n");
        exit(EXIT_FAILURE);
    }
    if(parse_int(argv[optind], &kernel_num, 0, MAX_KERNEL, KERNEL_ERR_MSG) != 0) {
        exit(EXIT_FAILURE);
    }

    // ----------------------------------------------------------------------
    // Get device information

    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, 0));
    printf("Running on: %s with compute capability %d.%d\n", prop.name, prop.major, prop.minor);

    // ----------------------------------------------------------------------

    printf("Allocating and initializing matrices...\n");
    // Allocate space for matrices and (expected) outputs
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* out = (float*)malloc(M * N * sizeof(float));
    float* expected = (float*)malloc(M * N * sizeof(float));

    // Allocate space on GPU for matrices and (expected) outputs
    float* d_A, *d_B, *d_out, *d_expected;
    cudaCheckError(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_out, M * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_expected, M * N * sizeof(float)));

    // Initialize the matrices
    init_matrix(A, M, K);
    init_matrix(B, K, N);
    init_matrix(out, M, N);
    memcpy(expected, out, M * N * sizeof(float));

    cudaCheckError(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_out, out, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_expected, expected, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create events for measuring performance
    cudaEvent_t cublasStart, cublasStop, kernelStart, kernelStop;
    cudaCheckError(cudaEventCreate(&cublasStart));
    cudaCheckError(cudaEventCreate(&cublasStop));
    cudaCheckError(cudaEventCreate(&kernelStart));
    cudaCheckError(cudaEventCreate(&kernelStop));

    // --------------------------------------------------------------------------
    // Perform SGEMM

    // Calculate expected output using cuBLAS
    cublasHandle_t handle;
    checkCublasStatus(cublasCreate(&handle));

    // Set alpha and beta values for the operation C = alpha * A * B + beta * C
    const float alpha = 2.0f;
    const float beta = 1.0f;

    // Run cuBLAS
    printf("Running cuBLAS SGEMM...\n");
    cudaCheckError(cudaEventRecord(cublasStart));
    checkCublasStatus(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, 
                               K, &beta, d_expected, N));
    cudaCheckError(cudaEventRecord(cublasStop));
    cudaCheckError(cudaMemcpy(expected, d_expected, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    checkCublasStatus(cublasDestroy(handle));

    cudaCheckError(cudaEventSynchronize(cublasStop));
    float cublasTime = 0; // in milliseconds
    cudaCheckError(cudaEventElapsedTime(&cublasTime, cublasStart, cublasStop));

    // Calculate output using student-implemented kernel
    if(kernel_num == 0) {
        printf("Running student sequential SGEMM...\n");
        cudaCheckError(cudaEventRecord(kernelStart));
        sgemm_sequential(M, K, N, alpha, A, B, beta, out);
        cudaCheckError(cudaEventRecord(kernelStop));
        cudaCheckError(cudaEventSynchronize(kernelStop));
    } else {
        printf("Running student SGEMM kernel %d...\n", kernel_num);
        cudaCheckError(cudaEventRecord(kernelStart));
        launch_kernel(kernel_num, M, K, N, alpha, d_A, d_B, beta, d_out);
        cudaCheckError(cudaEventRecord(kernelStop));
        cudaCheckError(cudaEventSynchronize(kernelStop));
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float kernelTime = 0; // in milliseconds
    cudaCheckError(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));

    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_out));
    cudaCheckError(cudaFree(d_expected));

    cudaCheckError(cudaEventDestroy(cublasStart));
    cudaCheckError(cudaEventDestroy(cublasStop));
    cudaCheckError(cudaEventDestroy(kernelStart));
    cudaCheckError(cudaEventDestroy(kernelStop));
    
    printf("Successfully ran both implementations!\n\nChecking for correctness...\n");
    float tolerance = 2e-2;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            if(fabs(out[i * N + j] - expected[i * N + j]) > tolerance) {
                printf("\nError: value of out[%d][%d] = %f instead of %f\n", i, j, out[i * N + j], expected[i * N + j]);
                exit(1);
            }
        }
    }

    free(A);
    free(B);
    free(out);
    free(expected);

    printf("Kernel %d passed the correctness test!\n", kernel_num);
    printf("cuBLAS ran in %.3fms\n", cublasTime);
    printf("Kernel %d ran in %.3fms, %.2fx cuBLAS's runtime\n", kernel_num, 
           kernelTime, kernelTime / cublasTime);
    return 0;
}
