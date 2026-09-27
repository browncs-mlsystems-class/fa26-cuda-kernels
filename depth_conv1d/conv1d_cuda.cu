#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <getopt.h>

#include <cuda_runtime.h>

#include "../cuda_error.cuh"
#include "conv1d_kernel.cuh"

// -------------------------------------------------------------------------------------
// Driver and helpers

#define DIM_ERR_MSG "Error: %s is not a valid tensor dimension.\n"
#define R_ERR_MSG "Error: %s is not a valid number of times to repeat.\n"

void print_usage(const char *prog_name) {
    printf("Usage: %s [-B <B>] [-D <D>] [-L <L>] [-K <K>] [-R <R>] [-h]\n", prog_name);
    printf("Performs 1D depthwise convolution for a short filter length, on an input "
           "signal of shape (B, D, L), filter of shape (D, K), and bias of shape (D), R times.\n");
    printf("  -B <M>            Batch dimension B\n");
    printf("  -D <D>            Depth dimension D\n");
    printf("  -L <L>            Length dimension L\n");
    printf("  -K <K>            Filter length K, odd and no more than %d\n", MAX_K);
    printf("  -R <R>            Number of times to run kernel\n");
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

void init_tensor(float *mat, int size) {
    for (int i = 0; i < size; i++){
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
    int B = 1, D = 8192, L = 8192, K = 3;

    int opt;
    while ((opt = getopt(argc, argv, "hB:D:L:K:R:")) != -1) {
        switch (opt) {
            case 'B':
                if(parse_int(optarg, &B, 1, INT_MAX, DIM_ERR_MSG) != 0) {
                    exit(EXIT_FAILURE);
                }
                break;
            case 'D':
                if(parse_int(optarg, &D, 1, INT_MAX, DIM_ERR_MSG) != 0) {
                    exit(EXIT_FAILURE);
                }
                break;
            case 'L':
                if(parse_int(optarg, &L, 1, INT_MAX, DIM_ERR_MSG) != 0) {
                    exit(EXIT_FAILURE);
                }
                break;
            case 'K':
                if(parse_int(optarg, &K, 1, MAX_K, DIM_ERR_MSG) != 0) {
                    exit(EXIT_FAILURE);
                }
                if(K % 2 == 0) {
                    fprintf(stderr, "Error: %d is not a valid filter length, must be odd.\n", K);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'R':
                if(parse_int(optarg, &R, 1, INT_MAX, DIM_ERR_MSG) != 0) {
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

    // Handle positional arguments
    if(optind != argc) {
        fprintf(stderr, "Error: expected no positional arguments, received %d\n", argc - optind);
        exit(EXIT_FAILURE);
    }

    // ----------------------------------------------------------------------
    // Get device information

    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, 0));
    printf("Running on: %s with compute capability %d.%d\n", prop.name, prop.major, prop.minor);

    // ----------------------------------------------------------------------

    printf("Allocating and initializing tensors...\n");
    // Allocate space for tensors and outputs
    float* u = (float*)malloc(B * D * L * sizeof(float));
    float* filter = (float*)malloc(D * K * sizeof(float));
    float* bias = (float*)malloc(D * sizeof(float));
    float* out = (float*)malloc(B * D * L * sizeof(float));

    // Allocate space on GPU for matrices and outputs
    float* d_u, *d_filter, *d_bias, *d_out;
    cudaCheckError(cudaMalloc(&d_u, B * D * L * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_filter, D * K * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_bias, D * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_out, B * D * L * sizeof(float)));

    // Initialize the matrices
    init_tensor(u, B * D * L);
    init_tensor(filter, D * K);
    init_tensor(bias, D);
    init_tensor(out, B * D * L);

    cudaCheckError(cudaMemcpy(d_u, u, B * D * L * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_filter, filter, D * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_bias, bias, D * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_out, out, B * D * L * sizeof(float), cudaMemcpyHostToDevice));

    // Create events for measuring performance
    cudaEvent_t kernelStart, kernelStop;
    cudaCheckError(cudaEventCreate(&kernelStart));
    cudaCheckError(cudaEventCreate(&kernelStop));

    // --------------------------------------------------------------------------
    // Perform 1D convolution

    // Warmup
    printf("Running student 1D depthwise convolution kernel %d times...\n", R);
    launch_kernel(d_u, d_filter, d_bias, d_out, B, L, D, K);
    cudaCheckError(cudaDeviceSynchronize());

    // Timed repeats
    cudaCheckError(cudaEventRecord(kernelStart));
    for (int r = 0; r < R; r++) {
        launch_kernel(d_u, d_filter, d_bias, d_out, B, L, D, K);
    }
    cudaCheckError(cudaEventRecord(kernelStop));
    cudaCheckError(cudaEventSynchronize(kernelStop));
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaMemcpy(out, d_out, B * D * L * sizeof(float), cudaMemcpyDeviceToHost));

    float kernelTime = 0;
    cudaCheckError(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));
    kernelTime /= float(R);

    cudaCheckError(cudaFree(d_u));
    cudaCheckError(cudaFree(d_filter));
    cudaCheckError(cudaFree(d_bias));
    cudaCheckError(cudaFree(d_out));

    cudaCheckError(cudaEventDestroy(kernelStart));
    cudaCheckError(cudaEventDestroy(kernelStop));

    if (R==1) {
        printf("Successfully ran kernel implementation!\n\nChecking for correctness...\n");
        float tolerance = 2e-2;
        int padding = K / 2;
        for(int b = 0; b < B; b++) {
            for(int d = 0; d < D; d++) {
                for(int l = 0; l < L; l++) {
                    float expected = 0;
                    for(int k = 0; k < K; k++) {
                        int u_len = l + k - padding;
                        expected += filter[d * K + k] * ((0 <= u_len && u_len < L) ? u[b * (D * L) + d * L + u_len] : 0);
                    }
                    expected += bias[d];
                    if(fabs(out[b * (D * L) + d * L + l] - expected) > tolerance) {
                        printf("\nError: value of out[%d][%d][%d] = %f instead of %f\n", b, d, l, out[b * (D * L) + d * L + l], expected);
                        exit(1);
                    }
    printf("Successfully ran kernel implementation!\n\nChecking for correctness...\n");
                }
            }
        }
    }

    printf("1D depthwise convolution kernel passed the correctness test!\n");
    printf("Kernel ran in %.3fms\n", kernelTime);

    free(u);
    free(filter);
    free(bias);
    free(out);

    return 0;
}
