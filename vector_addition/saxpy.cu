#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <getopt.h>

#include "../cuda_error.cuh"

// SAXPY kernel
// Calculates z = α * x + y
__global__ void saxpy(size_t n, float a, float* x, float* y, float* z) {
    // TODO: Implement!
}

float get_rand_float() {
    float tmp = (rand() % 5) + (float) (rand() % 1000) / 1000;
    if (rand() % 2 == 0) {
        tmp *= -1;
    }
    return tmp;
}

void usage(const char* progname) {
    printf("Usage: %s [-n <N>] [-h]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <N>  Number of elements in each array\n");
    printf("  -h  --help           Show this message and exit\n");
}

int main(int argc, char* argv[]) {
    // ----------------------------------------------------------------------
    // Parse command line arguments
    int N = 100 * 1000 * 1000;

    struct option long_options[] = {
        {"arraysize", 1, 0, 'n'},
        {"help",      0, 0, 'h'},
        {0, 0, 0, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "hn:", long_options, NULL)) != -1) {
        switch (opt) {
        case 'n':
            N = atoi(optarg);
            if(N <= 0) {
                fprintf(stderr, "Error: %s is not a valid array dimension\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'h':
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        default:
            fprintf(stderr, "Error: Unknown option: -%c\n", opt);
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    // ----------------------------------------------------------------------
    // Get device information

    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, 0));
    printf("Running on: %s with compute capability %d.%d\n", prop.name, prop.major, prop.minor);

    // ----------------------------------------------------------------------
    
    // Create timing events
    cudaEvent_t kernelStart, kernelStop;
    cudaCheckError(cudaEventCreate(&kernelStart));
    cudaCheckError(cudaEventCreate(&kernelStop));

    printf("Allocating and initializing arrays...\n");
    // Allocate memory for arrays on host
    float alpha = 2.0f;
    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float* z = (float*)malloc(N * sizeof(float));

    // Fill host arrays A and B
    for(int i = 0; i < N; i++) {
        x[i] = get_rand_float();
        y[i] = get_rand_float();
    }

    // TODO: Allocate memory for arrays d_x, d_y, and d_z on GPU
    float* d_x, *d_y, *d_z;

    // TODO: Copy data from host input arrays to device input arrays

    // TODO: Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk;
    int blk_in_grid;

    printf("Running kernel...\n");
    cudaCheckError(cudaEventRecord(kernelStart));

    // TODO: Launch kernel using specified parameters

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaEventRecord(kernelStop));
    cudaCheckError(cudaEventSynchronize(kernelStop));

    // TODO: Copy data from device output array to host output array

    // TODO: Free all device memory

    // Verify results
    printf("Kernel successfully ran! Checking for correctness...\n");
    float tolerance = 1.0e-14;
    for(int i = 0; i < N; i++) {
        float expected = alpha * x[i] + y[i];
        if(fabs(z[i] - expected) > tolerance) {
            printf("\nError: value of z[%d] = %f instead of %f\n", i, z[i], expected);
            exit(1);
        }
    }

    // Free CPU memory
    free(x);
    free(y);
    free(z);

    // Measure elapsed time
    float kernelTime = 0; // in milliseconds
    cudaCheckError(cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop));

    // Destroy events
    cudaCheckError(cudaEventDestroy(kernelStart));
    cudaCheckError(cudaEventDestroy(kernelStop));

    printf("Correctness tests passed!\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");
    printf("Kernel ran in %.3fms\n", kernelTime);

    return 0;
}
