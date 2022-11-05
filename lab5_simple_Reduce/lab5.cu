#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("") 

__global__ void reduce0(double *input, double *output) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = input[i];

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
            // printf("[%d] %lf\n", tid, sdata[tid]);
        }

        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

double sumCpu(double *c, int size) {
    double sum;
    for (unsigned int i = 0; i < size; i++)
         sum+= c[i];
    return sum;
}

int main(void) {
    
    srand(time(NULL));

    struct timeval t1, t2;
    float elapsedTime;
    cudaEvent_t start, stop;

    double *h_in;
    double *h_out;
    double *d_in;
    double *d_out;

    int numThreadsPerBlock = 1024;
    long numInputElements = 2<<20; // 2<<20 .. 2 << 25
    int numOutputElements = (numInputElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

    fprintf(stdout, "N %d block %d numThreadsPerBlock %d\n", numInputElements, numOutputElements, numThreadsPerBlock);


    h_in = (double *)malloc(numInputElements * sizeof(double));
    h_out = (double *)malloc(numOutputElements * sizeof(double));


    for (int i = 0; i < numInputElements; i++) {
        h_in[i] = 1.0;
    }

    const dim3 blockSize(numThreadsPerBlock, 1, 1);
    const dim3 gridSize(numOutputElements, 1, 1);

    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_in, numInputElements * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_out, numOutputElements * sizeof(double)));

    CUDA_CHECK_RETURN(cudaMemcpy(d_in, h_in, numInputElements * sizeof(double), cudaMemcpyHostToDevice));
    

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

    reduce0 <<< gridSize, blockSize, numThreadsPerBlock*sizeof(double) >>> (d_in, d_out);
    
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));

    CUDA_CHECK_RETURN(cudaEventDestroy(start));
    CUDA_CHECK_RETURN(cudaEventDestroy(stop));    
    
    CUDA_CHECK_RETURN(cudaMemcpy(h_out, d_out, numOutputElements * sizeof(double), cudaMemcpyDeviceToHost));

    double sumGPU = 0.0;

    for (int i = 0; i < numOutputElements; i++) {
        sumGPU += h_out[i];
        // printf("%lf\n", h_out[i]);
    }


    printf("GPU Result: %lf; elapsed time %g ms\n", sumGPU, elapsedTime);

    double sum_CPU = 0.0;

    gettimeofday(&t1, NULL);
    sum_CPU = sumCpu(h_in, numInputElements);
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0)*1000.0;

    printf("CPU Result: %lf; elapsed time %g ms\n", sum_CPU, elapsedTime);

    CUDA_CHECK_RETURN(cudaFree(d_in));
    CUDA_CHECK_RETURN(cudaFree(d_out));


    free(h_in);
    free(h_out);

    return 0;
}
