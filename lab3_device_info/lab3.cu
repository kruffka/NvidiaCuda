#include <cuda.h>
#include <stdio.h>

__global__ void gTest(float* a)
{
    a[threadIdx.x + blockDim.x * blockIdx.x] = (float)((threadIdx.x + blockDim.x * blockIdx.x) * 2);
}

int main() {
    int N = 1 << 19, blocks, threads = 64;\

    blocks = (N + threads - 1)/threads;

    float* a = new float[N];
    float* d_a;

    int dev;
    cudaSetDevice(dev); 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    
    printf(" Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
    printf(" Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf(" Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf(" Warp size: %d\n", deviceProp.warpSize);
    printf(" Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    
   
    printf("Occupancy: %g\n", (float)(threads * 8) / (float)((deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize) * deviceProp.warpSize));

    cudaMalloc((void**)&d_a, N * sizeof(float));
    gTest <<< blocks, threads >>> (d_a);
    cudaDeviceSynchronize();
    cudaMemcpy(a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = N - 4; i < N; i++)
    {
        printf("%g\n", a[i]);
    }
    free(a);
    cudaFree(d_a);
 
    return 0;
}