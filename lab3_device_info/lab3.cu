#include <cuda.h>
#include <stdio.h>

__global__ void gTest(float* a)
{
    a[threadIdx.x + blockDim.x * blockIdx.x] = (float)((threadIdx.x + blockDim.x * blockIdx.x) * 2);
}

int main() {
    int m, n, k;
    scanf("%d %d %d", &m, &n, &k);

    float* mas = new float[m];
    float* da;

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
    
   
    printf("Occupancy: %g\n", (float)(k * 8) / (float)((deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize) * deviceProp.warpSize));

    cudaMalloc((void**)&da, m * sizeof(float));
    gTest <<< n, k >>> (da);
    cudaDeviceSynchronize();
    cudaMemcpy(mas, da, m * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = m - 4; i < m; i++)
    {
        printf("%g\n", mas[i]);
    }
    free(mas);
    cudaFree(da);
 
    return 0;
}