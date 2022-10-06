#include <cuda.h>
#include <stdio.h>


__global__ void sum(float* a, float* b, float* c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}


#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != 0) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value),  __LINE__, __FILE__) : printf("") 

int main() { 

    float *a, *b, *c;

    float *d_a, *d_b, *d_c;

    float elapsedTime;
    cudaEvent_t start, stop;

    FILE *file = fopen("./results", "w");
    if (file == NULL) {
        printf("error opening file\n");
        exit(0); 
    }

    for (int threads = 1; threads < 1024; threads <<= 1) {
        for (int N = 1 << 10; N < (1 << 23); N <<= 1) {
            int block = (N + threads - 1) / threads;
            printf("thread %d N %d block %d\n", threads, N, block);

            a = new float[block * threads];
            b = new float[block * threads];
            c = new float[block * threads];

            for(int i = 0; i < block * threads; i++)
            {
                a[i] = i;
                b[i] = i;
            }


            CUDA_CHECK_RETURN(cudaEventCreate(&start));
            CUDA_CHECK_RETURN(cudaEventCreate(&stop));

            CUDA_CHECK_RETURN(cudaMalloc((void **)&d_a, block * threads * sizeof(float)));
            CUDA_CHECK_RETURN(cudaMalloc((void **)&d_b, block * threads * sizeof(float)));
            CUDA_CHECK_RETURN(cudaMalloc((void **)&d_c, block * threads * sizeof(float)));

            CUDA_CHECK_RETURN(cudaMemcpy(d_a, a, block * threads * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(d_b, b, block * threads * sizeof(float), cudaMemcpyHostToDevice));
        
            CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
            sum <<< block, threads >>> (d_a, d_b, d_c);
            CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
            CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

            CUDA_CHECK_RETURN(cudaGetLastError());

            CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
        
            fprintf(file, "elapsedTime %g block %d thread %d\n", elapsedTime, block, threads);

            CUDA_CHECK_RETURN(cudaEventDestroy(start));
            CUDA_CHECK_RETURN(cudaEventDestroy(stop));

            CUDA_CHECK_RETURN(cudaMemcpy(c, d_c, block * threads * sizeof(float), cudaMemcpyDeviceToHost));

            free(a);
            free(b);
            free(c);

            CUDA_CHECK_RETURN(cudaFree(d_a));
            CUDA_CHECK_RETURN(cudaFree(d_b));
            CUDA_CHECK_RETURN(cudaFree(d_c));   
        }
    }
    
    fclose(file);

    return 0;
}
