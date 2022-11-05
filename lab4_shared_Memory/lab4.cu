#include <cuda.h>
#include <stdio.h>

// Инициализация матрицы
__global__ void gInitializeStorage(float* storage_d){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int N = blockDim.x*gridDim.x;

    storage_d[i+j*N]=(float)(i+j*N);
}

// Простое транспонирование
__global__ void gTranspose0(float* storage_d, float* storage_d_t){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int N = blockDim.x*gridDim.x;

    storage_d_t[j+i*N] = storage_d[i+j*N];
}

// Наивное использование shared memory (Инициализация)
__global__ void gTranspose11(float* storage_d, float* storage_d_t){
    extern __shared__ float buffer[];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int N = blockDim.x*gridDim.x;
    buffer[threadIdx.y + threadIdx.x*blockDim.y] = storage_d[i + j*N];
    __syncthreads();

    i = threadIdx.x + blockIdx.y*blockDim.x;
    j = threadIdx.y + blockIdx.x*blockDim.y;

    storage_d_t[i + j*N] = buffer[threadIdx.x + threadIdx.y*blockDim.x];
}

// Наивное использование shared memory. Транспонирование
#define SH_DIM 32
__global__ void gTranspose12(float* storage_d,float* storage_d_t){
    __shared__ float buffer_s[SH_DIM][SH_DIM];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int N = blockDim.x*gridDim.x;
    buffer_s[threadIdx.y][threadIdx.x] = storage_d[i + j*N];
    __syncthreads();
    i = threadIdx.x + blockIdx.y*blockDim.x;
    j = threadIdx.y + blockIdx.x*blockDim.y;
    storage_d_t[i + j*N] = buffer_s[threadIdx.x][threadIdx.y];
}

// Использование shared memory с разрешением конфликтов банков
__global__ void gTranspose2(float* storage_d,float* storage_d_t){
    __shared__ float buffer[SH_DIM][SH_DIM+1];
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int j=threadIdx.y+blockIdx.y*blockDim.y;
    int N=blockDim.x*gridDim.x;
    buffer[threadIdx.y][threadIdx.x]=storage_d[i+j*N];
    __syncthreads();
    i=threadIdx.x+blockIdx.y*blockDim.x;
    j=threadIdx.y+blockIdx.x*blockDim.y;
    storage_d_t[i+j*N]=buffer[threadIdx.x][threadIdx.y];
}

void output(float* a, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%g ", a[j + i * n]);
        }
        printf("\n");
    }
    printf("\n");
}

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("") 

int main(int argc, char* argv[]){
    
    if(argc < 3) {
        fprintf(stderr, "USAGE: matrix <dimension of matrix> <dimension_of_threads>\n");
        return -1;
    }

    int N = atoi(argv[1]);
    int dim_of_threads = atoi(argv[2]);
    
    if(N % dim_of_threads)
    {
        fprintf(stderr, "change dimensions\n");
        return -1;
    }
    
    int dim_of_blocks = N/dim_of_threads;
    const int max_size = 1 << 8;
    if(dim_of_blocks > max_size)
    {
        fprintf(stderr, "too many blocks\n");
        return -1;
    }
    float* d_a, *d_a1, *c;
    
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_a, N * N * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_a1, N * N * sizeof(float)));    
    c = new float[N * N];

    gInitializeStorage <<< dim3(dim_of_blocks, dim_of_blocks), dim3(dim_of_threads, dim_of_threads) >>> (d_a);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(c, d_a, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // output(c, N);
    memset(c, 0.0, N*N*sizeof(float)); 


    gTranspose0 <<< dim3(dim_of_blocks, dim_of_blocks), dim3(dim_of_threads,dim_of_threads) >>> (d_a, d_a1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(c, d_a1, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);
    memset(c, 0.0, N*N*sizeof(float)); 

    gTranspose11 <<<dim3(dim_of_blocks, dim_of_blocks), dim3(dim_of_threads,dim_of_threads), dim_of_threads*dim_of_threads*sizeof(float) >>> (d_a, d_a1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    CUDA_CHECK_RETURN(cudaMemcpy(c, d_a1, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);
    memset(c, 0.0, N*N*sizeof(float)); 

    gTranspose12 <<< dim3(dim_of_blocks, dim_of_blocks), dim3(dim_of_threads,dim_of_threads) >>> (d_a, d_a1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    CUDA_CHECK_RETURN(cudaMemcpy(c, d_a1, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);
    memset(c, 0.0, N*N*sizeof(float)); 
   
    gTranspose2 <<< dim3(dim_of_blocks, dim_of_blocks), dim3(dim_of_threads,dim_of_threads) >>> (d_a, d_a1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    CUDA_CHECK_RETURN(cudaMemcpy(c, d_a1, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);
  
    CUDA_CHECK_RETURN(cudaFree(d_a));
    CUDA_CHECK_RETURN(cudaFree(d_a1));
    free(c);

    return 0;
}