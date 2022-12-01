#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>



__global__ void mult(float *a, float *b, float *c, int row_num, int col_num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    c[i] = 0;

    for (j = 0; j < col_num; j++) {
        c[i] += a[j * row_num + i] * b[j];
    }
}

void cuda_raw(float *matrix, float *vector, float *vector_res, int row_num, int col_num) {

    float *a_device, *x_device, *y_device;

    cudaMalloc((void **)&a_device, row_num * col_num * sizeof(*matrix));
    cudaMalloc((void **)&x_device, col_num * sizeof(*vector));
    cudaMalloc((void **)&y_device, row_num * sizeof(*vector_res));

    cudaMemcpy(a_device, matrix, row_num * col_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_device, vector, col_num * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(row_num);
    dim3 grid(1);

    mult <<< grid, threads >>> (a_device, x_device, y_device, row_num, col_num);

    cudaDeviceSynchronize();

    cudaMemcpy(vector_res, y_device, row_num * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_device);
    cudaFree(x_device);
    cudaFree(y_device);
}

void cuda_thrust(float *matrix, float *vector, float *vector_res, int row_num, int col_num) {
    thrust::device_vector<float> a_device(matrix, matrix + (row_num * col_num));
    thrust::device_vector<float> b_device(col_num);
    thrust::device_vector<float> x_device(vector, vector + col_num);
    thrust::device_vector<float> y_device(col_num);

    float res = 0.0;

    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++)
            b_device[j] = a_device[j * row_num + i];

        thrust::transform(
                b_device.begin(),
                b_device.end(),
                x_device.begin(),
                y_device.begin(),
                thrust::multiplies<float>());

        for (int j = 0; j < col_num; j++)
            res = thrust::reduce(y_device.begin(), y_device.end());
            vector_res[i] = res;
        res = 0.0;
    }
}

// #define DEBUG_PRINT

int main(int argc, char *argv[]) {
    
    srand(time(NULL));

    if(argc < 3) {
        fprintf(stderr, "Usage: <row_num> <col_num>\n");
        return -1;
    }

    int row_num = atoi(argv[1]);
    int col_num = atoi(argv[2]);

    clock_t start, end;

    float *matrix = (float *)malloc(row_num * col_num * sizeof(float));
    float *vector = (float *)malloc(col_num * sizeof(float));
    float *vector_res = (float *)malloc(row_num * sizeof(float));

    // Заполним матрицу
    for (int j = 0; j < col_num; j++) {
        for (int i = 0; i < row_num; i++) {
            matrix[j * row_num + i] = rand() % 100 - 25;
        }
    }

    #ifdef DEBUG_PRINT
        printf("Matrix:\n");
        for(int i = 0; i < row_num; i++) {
            for(int j = 0; j < col_num; j++) {
                printf("%g\t", matrix[j * row_num + i]);
            }
            printf("\n");
        }

        for (int i = 0; i < col_num; i++) {
            vector[i] = rand() % 100 - 25;
        }

        printf("Vector:\n");
        for (int i = 0; i < col_num; i++) {
            printf("%g\t", vector[i]);
        }
        printf("\n");
    #endif

    // расскоментировать одну из функций
    start = clock();
    cuda_raw(matrix, vector, vector_res, row_num, col_num);
    // cuda_thrust(matrix, vector, vector_res, row_num, col_num);
    end = clock();
  
    #ifdef DEBUG_PRINT
        printf("Result:\n");
        for (int i = 0; i < row_num; i++) {
            printf("%g\t", vector_res[i]);
        }
        printf("\n");
    #endif

    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time in seconds = %f\n", time);

    free(matrix);
    free(vector);
    free(vector_res);
}
