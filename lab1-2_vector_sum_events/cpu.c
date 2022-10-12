#include <stdio.h> 
#include <stdint.h> 
#include <malloc.h> 
#include <pthread.h> 
#include <sys/time.h> 
 
typedef struct thread_info { 
         
    pthread_t thread_id; 
    int threadIdx; 
    int blockIdx; 
    int blockDim; 
    double execution_time; 
 
} thread_info_t; 
 
float *a, *b, *c; 
 
 
void sum(int threadIdx, int blockIdx, int blockDim) { 
 
    int start = threadIdx + blockIdx * blockDim; 
    int end = threadIdx + blockIdx * blockDim + blockDim; 
    for (int index = start; index < end; index++) { 
        c[index] = a[index] + b[index]; 
        // printf("[%d] thread_id %lx a %f b %f c %f\n", index, pthread_self(), a[index], b[index], c[index]); 
    } 
 
} 
void *thread_routine(void *arg){ 
    thread_info_t *tinfo = (thread_info_t *)arg; 
 
    struct timeval tv1, tv2; 
 
    gettimeofday(&tv1, NULL); 
    sum(tinfo->threadIdx, tinfo->blockIdx, tinfo->blockDim); 
    gettimeofday(&tv2, NULL); 
 
    tinfo->execution_time = (tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec)/1000000.0 ) * 1000.0; // ms
 
} 
 
 
 
int main(void) { 
 
 
    int threads = 3;// 1024; 
    int N = 1 << 11;//1 << 10; 
 
    FILE *file = fopen("./cpu_result", "w"); 
    for (int N = 1 << 10; N < (1 << 23) + 1; N <<= 1) { 
        for (int threads = 1; threads <= 1024; threads <<= 1) { 
 
            thread_info_t thread_info[threads]; 
 
            int block = (N + threads - 1) / threads; 
            printf("thread %d N %d block %d\n", threads, N, block); 
 
            a = (float *)malloc(block * threads * sizeof(float)); 
            b = (float *)malloc(block * threads * sizeof(float)); 
            c = (float *)malloc(block * threads * sizeof(float)); 
 
            for(int i = 0; i < block * threads; i++) { 
                a[i] = i; 
                b[i] = i; 
                // printf("[%d] a %f b %f\n", i, a[i], b[i]); 
 
            } 
 
            for (int i = 0; i < threads; i++) { 
                thread_info[i].threadIdx = 0; 
                thread_info[i].blockIdx = i; 
                thread_info[i].blockDim = block; // size of block 
                thread_info[i].execution_time = 0.0; 
                pthread_create(&thread_info[i].thread_id, NULL, thread_routine, &thread_info[i]); 
            } 
 
            for (int i = 0; i < threads; i++) { 
                pthread_join(thread_info[i].thread_id, NULL); 
            } 
 
            float elapsed_time = 0.0; 
            for (int i = 0; i < threads; i++) { 
                elapsed_time += thread_info[i].execution_time; 
            } 
 
            // for (int i = N-5; i < N; i++) { 
            //     fprintf(stdout, "%f\n", c[i]); 
            // } 
 
            fprintf(file, "elapsedTime %g block %d thread %d\n", elapsed_time, block, threads); 
 
 
            free(a); 
            free(b); 
            free(c); 
 
        } 
    } 
    fclose(file); 
 
    return 0; 
}