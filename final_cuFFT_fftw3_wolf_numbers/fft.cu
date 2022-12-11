#include <cufft.h>
#include <stdio.h>
#include <malloc.h>
#include <time.h>

// for (( i=1938; i <= 1991; i++ )); do curl "http://www.gaoran.ru/database/csa/wolf_numbers/w$i.dat" >> wolf.dat; done

/* http://www.gaoran.ru/database/csa/daily_wolf_r.html
 * Формат данных по числам Вольфа
 * 1-2	 Месяц
 * 4-5	 День
 * 7-9   Число Вольфа (весь диск, 999=нет данных)
 * 11-13 Число Вольфа (центральная зона, 999=нет данных)
 */

#define n 19722
#define NX n
#define BATCH 1


int main() {

    cufftHandle plan;
    cufftComplex *devPtr;
    cufftComplex data[NX*BATCH];
    clock_t start, end;

    cudaMalloc((void**)&devPtr, sizeof(cufftComplex)*NX*BATCH);
    if (cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return -1;
    }

    printf("n = %d\n", n);
    float wolf_day_avg[n];

    FILE *file = fopen("wolf.dat", "r");
    if (file == NULL) {
        printf("Error opening file\n");
        exit(1);
    }



    for (int i = 0; i < n; i++) {
        int day, month, wolf_disk, wolf_center;
        fscanf(file, "%d %d %d %d \n", &day, &month, &wolf_disk, &wolf_center);

        // нет данных
        if (wolf_disk == 999) {
            wolf_disk = 0;
        }
        if (wolf_center == 999) {
            wolf_center = 0;
        }

        wolf_day_avg[i] = (wolf_disk + wolf_center) / 2.0;

        if (i > n - 10) {
            printf("[%d] %d %d %d %d avg = %f\n", i, day, month, wolf_disk, wolf_center, wolf_day_avg[i]);
        }

    }

    fclose(file);


    for (int i = 0; i < NX*BATCH; i++) {
        data[i].x = wolf_day_avg[i];
        data[i].y = 0.0;
    }
    
    cudaMemcpy(devPtr, data, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyHostToDevice);

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        return -1;
    } 
        
    // CUFFT plan
    if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH)){
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return -1;
    }
    
    start = clock();

    if (cufftExecC2C(plan, devPtr, devPtr, CUFFT_FORWARD) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
        return -1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        return -1;
    } 

    end = clock();

    cudaMemcpy(data, devPtr, NX*BATCH*sizeof(cufftComplex), cudaMemcpyDeviceToHost);


    file = fopen("wolf_fft.dat", "w");
    if (file == NULL) {
        printf("Error write file\n");
        exit(1);
    }
    // data после fft
    for(int i = 0; i < n; i++) {
        fprintf(file, "%f %f\n", data[i].x, data[i].y);

        if (i < 5)
            printf("[%d] %g\t%g\n", i, data[i].x, data[i].y);

    }

    cufftDestroy(plan);
    cudaFree(data);
    cudaFree(devPtr);
    fclose(file);

    double time = (double)(end - start) / (CLOCKS_PER_SEC / 1000);
    printf("Time in ms = %f\n", time);

 return 0;
}