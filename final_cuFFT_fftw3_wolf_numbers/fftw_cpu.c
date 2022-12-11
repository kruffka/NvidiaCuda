#include "stdio.h"

#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
    fftw_complex *in, *out;
    fftw_plan p;
    clock_t start, end;

    int N = 19722;

    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    float wolf_day_avg[N];

    FILE *file = fopen("wolf.dat", "r");
    if (file == NULL) {
        printf("Error opening file\n");
        exit(1);
    }


    for (int i = 0; i < N; i++) {
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

        if (i > N - 10) {
            printf("[%d] %d %d %d %d avg = %f\n", i, day, month, wolf_disk, wolf_center, wolf_day_avg[i]);
        }

    }

    for (int i = 0; i < N; i++) {
        in[i][0] = wolf_day_avg[i];
        in[i][1] = 0.0;
    }
    

    fclose(file);

    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    start = clock();

    fftw_execute(p);

    end = clock();

    // data после fft
    for(int i = 0; i < N; i++) {
        fprintf(file, "%f %f\n", out[i][0], out[i][1]);

        if (i < 5)
            printf("[%d] %g\t%g\n", i, out[i][0], out[i][1]);

    }

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);

    double time = (double)(end - start) / (CLOCKS_PER_SEC / 1000);
    printf("Time in ms = %f\n", time);

    return 0;
}