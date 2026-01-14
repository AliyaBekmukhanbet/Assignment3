%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000        // Размер обрабатываемого массива
#define BLOCK_SIZE 256   // Количество потоков в одном CUDA-блоке

// CUDA-ядро: каждый поток обрабатывает один элемент массива в global memory
__global__ void multiply_global(float *d_array, float scalar, int n) {
    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Защита от выхода за границы массива
    if (idx < n) {
        d_array[idx] *= scalar;
    }
}

int main() {
    float *h_array = (float*)malloc(N * sizeof(float)); // Массив в памяти CPU
    float *d_array;                                    // Массив в памяти GPU

    cudaMalloc(&d_array, N * sizeof(float));            // Выделение памяти на GPU

    // Инициализация данных на CPU
    for (int i = 0; i < N; i++)
        h_array[i] = (float)i;

    // Копирование данных CPU → GPU
    cudaMemcpy(d_array, h_array, N * sizeof(float),
               cudaMemcpyHostToDevice);

    // CUDA-события для измерения времени выполнения ядра
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Конфигурация запуска ядра
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    multiply_global<<<grid_size, BLOCK_SIZE>>>(d_array, 2.0f, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Получение времени выполнения ядра
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Копирование результата обратно (для проверки)
    cudaMemcpy(h_array, d_array, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("Время выполнения (global memory): %f мс\n", milliseconds);

    // Освобождение ресурсов
    free(h_array);
    cudaFree(d_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
