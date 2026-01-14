%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define WIDTH 1000          // Ширина матрицы (row-major)
#define HEIGHT (N / WIDTH)  // Высота матрицы
#define BLOCK_SIZE 256

// Ядро с коалесцированным доступом к памяти:
// соседние потоки обращаются к соседним элементам памяти
__global__ void coalesced_copy(float *d_out, float *d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx];
    }
}

// Ядро с некоалесцированным доступом:
// имитация доступа по столбцам в row-major матрице
// потоки одного варпа обращаются к памяти с большим шагом (stride)
__global__ void non_coalesced_copy(float *d_out, float *d_in,
                                   int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int row = idx / width;
        int col = idx % width;

        // Столбцовый доступ → плохая коалесценция → больше транзакций памяти
        d_out[idx] = d_in[col * HEIGHT + row];
    }
}

// Функция измерения времени выполнения ядра с помощью CUDA Events
// Используется для сравнения разных паттернов доступа к памяти
void measure_time(const char* label,
                  void (*kernel)(float*, float*, int, int),
                  float *d_out, float *d_in,
                  int param, int n,
                  int grid_size, int block_size) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Выбор соответствующего ядра
    if (strcmp(label, "Coalesced") == 0) {
        ((void (*)(float*, float*, int))kernel)
            <<<grid_size, block_size>>>(d_out, d_in, n);
    } else {
        kernel<<<grid_size, block_size>>>(d_out, d_in, param, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s время выполнения: %f мс\n", label, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {

    // Память на CPU и GPU
    float *h_in  = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    float *d_in, *d_out;

    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // Инициализация входных данных
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
    }
    cudaMemcpy(d_in, h_in, N * sizeof(float),
               cudaMemcpyHostToDevice);

    // Конфигурация сетки
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Тест коалесцированного доступа
    cudaMemset(d_out, 0, N * sizeof(float));
    measure_time("Coalesced",
                 (void (*)(float*, float*, int, int))coalesced_copy,
                 d_out, d_in, 0, N,
                 grid_size, BLOCK_SIZE);

    // Тест некоалесцированного доступа
    cudaMemset(d_out, 0, N * sizeof(float));
    measure_time("Non-coalesced",
                 non_coalesced_copy,
                 d_out, d_in, WIDTH, N,
                 grid_size, BLOCK_SIZE);

    // Освобождение ресурсов
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
