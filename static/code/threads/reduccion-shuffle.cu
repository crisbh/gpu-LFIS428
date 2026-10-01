/* Reducción paralela "moderna": la suma de todos los elementos de un arreglo
 * en UN solo kernel, combinando lo que vimos en esta clase:
 *
 *   1. grid-stride loop: cada thread suma primero varios elementos;
 *   2. warp primitives: __shfl_down_sync reduce los 32 valores de un warp
 *      intercambiando registros, sin memoria compartida ni __syncthreads();
 *   3. memoria compartida: un valor por warp, para reducir el bloque;
 *   4. operación atómica: un atomicAdd por bloque suma al total global.
 *
 * Compilar:  nvcc -arch=sm_75 reduccion-shuffle.cu -o reduccion-shuffle.x
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOQUE 256         // threads por bloque (múltiplo de 32)
#define MASCARA 0xffffffff // los 32 threads del warp participan

// Suma los valores de los 32 threads de un warp. Al final, el thread 0 del
// warp (lane 0) tiene la suma completa.
__device__ float sumaWarp(float v) {
  for (int d = 16; d > 0; d >>= 1)
    v += __shfl_down_sync(MASCARA, v, d);
  return v;
}

__global__ void reduccionShuffle(const float *data, float *total, int N) {
  // 1. grid-stride: cada thread acumula varios elementos en un registro
  float suma = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    suma += data[i];

  // 2. reducir dentro de cada warp
  suma = sumaWarp(suma);

  // 3. el lane 0 de cada warp deja su resultado en memoria compartida
  __shared__ float parcial[BLOQUE / 32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  if (lane == 0)
    parcial[warp] = suma;
  __syncthreads();

  // el primer warp reduce los resultados de todos los warps del bloque
  if (warp == 0) {
    suma = (lane < BLOQUE / 32) ? parcial[lane] : 0.f;
    suma = sumaWarp(suma);

    // 4. una sola operación atómica por bloque
    if (lane == 0)
      atomicAdd(total, suma);
  }
}

int main() {
  int N = 1 << 24;
  size_t bytes = N * sizeof(float);

  // datos en el host: números aleatorios pequeños
  float *h_data = (float *)malloc(bytes);
  srand(2019);
  for (int i = 0; i < N; i++)
    h_data[i] = (float)(rand() & 0xFF) / (float)RAND_MAX;

  // referencia en el CPU, acumulando en double para tener un valor exacto
  double referencia = 0.0;
  for (int i = 0; i < N; i++)
    referencia += h_data[i];

  float *d_data, *d_total;
  cudaMalloc((void **)&d_data, bytes);
  cudaMalloc((void **)&d_total, sizeof(float));
  cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

  // Con grid-stride no hace falta un thread por elemento: basta con llenar
  // el GPU. 32 bloques de 256 threads por SM es más que suficiente.
  int n_sm;
  cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
  int n_bloques = 32 * n_sm;

  // medir con eventos de CUDA (los veremos en la clase de streams)
  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);

  cudaMemset(d_total, 0, sizeof(float));
  cudaEventRecord(inicio);
  reduccionShuffle<<<n_bloques, BLOQUE>>>(d_data, d_total, N);
  cudaEventRecord(fin);
  cudaEventSynchronize(fin);

  float ms, total;
  cudaEventElapsedTime(&ms, inicio, fin);
  cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost);

  printf("N = %d, %d bloques de %d threads (%d SMs)\n", N, n_bloques, BLOQUE,
         n_sm);
  printf("CPU (double): %f\n", referencia);
  printf("GPU (float):  %f   error relativo %.2e\n", total,
         fabs(total - referencia) / referencia);
  printf("tiempo del kernel: %.3f ms (%.1f GB/s)\n", ms,
         bytes / 1e9 / (ms / 1e3));

  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);
  cudaFree(d_data);
  cudaFree(d_total);
  free(h_data);
  return 0;
}
