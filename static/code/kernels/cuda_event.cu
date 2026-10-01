/* Medir el tiempo de un kernel con eventos de CUDA.
 *
 * Un evento es una marca que se pone en la fila de un stream. El GPU anota
 * la hora en que la fila llega a esa marca, así que la diferencia entre dos
 * eventos mide el tiempo en el GPU, sin el ruido del lado del host.
 *
 * Para comparar, el mismo kernel se mide también con un cronómetro del host,
 * que incluye el lanzamiento y la sincronización.
 *
 * Compilar:  nvcc -arch=sm_75 cuda_event.cu -o cuda_event.x
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOQUE 256
#define ITER 2000

__global__ void calcular(float *c, const float *a, const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = a[idx];
    for (int i = 0; i < ITER; i++)
      v = v * 0.9999f + b[idx];
    c[idx] = v;
  }
}

double segundos() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main() {
  int n = 1 << 24;
  size_t bytes = n * sizeof(float);

  float *h_a = (float *)malloc(bytes), *h_b = (float *)malloc(bytes);
  srand(2019);
  for (int i = 0; i < n; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, bytes);
  cudaMalloc((void **)&d_b, bytes);
  cudaMalloc((void **)&d_c, bytes);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int grid = (n + BLOQUE - 1) / BLOQUE;

  // calentamiento
  calcular<<<grid, BLOQUE>>>(d_c, d_a, d_b, n);
  cudaDeviceSynchronize();

  // crear los eventos
  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);

  double t0 = segundos();
  cudaEventRecord(inicio); // marca antes del kernel
  calcular<<<grid, BLOQUE>>>(d_c, d_a, d_b, n);
  cudaEventRecord(fin); // marca después del kernel

  // el host todavía no sabe nada: el lanzamiento es asincrónico.
  // Esperamos a que el GPU llegue a la marca "fin".
  cudaEventSynchronize(fin);
  double t_host = (segundos() - t0) * 1e3;

  float t_evento;
  cudaEventElapsedTime(&t_evento, inicio, fin);

  printf("tiempo medido con eventos:     %8.3f ms\n", t_evento);
  printf("tiempo medido desde el host:   %8.3f ms\n", t_host);

  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  return 0;
}
