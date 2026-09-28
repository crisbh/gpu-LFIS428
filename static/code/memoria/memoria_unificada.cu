/* Memoria unificada: cómo y cuándo se mueven los datos.
 *
 * Dos arreglos x, y en memoria unificada (cudaMallocManaged), sin ningún
 * cudaMemcpy. El mismo kernel (y = x + y) se mide cuatro veces:
 *   1. primer acceso: los datos están en el host y migran por fallos de página;
 *   2. los datos ya están en el GPU: solo se mide el kernel;
 *   3. cudaMemPrefetchAsync: se mueven los datos al GPU de una vez, antes;
 *   4. el kernel después del prefetch.
 *
 * Compilar:  nvcc -arch=sm_75 memoria_unificada.cu -o memoria_unificada.x
 * (necesita common.h en el mismo directorio)
 */

#include "common.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOQUE 256

// Kernel que suma los elementos de dos arreglos (grid-stride loop)
__global__ void suma(int n, float *x, float *y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride)
    y[i] = x[i] + y[i];
}

// Inicializar en el host: si las páginas estaban en el GPU, este acceso
// las trae de vuelta al CPU
void inicializar(float *x, float *y, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

// Lanza el kernel y devuelve cuánto tardó, en ms
double medirSuma(int n, float *x, float *y) {
  int grid = (n + BLOQUE - 1) / BLOQUE;
  double comienzo = segundos();
  suma<<<grid, BLOQUE>>>(n, x, y);
  CHECK(cudaDeviceSynchronize());
  return (segundos() - comienzo) * 1e3;
}

// Mover un arreglo al GPU antes de usarlo.
// La firma de cudaMemPrefetchAsync cambió en CUDA 13.
void prefetch(float *ptr, size_t bytes, int dispositivo) {
#if CUDART_VERSION >= 13000
  cudaMemLocation destino;
  destino.type = cudaMemLocationTypeDevice;
  destino.id = dispositivo;
  CHECK(cudaMemPrefetchAsync(ptr, bytes, destino, 0));
#else
  CHECK(cudaMemPrefetchAsync(ptr, bytes, dispositivo));
#endif
}

int main(void) {
  int N = 1 << 24;
  size_t bytes = N * sizeof(float);
  float *x, *y;
  int dispositivo = 0;

  // Crear el contexto de CUDA antes de medir nada
  CHECK(cudaFree(0));

  // ¿Soporta el GPU la migración de páginas a pedido?
  int aPedido = 0;
  CHECK(cudaDeviceGetAttribute(&aPedido, cudaDevAttrConcurrentManagedAccess,
                               dispositivo));

  printf("N = %d floats por arreglo (%zu MB), dos arreglos\n\n", N,
         bytes >> 20);

  // Asignar memoria unificada: el mismo puntero sirve en el host y el device
  CHECK(cudaMallocManaged(&x, bytes));
  CHECK(cudaMallocManaged(&y, bytes));

  // 1. Datos recién inicializados en el host: el kernel provoca fallos de
  //    página y los datos cruzan el PCIe mientras el kernel corre
  inicializar(x, y, N);
  double t1 = medirSuma(N, x, y);
  printf("1. kernel, primer acceso (con fallos de página): %9.3f ms\n", t1);

  // 2. Mismo kernel, con los datos ya en el GPU
  double t2 = medirSuma(N, x, y);
  printf("2. kernel, datos ya en el GPU:                    %9.3f ms\n", t2);

  // 3. y 4. Traer los datos de vuelta al host, moverlos al GPU con
  //    cudaMemPrefetchAsync y lanzar el kernel después
  inicializar(x, y, N);
  if (aPedido) {
    double comienzo = segundos();
    prefetch(x, bytes, dispositivo);
    prefetch(y, bytes, dispositivo);
    CHECK(cudaDeviceSynchronize());
    double t3 = (segundos() - comienzo) * 1e3;
    printf("3. cudaMemPrefetchAsync de x e y:                 %9.3f ms"
           "  (%.1f GB/s)\n",
           t3, 2.0 * bytes / 1e9 / (t3 / 1e3));
  } else {
    printf("3. este GPU no soporta migración a pedido: sin prefetch\n");
  }
  double t4 = medirSuma(N, x, y);
  printf("4. kernel, después del prefetch:                  %9.3f ms\n", t4);

  // Verificar en el host: tras la segunda inicialización hubo un solo
  // lanzamiento, así que y = 1 + 2 = 3 en todos los elementos
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  printf("\nerror máximo: %f\n", maxError);

  // La memoria unificada se libera con cudaFree
  CHECK(cudaFree(x));
  CHECK(cudaFree(y));

  return 0;
}
