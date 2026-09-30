/* Memoria paginable contra memoria pinned.
 *
 * El mismo buffer de 64 MB se asigna dos veces en el host: como memoria
 * paginable (malloc) y como memoria pinned (cudaMallocHost). Para cada uno
 * se mide:
 *   - cuánto cuesta asignarlo e inicializarlo;
 *   - la velocidad de las copias host -> device y device -> host.
 *
 * Compilar:  nvcc -arch=sm_75 memoriaPinned.cu -o memoriaPinned.x
 * (necesita common.h en el mismo directorio)
 */

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define REPS 10 // copias por medición, para promediar

// Inicializar el buffer en el host. Con malloc, este primer acceso es el que
// realmente asigna las páginas: malloc solo reserva direcciones virtuales.
void inicializar(float *h, int n) {
  for (int i = 0; i < n; i++)
    h[i] = 0.5f;
}

// Mide la velocidad (en GB/s) de las copias host -> device y device -> host
void medir(float *h, float *d, size_t bytes, double *htod, double *dtoh) {
  // calentamiento: una copia en cada sentido, sin medir
  CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());

  // Sincronizar después de cada copia: desde memoria paginable, cudaMemcpy
  // host -> device puede volver antes de que el DMA termine.
  double comienzo = segundos();
  for (int r = 0; r < REPS; r++) {
    CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
  }
  *htod = (double)bytes * REPS / 1e9 / (segundos() - comienzo);

  comienzo = segundos();
  for (int r = 0; r < REPS; r++) {
    CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
  }
  *dtoh = (double)bytes * REPS / 1e9 / (segundos() - comienzo);
}

// Los datos deben volver intactos del GPU
void verificar(const char *nombre, float *h, int n) {
  for (int i = 0; i < n; i++) {
    if (h[i] != 0.5f) {
      printf("ERROR en %s, elemento %d: %f\n", nombre, i, h[i]);
      return;
    }
  }
}

int main() {
  int N = 1 << 24;
  size_t bytes = N * sizeof(float);
  double htod, dtoh;

  // Crear el contexto de CUDA antes de medir nada
  CHECK(cudaFree(0));

  // memoria en el device, la misma para las dos pruebas
  float *d_a;
  CHECK(cudaMalloc((void **)&d_a, bytes));

  printf("%zu MB por copia, promedio de %d copias\n\n", bytes >> 20, REPS);
  printf("memoria     asignar+iniciar (ms)   HtoD (GB/s)   DtoH (GB/s)\n");

  // 1. Memoria paginable
  double comienzo = segundos();
  float *h_pag = (float *)malloc(bytes);
  inicializar(h_pag, N);
  double t_pag = (segundos() - comienzo) * 1e3;

  medir(h_pag, d_a, bytes, &htod, &dtoh);
  verificar("paginable", h_pag, N);
  printf("paginable   %20.3f   %11.2f   %11.2f\n", t_pag, htod, dtoh);
  free(h_pag);

  // 2. Memoria pinned: el sistema operativo tiene que fijar cada página
  comienzo = segundos();
  float *h_pin;
  CHECK(cudaMallocHost((void **)&h_pin, bytes));
  inicializar(h_pin, N);
  double t_pin = (segundos() - comienzo) * 1e3;

  medir(h_pin, d_a, bytes, &htod, &dtoh);
  verificar("pinned", h_pin, N);
  printf("pinned      %20.3f   %11.2f   %11.2f\n", t_pin, htod, dtoh);
  CHECK(cudaFreeHost(h_pin));

  // liberar memoria del device
  CHECK(cudaFree(d_a));

  return EXIT_SUCCESS;
}
