/* Histograma con operaciones atómicas.
 *
 * Contamos cuántas veces aparece cada valor (0 a 255) en un arreglo de
 * bytes. Muchos threads quieren incrementar el MISMO contador a la vez:
 * sin atómicas habría race conditions y se perderían cuentas.
 *
 * Dos kernels:
 *   - histGlobal:     cada thread hace atomicAdd directo en memoria global;
 *   - histCompartida: cada bloque arma su histograma en memoria compartida
 *                     y al final suma sus 256 contadores al global.
 *
 * Y dos conjuntos de datos:
 *   - aleatorio: los valores se reparten entre los 256 contadores;
 *   - constante: todos los valores son iguales, así que TODOS los threads
 *                compiten por el mismo contador (contención máxima).
 *
 * Compilar:  nvcc -arch=sm_75 histograma.cu -o histograma.x
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NBINS 256
#define BLOQUE 256
#define REPS 10 // lanzamientos por medición, para promediar

__global__ void histGlobal(const unsigned char *data, unsigned int *hist,
                           int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    atomicAdd(&hist[data[i]], 1);
}

__global__ void histCompartida(const unsigned char *data, unsigned int *hist,
                               int N) {
  __shared__ unsigned int h[NBINS];

  // inicializar el histograma del bloque
  for (int b = threadIdx.x; b < NBINS; b += blockDim.x)
    h[b] = 0;
  __syncthreads();

  // atómicas en memoria compartida: solo compiten los threads del bloque
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    atomicAdd(&h[data[i]], 1);
  __syncthreads();

  // una atómica global por contador y por bloque
  for (int b = threadIdx.x; b < NBINS; b += blockDim.x)
    if (h[b] > 0)
      atomicAdd(&hist[b], h[b]);
}

// Mide el tiempo promedio (ms) de un kernel y verifica el histograma
float medir(int k, const unsigned char *d_data, unsigned int *d_hist, int N,
            int n_bloques, const unsigned int *ref) {
  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);
  float total_ms = 0.f;

  for (int r = 0; r <= REPS; r++) { // r = 0 es calentamiento, no se mide
    cudaMemset(d_hist, 0, NBINS * sizeof(unsigned int));
    cudaEventRecord(inicio);
    if (k == 0)
      histGlobal<<<n_bloques, BLOQUE>>>(d_data, d_hist, N);
    else
      histCompartida<<<n_bloques, BLOQUE>>>(d_data, d_hist, N);
    cudaEventRecord(fin);
    cudaEventSynchronize(fin);
    float ms;
    cudaEventElapsedTime(&ms, inicio, fin);
    if (r > 0)
      total_ms += ms;
  }

  // verificar contra el histograma calculado en el CPU
  unsigned int h_hist[NBINS];
  cudaMemcpy(h_hist, d_hist, NBINS * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  for (int b = 0; b < NBINS; b++) {
    if (h_hist[b] != ref[b]) {
      printf("ERROR en el contador %d: %u en vez de %u\n", b, h_hist[b],
             ref[b]);
      break;
    }
  }

  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);
  return total_ms / REPS;
}

int main() {
  int N = 1 << 24;
  unsigned char *h_data = (unsigned char *)malloc(N);
  unsigned int ref[NBINS];

  unsigned char *d_data;
  unsigned int *d_hist;
  cudaMalloc((void **)&d_data, N);
  cudaMalloc((void **)&d_hist, NBINS * sizeof(unsigned int));

  int n_sm;
  cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
  int n_bloques = 32 * n_sm;

  const char *nombre_datos[] = {"aleatorio", "constante"};
  const char *nombre_kernel[] = {"global", "compartida"};

  printf("N = %d bytes, %d bloques de %d threads, promedio de %d\n\n", N,
         n_bloques, BLOQUE, REPS);
  printf("datos       atómicas en   tiempo (ms)\n");

  srand(2019);
  for (int d = 0; d < 2; d++) {
    // generar los datos y el histograma de referencia en el CPU
    for (int i = 0; i < N; i++)
      h_data[i] = (d == 0) ? (unsigned char)(rand() & 0xFF) : 7;
    memset(ref, 0, sizeof(ref));
    for (int i = 0; i < N; i++)
      ref[h_data[i]]++;
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);

    for (int k = 0; k < 2; k++) {
      float ms = medir(k, d_data, d_hist, N, n_bloques, ref);
      printf("%-10s  %-11s  %11.3f\n", nombre_datos[d], nombre_kernel[k], ms);
    }
  }

  cudaFree(d_data);
  cudaFree(d_hist);
  free(h_data);
  return 0;
}
