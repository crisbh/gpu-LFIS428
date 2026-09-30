/* Memoria constante contra memoria global.
 *
 * Cada thread suma los 360 ángulos de una tabla. Hay cuatro versiones:
 *   - la tabla está en memoria global o en memoria constante.
 *   - el acceso es uniforme (todos los threads del warp leen el MISMO ángulo
 *     en cada iteración) o disperso (cada thread lee un ángulo distinto).
 *
 * Compilar:  nvcc -arch=sm_75 memoria_constante.cu -o memoria_constante.x
 * (necesita common.h en el mismo directorio)
 */

#include "common.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NANG 360   // número de ángulos en la tabla
#define BLOQUE 256 // threads por bloque
#define REPS 20    // lanzamientos por medición, para promediar

// declarar memoria constante: global scope, fuera de cualquier kernel
__constant__ float c_angulo[NANG];

// La suma se acumula en una variable local (un registro) y se escribe una
// sola vez al final, igual en los cuatro kernels: así lo único que cambia
// entre ellos es de dónde se leen los ángulos.

// Tabla en memoria global, todos los threads leen el mismo ángulo
__global__ void globalUniforme(float *darray, const float *d_angulo, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  float suma = 0.f;
#pragma unroll 10
  for (int l = 0; l < NANG; l++)
    suma += d_angulo[l];
  darray[idx] = suma;
}

// Tabla en memoria constante, todos los threads leen el mismo ángulo
__global__ void constanteUniforme(float *darray, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  float suma = 0.f;
#pragma unroll 10
  for (int l = 0; l < NANG; l++)
    suma += c_angulo[l];
  darray[idx] = suma;
}

// Tabla en memoria global, cada thread del warp lee un ángulo distinto
__global__ void globalDisperso(float *darray, const float *d_angulo, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  float suma = 0.f;
#pragma unroll 10
  for (int l = 0; l < NANG; l++)
    suma += d_angulo[(l + threadIdx.x) % NANG];
  darray[idx] = suma;
}

// Tabla en memoria constante, cada thread del warp lee un ángulo distinto
__global__ void constanteDisperso(float *darray, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  float suma = 0.f;
#pragma unroll 10
  for (int l = 0; l < NANG; l++)
    suma += c_angulo[(l + threadIdx.x) % NANG];
  darray[idx] = suma;
}

// Lanza el kernel número k (0 a 3, en el orden de arriba)
void lanzar(int k, float *darray, float *d_angulo, int N) {
  int bloques = (N + BLOQUE - 1) / BLOQUE;
  switch (k) {
  case 0:
    globalUniforme<<<bloques, BLOQUE>>>(darray, d_angulo, N);
    break;
  case 1:
    constanteUniforme<<<bloques, BLOQUE>>>(darray, N);
    break;
  case 2:
    globalDisperso<<<bloques, BLOQUE>>>(darray, d_angulo, N);
    break;
  case 3:
    constanteDisperso<<<bloques, BLOQUE>>>(darray, N);
    break;
  }
}

// Mide el tiempo promedio (en ms) del kernel k y verifica el resultado:
// todos los threads suman los mismos 360 ángulos, así que todos deben dar
// "esperado" (salvo redondeo: en el acceso disperso el orden cambia).
double medir(int k, float *darray, float *d_angulo, float *h_array, int N,
             double esperado) {
  // calentamiento: el primer lanzamiento incluye costos de inicialización
  lanzar(k, darray, d_angulo, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemset(darray, 0, sizeof(float) * N));

  double comienzo = segundos();
  for (int r = 0; r < REPS; r++)
    lanzar(k, darray, d_angulo, N);
  CHECK(cudaDeviceSynchronize());
  double tiempo = (segundos() - comienzo) / REPS * 1e3;

  CHECK(cudaMemcpy(h_array, darray, sizeof(float) * N,
                   cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    if (fabs(h_array[i] - esperado) > 0.1) {
      printf("ERROR en el elemento %d: %f en vez de %f\n", i, h_array[i],
             esperado);
      break;
    }
  }

  return tiempo;
}

int main(int argc, char **argv) {
  int N = 1 << 20;
  float h_angulo[NANG];
  float *darray, *d_angulo;
  float *h_array = (float *)malloc(sizeof(float) * N);

  // inicializar la tabla de ángulos en el host (en radianes)
  double esperado = 0.0;
  for (int l = 0; l < NANG; l++) {
    h_angulo[l] = acos(-1.0f) * l / 180.0f;
    esperado += h_angulo[l];
  }

  // asignar memoria en el device
  CHECK(cudaMalloc((void **)&darray, sizeof(float) * N));
  CHECK(cudaMalloc((void **)&d_angulo, sizeof(float) * NANG));

  // copiar la tabla a la memoria global y a la memoria constante
  CHECK(cudaMemcpy(d_angulo, h_angulo, sizeof(float) * NANG,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_angulo, h_angulo, sizeof(float) * NANG));

  const char *memoria[] = {"global", "constante", "global", "constante"};
  const char *acceso[] = {"uniforme", "uniforme", "disperso", "disperso"};

  printf("N = %d threads, bloques de %d, promedio de %d lanzamientos\n\n", N,
         BLOQUE, REPS);
  printf("memoria     acceso     tiempo (ms)\n");
  for (int k = 0; k < 4; k++) {
    double t = medir(k, darray, d_angulo, h_array, N, esperado);
    printf("%-10s  %-9s  %11.3f\n", memoria[k], acceso[k], t);
  }

  // liberar memoria del device y del host
  CHECK(cudaFree(darray));
  CHECK(cudaFree(d_angulo));
  free(h_array);

  return 0;
}
