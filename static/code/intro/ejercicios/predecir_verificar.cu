// ============================================================
// Ejercicio 1: Predecir y verificar
// ============================================================
// OBJETIVO: entender la relación entre threadIdx, blockIdx,
//           blockDim y gridDim.
//
// INSTRUCCIONES:
//   1. Antes de ejecutar, escribe a mano los valores que
//      crees que tendrán A[0], A[1], ..., A[11] con la
//      configuración <<<3, 4>>> que ya viene puesta.
//   2. Compila, ejecuta y compara con tu predicción.
//   3. Cambia la configuración a <<<2, 6>>> (blocks = 2,
//      threads = 6) y repite el ejercicio.
// ============================================================

#include <stdio.h>

#define N 12 // 12 threads en ambas configuraciones

__global__ void escribir_indices(int *a) {
  // Cada thread escribe su "id":
  //   a[indice_global] = blockIdx.x * 100 + threadIdx.x
  // Multiplicamos por 100 para separar visualmente los bloques.
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  a[idx] = blockIdx.x * 100 + threadIdx.x;
}

int main(void) {
  int a[N];
  int *d_a;
  int size = N * sizeof(int);

  cudaMalloc((void **)&d_a, size);

  // Configuración de lanzamiento (cambia estos dos valores):
  //   Primera ejecución:  blocks = 3, threads = 4
  //   Segunda ejecución:  blocks = 2, threads = 6
  int blocks = 3;
  int threads = 4;
  escribir_indices<<<blocks, threads>>>(d_a);

  cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

  printf("Resultado con <<<%d, %d>>>:\n", blocks, threads);
  for (int idx = 0; idx < N; idx++)
    printf("  a[%2d] = %d\n", idx, a[idx]);

  cudaFree(d_a);
  return 0;
}
