// ============================================================
// Ejercicio: la multiplicación matriz-vector en el roofline
// ============================================================
//
// Producto y = A x, con A de N x N y un thread por cada fila de A,
// es decir por cada elemento de y.
//
// Para las otras partes del ejercicio, cambiar N acá abajo y recompilar.

#include <stdio.h>
#include <stdlib.h>

#define N 1024

__global__ void matvec(const float *A, const float *x, float *y, int n) {
  int fila = threadIdx.x + blockIdx.x * blockDim.x;

  if (fila < n) {
    float suma = 0.0f;
    for (int k = 0; k < n; k++)
      suma += A[fila * n + k] * x[k];
    y[fila] = suma;
  }
}

int main(void) {
  size_t sizeA = (size_t)N * N * sizeof(float);
  size_t sizeV = N * sizeof(float);

  float *A = (float *)malloc(sizeA);
  float *x = (float *)malloc(sizeV);
  float *y = (float *)malloc(sizeV);

  // (i + 2k) % 7 no es simétrico: si se equivocan los índices, el resultado cambia
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    for (int k = 0; k < N; k++)
      A[(size_t)i * N + k] = (i + 2 * k) % 7;
  }

  float *d_A, *d_x, *d_y;
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_x, sizeV);
  cudaMalloc(&d_y, sizeV);
  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, sizeV, cudaMemcpyHostToDevice);

  int hilos = 256;
  int bloques = (N + hilos - 1) / hilos;
  matvec<<<bloques, hilos>>>(d_A, d_x, d_y, N);

  cudaMemcpy(y, d_y, sizeV, cudaMemcpyDeviceToHost);

  // Verificación: una fila, contra el mismo cálculo hecho en el host
  int f = 1;
  float esperado = 0.0f;
  for (int k = 0; k < N; k++)
    esperado += A[(size_t)f * N + k] * x[k];
  printf("N = %d\n", N);
  printf("y[%d] = %.1f  (esperado %.1f)\n", f, y[f], esperado);

  free(A);
  free(x);
  free(y);
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}
