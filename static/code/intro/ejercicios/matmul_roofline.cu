// ============================================================
// Ejercicio: la multiplicación de matrices en el roofline
// ============================================================
//
// Multiplicación 'ingenua' C = A * B, con matrices de N x N y un thread por
// cada elemento de C. Cada thread recorre una fila de A y una columna de B.
//
// Pero el tráfico MÍNIMO indispensable es mucho menor: basta leer A y B una
// vez y escribir C una vez, o sea 12*N*N bytes en total, lo que daría
// AI ideal = 2N^3 / (12 N^2) = N/6
// que para N = 1024 son 170 FLOP/byte.

#include <stdio.h>
#include <stdlib.h>

__global__ void matmul(const float *A, const float *B, float *C, int n) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int fila = threadIdx.y + blockIdx.y * blockDim.y;

  if (fila < n && col < n) {
    float suma = 0.0f;
    for (int k = 0; k < n; k++)
      suma += A[fila * n + k] * B[k * n + col];
    C[fila * n + col] = suma;
  }
}

// Valores deterministas y NO simétricos: si se equivocan los índices de B,
// el resultado cambia y la verificación lo detecta.
static float valorA(int i, int k) { return ((i + k) % 7) * 0.1f; }
static float valorB(int k, int j) { return ((k + 2 * j) % 5) * 0.1f; }

int main(void) {
  int N = 1024;
  size_t size = (size_t)N * N * sizeof(float);

  float *A = (float *)malloc(size);
  float *B = (float *)malloc(size);
  float *C = (float *)malloc(size);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      A[i * N + j] = valorA(i, j);
      B[i * N + j] = valorB(i, j);
    }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  dim3 hilos(16, 16);
  dim3 bloques((N + hilos.x - 1) / hilos.x, (N + hilos.y - 1) / hilos.y);

  // Lanzamiento "en frío" antes de medir
  matmul<<<bloques, hilos>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);
  cudaEventRecord(inicio);
  matmul<<<bloques, hilos>>>(d_A, d_B, d_C, N);
  cudaEventRecord(fin);
  cudaEventSynchronize(fin);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error en el GPU: %s\n", cudaGetErrorString(err));
    return 1;
  }

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, inicio, fin);
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  // Verificación: tres elementos sueltos contra el producto punto en el host
  // (hacer la multiplicación completa en el host costaría N^3 operaciones)
  int prueba[3][2] = {{0, 0}, {N / 2, N / 3}, {N - 1, N - 1}};
  printf("\n  Verificación:\n");
  for (int t = 0; t < 3; t++) {
    int i = prueba[t][0], j = prueba[t][1];
    float esperado = 0.0f;
    for (int k = 0; k < N; k++)
      esperado += valorA(i, k) * valorB(k, j);
    printf("    C[%4d][%4d] = %12.4f  (esperado %12.4f)\n", i, j,
           C[i * N + j], esperado);
  }

  double s = ms * 1.0e-3;
  double flop = 2.0 * N * N * N;
  double bytes_ingenuo = 8.0 * N * N * N + 4.0 * N * N;
  double bytes_minimo = 12.0 * N * N;

  printf("\n  N = %d  (matrices de %.2f MB cada una)\n", N, size / 1.0e6);
  printf("  duración del kernel  : %.3f ms\n", ms);
  printf("  rendimiento          : %.1f GFLOP/s  (%.1f%% del peak de la T4)\n",
         flop / s / 1.0e9, 100.0 * flop / s / 8.1e12);
  printf("\n  FLOP totales         : %.2e\n", flop);
  printf("  tráfico ingenuo      : %8.2f MB   -> AI = %6.3f FLOP/byte\n",
         bytes_ingenuo / 1.0e6, flop / bytes_ingenuo);
  printf("  tráfico mínimo       : %8.2f MB   -> AI = %6.1f FLOP/byte\n",
         bytes_minimo / 1.0e6, flop / bytes_minimo);
  printf("\n  (el punto de inflexión de la T4 está en ~25 FLOP/byte)\n\n");

  free(A);
  free(B);
  free(C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);
  return 0;
}
