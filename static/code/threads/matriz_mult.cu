/* Programa implementa el cálculo C = alpha * A * B + beta * C
 * que es parte de la libreria BLAS (SGEMM)
 */

#include <stdio.h>
#include <stdlib.h>

// Tamaño de bloque por defecto; se puede cambiar desde la línea de comandos:
//   ./matriz_mult.x            -> bloques de 4 x 4
//   ./matriz_mult.x 16 16      -> bloques de 16 x 16
#define BLOCK_DIM_X 4
#define BLOCK_DIM_Y 4

__global__ void matriz_mult(const float *A, const float *B, float *C, int N,
                            int M, int K, float alpha, float beta) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int fila = blockIdx.y * blockDim.y + threadIdx.y;

  if (fila >= N || col >= M) // proteger contra el exceso de threads
    return;

  float suma = 0.f;
  for (int i = 0; i < K; ++i) {
    suma += A[fila * K + i] * B[i * M + col];
  }

  C[fila * M + col] = alpha * suma + beta * C[fila * M + col];
}

void random_init(float *data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = (rand() & 0xFF) / (float)RAND_MAX;
  }
}

int main(int argc, char *argv[]) {
  float *A, *B, *C;
  float *d_A, *d_B, *d_C;
  int N, M, K;
  float alpha = 2.f;
  float beta = 1.f;
  N = M = K = 256;

  // tamaño de bloque: argumentos opcionales (por defecto 4 x 4)
  if (argc == 2) {
    printf("Uso: %s [bx by]   (por ejemplo %s 16 16)\n", argv[0], argv[0]);
    return 1;
  }
  int bx = (argc > 2) ? atoi(argv[1]) : BLOCK_DIM_X;
  int by = (argc > 2) ? atoi(argv[2]) : BLOCK_DIM_Y;
  if (bx < 1 || by < 1 || bx * by > 1024) {
    printf("Tamaño de bloque inválido: %d x %d (máximo 1024 threads)\n", bx, by);
    return 1;
  }

  // asignación de memoria en el host
  A = (float *)malloc(N * K * sizeof(float));
  B = (float *)malloc(K * M * sizeof(float));
  C = (float *)malloc(N * M * sizeof(float));

  // asignación de memoria en el GPU
  cudaMalloc((void **)&d_A, N * K * sizeof(float));
  cudaMalloc((void **)&d_B, K * M * sizeof(float));
  cudaMalloc((void **)&d_C, N * M * sizeof(float));

  // incializar las matrices
  random_init(A, N * K);
  random_init(B, K * M);
  random_init(C, N * M);

  // copiar valores iniciales al GPU
  cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice);

  // invocar kernel
  // el grid se redondea hacia arriba, así cubre la matriz con cualquier
  // tamaño de bloque; la guardia del kernel descarta los threads que sobran
  dim3 dimBlock(bx, by);
  dim3 dimGrid((M + bx - 1) / bx, (N + by - 1) / by);
  printf("Matriz %d x %d, bloques de %d x %d (%d threads), grid de %d x %d\n",
         N, M, bx, by, bx * by, dimGrid.x, dimGrid.y);
  matriz_mult<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, M, K, alpha, beta);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    printf("Error en el kernel: %s\n", cudaGetErrorString(err));

  // liberar espacio de memoria en el GPU
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // liberar memoria del host
  free(A);
  free(B);
  free(C);

  return 0;
}
