#include <stdio.h>
#include <stdlib.h>

// SAXPY: Y = a*X + Y  (single-precision A*X Plus Y)
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) // proteger contra el exceso de threads
    y[i] = a * x[i] + y[i];
}

int main(void) {
  int N = 1 << 20; // 1.048.576 elementos
  size_t size = N * sizeof(float);

  float *x = (float *)malloc(size);
  float *y = (float *)malloc(size);
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  float *d_x, *d_y;
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

  // Lanzar suficientes bloques de 256 threads para cubrir los N elementos
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  saxpy<<<blocks, threads>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

  free(x);
  free(y);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}
