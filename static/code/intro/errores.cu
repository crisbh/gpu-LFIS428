#include <stdio.h>
#include <stdlib.h>

// Macro para revisar el resultado de las funciones del API de CUDA
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                                  \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("codigo:%d, mensaje: %s\n", err, cudaGetErrorString(err));        \
      exit(1);                                                                 \
    }                                                                          \
  }

#define N 1024

__global__ void suma_device(int *a, int *b, int *c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  c[idx] = a[idx] + b[idx];
}

int main(void) {
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int size = N * sizeof(int);

  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 2 * i;
  }

  // El macro CHECK envuelve cada llamada del API de CUDA
  CHECK(cudaMalloc((void **)&d_a, size));
  CHECK(cudaMalloc((void **)&d_b, size));
  CHECK(cudaMalloc((void **)&d_c, size));
  CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

  // Invocación correcta del kernel
  suma_device<<<4, 256>>>(d_a, d_b, d_c);
  // El lanzamiento de un kernel no devuelve nada: Checkeamos el error aparte
  CHECK(cudaGetLastError());      // error al lanzar el kernel?
  CHECK(cudaDeviceSynchronize()); // error durante la ejecución?

  CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
  printf("c[0] = %d, c[N-1] = %d\n", c[0], c[N - 1]);

  // Invocación INCORRECTA: 2048 > 1024 threads por bloque.
  // Descomenta las dos líneas de abajo para ver el macro en acción:
  // suma_device<<<1, 2048>>>(d_a, d_b, d_c);
  // CHECK(cudaGetLastError());

  free(a);
  free(b);
  free(c);
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));
  return 0;
}
