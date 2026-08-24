// ============================================================
// Ejercicio 2: El bug de los límites
// ============================================================
// OBJETIVO: identificar y arreglar el bug más común en CUDA:
//           threads que escriben fuera del arreglo.
//
// INSTRUCCIONES:
//   1. Ejecuta el código tal como está con N = 1000.
//      ¿Da resultados correctos? ¿Por qué (no)?
//   2. Modifica el kernel para arreglar el problema.
//   3. Extensión: prueba con N = 50, N = 999, N = 1024.
//      ¿Qué fórmula general para blocks funciona siempre?
// ============================================================

#include <stdio.h>
#include <stdlib.h>

__global__ void suma_device(int *a, int *b, int *c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // TODO: falta una línea aquí para evitar el acceso fuera de rango.
  c[idx] = a[idx] + b[idx];
}

int main(void) {
  int N = 1000; // <-- prueba también con 50, 999, 1024
  int size = N * sizeof(int);

  // Host
  int *a = (int *)malloc(size);
  int *b = (int *)malloc(size);
  int *c = (int *)malloc(size);
  for (int idx = 0; idx < N; idx++) {
    a[idx] = idx;
    b[idx] = 2 * idx;
    c[idx] = -1; // guardia: si queda en -1, ese slot no fue calculado
  }

  // Device
  int *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Lanzamiento: 4 bloques de 256 threads = 1024 threads en total
  int threads = 256;
  int blocks = 4;
  suma_device<<<blocks, threads>>>(d_a, d_b, d_c, N);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Verificación
  int errores = 0;
  for (int idx = 0; idx < N; idx++) {
    int esperado = a[idx] + b[idx]; // = 3 * idx
    if (c[idx] != esperado) {
      if (errores < 5)
        printf("Error en idx=%d: esperado %d, obtenido %d\n", idx, esperado,
               c[idx]);
      errores++;
    }
  }
  if (errores == 0)
    printf("Todos los resultados coinciden.\n");
  else
    printf("Total de errores: %d\n", errores);

  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
