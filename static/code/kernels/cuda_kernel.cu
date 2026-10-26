/* El costo de lanzar un kernel.
 *
 * La misma operación, N_LOOP pasos de SAXPY (y = alpha * x + y), se hace de
 * tres formas:
 *   1. lanzar un kernel por paso, desde un ciclo en el host (saxpySimple);
 *   2. un solo kernel con el ciclo adentro (saxpyIterativo);
 *   3. un kernel que se relanza a sí mismo desde el GPU (saxpyRecursivo),
 *      con paralelismo dinámico.
 * Las tres hacen exactamente las mismas cuentas, y cada paso depende del
 * anterior, así que el compilador no puede eliminar el ciclo. Lo único que
 * cambia es cuántas veces se lanza un kernel y desde dónde.
 *
 * Con 2^10 elementos el trabajo de cada paso es mínimo: lo que se mide es,
 * sobre todo, el costo de los lanzamientos.
 *
 * Compilar (el paralelismo dinámico necesita -rdc=true y cudadevrt):
 *   nvcc -arch=sm_75 -rdc=true cuda_kernel.cu -o cuda_kernel.x -lcudadevrt
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOQUE 256
#define N_LOOP 16 // la recursión anida N_LOOP grids; el límite es 24

__global__ void saxpySimple(float *y, const float *x, float alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  y[idx] = alpha * x[idx] + y[idx];
}

__global__ void saxpyIterativo(float *y, const float *x, float alpha,
                               int n_loop) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float xi = x[idx], v = y[idx];
  for (int i = 0; i < n_loop; i++)
    v = alpha * xi + v;
  y[idx] = v;
}

// Cada bloque hace un paso sobre sus BLOQUE elementos y su thread 0 lanza un
// grid hijo de un bloque sobre esos mismos elementos, para el paso siguiente.
// El __syncthreads() asegura que el hijo vea lo que escribió todo el bloque.
__global__ void saxpyRecursivo(float *y, const float *x, float alpha,
                               int profundidad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  y[idx] = alpha * x[idx] + y[idx];
  __syncthreads();
  if (threadIdx.x == 0 && profundidad > 1) {
    int off = blockIdx.x * blockDim.x;
    saxpyRecursivo<<<1, blockDim.x>>>(y + off, x + off, alpha,
                                      profundidad - 1);
  }
}

// Ejecuta la variante elegida sobre y (que debe partir en 0)
void ejecutar(int variante, float *d_y, const float *d_x, float alpha,
              int grid) {
  switch (variante) {
  case 0:
    for (int i = 0; i < N_LOOP; i++)
      saxpySimple<<<grid, BLOQUE>>>(d_y, d_x, alpha);
    break;
  case 1:
    saxpyIterativo<<<grid, BLOQUE>>>(d_y, d_x, alpha, N_LOOP);
    break;
  case 2:
    // el grid padre no termina hasta que terminan todos sus hijos
    saxpyRecursivo<<<grid, BLOQUE>>>(d_y, d_x, alpha, N_LOOP);
    break;
  }
}

int main() {
  const char *nombre[] = {"ciclo en el host  ", "ciclo en el kernel",
                          "recursión         "};
  int n = 1 << 10;
  size_t bytes = n * sizeof(float);
  int grid = n / BLOQUE;
  float alpha = 0.5f;

  float *h_x = (float *)malloc(bytes), *h_y = (float *)malloc(bytes);
  for (int i = 0; i < n; i++)
    h_x[i] = 1.0f;

  float *d_x, *d_y;
  cudaMalloc((void **)&d_x, bytes);
  cudaMalloc((void **)&d_y, bytes);
  cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);

  // con x = 1 e y = 0, después de N_LOOP pasos y = N_LOOP * alpha
  float esperado = N_LOOP * alpha;

  printf("%d pasos de SAXPY sobre %d elementos\n", N_LOOP, n);
  for (int v = 0; v < 3; v++) {
    // calentamiento: carga el kernel (CUDA 12 los carga en el primer uso)
    cudaMemset(d_y, 0, bytes);
    ejecutar(v, d_y, d_x, alpha, grid);
    cudaDeviceSynchronize();

    cudaMemset(d_y, 0, bytes);
    cudaEventRecord(inicio);
    ejecutar(v, d_y, d_x, alpha, grid);
    cudaEventRecord(fin);
    cudaEventSynchronize(fin);
    float ms;
    cudaEventElapsedTime(&ms, inicio, fin);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    int errores = 0;
    for (int i = 0; i < n; i++)
      if (fabsf(h_y[i] - esperado) > 1e-4f)
        errores++;

    printf("%s: %8.3f ms%s\n", nombre[v], ms,
           errores ? "   ERROR: resultado incorrecto" : "");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("ERROR: %s\n", cudaGetErrorString(err));

  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);
  cudaFree(d_x);
  cudaFree(d_y);
  free(h_x);
  free(h_y);
  return 0;
}
