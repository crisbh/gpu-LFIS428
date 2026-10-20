#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void saxpy(int N, float a, float *x, float *y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

void saxpy_cpu(int N, float a, float *x, float *y) {
  for (int i = 0; i < N; i++) {
    y[i] = a * x[i] + y[i];
  }
}

void inicializar(float *data, int N) {
  for (int i = 0; i < N; ++i) {
    data[i] = (rand()) / (float)RAND_MAX;
  }
}

int main() {
  int N = 1 << 24;

  // El grid NO depende de N: basta con llenar el GPU. En la T4 caben 4
  // bloques de 256 threads por SM, así que lanzamos 4 bloques por SM y cada
  // thread procesa varios elementos en el ciclo del kernel.
  int n_threads = 256;
  int n_sm;
  cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
  int n_bloques = 4 * n_sm;
  printf("N = %d, %d bloques de %d threads: %d elementos por thread\n", N,
         n_bloques, n_threads, (N + n_bloques * n_threads - 1) /
                                   (n_bloques * n_threads));

  float a = 2.0f;
  float *h_x, *h_y, *resultado_gpu;
  float *d_x, *d_y;

  // asignación de memoria en el host
  h_x = (float *)malloc(N * sizeof(float));
  h_y = (float *)malloc(N * sizeof(float));
  resultado_gpu = (float *)malloc(N * sizeof(float));

  // asignación de memoria en el GPU
  cudaMalloc((void **)&d_x, N * sizeof(float));
  cudaMalloc((void **)&d_y, N * sizeof(float));

  // incializar los vectores
  inicializar(h_x, N);
  inicializar(h_y, N);

  // copiar valores iniciales al GPU
  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

  // invocar kernel
  saxpy<<<n_bloques, n_threads>>>(N, a, d_x, d_y);

  // copiar resultado del GPU al host
  cudaMemcpy(resultado_gpu, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  // calcular en el host para comparar
  saxpy_cpu(N, a, h_x, h_y);

  int error = 0;
  float err;
  for (int i = 0; i < N; i++) {
    err = fabsf(h_y[i] - resultado_gpu[i]);
    if (err > 1e-6f * fabsf(h_y[i])) {
      error = 1;
    }
  }
  if (error == 1) {
    printf("Vectores no coinciden.\n");
  } else {
    printf("Exito! Vectores coinciden.\n");
  }

  // liberar espacio de memoria en el GPU
  cudaFree(d_x);
  cudaFree(d_y);

  // liberar memoria del host
  free(h_x);
  free(h_y);
  free(resultado_gpu);

  return 0;
}
