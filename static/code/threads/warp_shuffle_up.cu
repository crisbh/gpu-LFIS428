#include <stdio.h>
#define FULL_MASK 0xffffffff // los 32 threads del warp participan

// Suma acumulada (prefix sum inclusiva) dentro de un warp: al final, el lane t
// tiene data[0] + data[1] + ... + data[t].
// __shfl_up_sync(mascara, v, d) entrega a cada lane el valor v del lane que
// está d posiciones más ABAJO (t - d). Los lanes t < d no tienen de quién
// recibir y obtienen su propio valor, así que no deben sumarlo.
__global__ void suma_acumulada_gpu(int *data) {
  int lane = threadIdx.x % 32;
  int val = data[threadIdx.x];
  for (int d = 1; d < 32; d <<= 1) {
    int vecino = __shfl_up_sync(FULL_MASK, val, d);
    if (lane >= d)
      val += vecino;
  }
  data[threadIdx.x] = val;
}

void suma_acumulada_cpu(int *data, int N) {
  for (int i = 1; i < N; i++) {
    data[i] += data[i - 1];
  }
}

void inicializar(int *data, int N) {
  for (int i = 0; i < N; ++i) {
    data[i] = i;
  }
}

int main() {
  // Este programa no funciona para N != 32...
  int N = 1 << 5;

  // Usamos un "warp" de threads solamente
  int n_threads = 32;
  int n_bloques = 1;

  int *h_data, *h_ref, *d_data;

  // asignación de memoria en el host
  h_data = (int *)malloc(N * sizeof(int));
  h_ref = (int *)malloc(N * sizeof(int));

  // asignación de memoria en el GPU
  cudaMalloc((void **)&d_data, N * sizeof(int));

  // incializar el array
  inicializar(h_data, N);
  inicializar(h_ref, N);

  // copiar valores iniciales al GPU
  cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

  // invocar kernel
  suma_acumulada_gpu<<<n_bloques, n_threads>>>(d_data);

  // copiar resultado del GPU al host
  cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

  // calcular en el host para comparar
  suma_acumulada_cpu(h_ref, N);

  int errores = 0;
  for (int i = 0; i < N; i++) {
    printf("lane %2d   host: %4d   GPU: %4d\n", i, h_ref[i], h_data[i]);
    if (h_ref[i] != h_data[i])
      errores++;
  }
  printf(errores ? "Hay %d diferencias.\n" : "Resultados iguales.\n", errores);

  // liberar espacio de memoria en el GPU
  cudaFree(d_data);

  // liberar memoria del host
  free(h_data);
  free(h_ref);

  return 0;
}
