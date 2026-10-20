#include <stdio.h>
#define FULL_MASK 0xffffffff // los 32 threads del warp participan

// Reducción "en mariposa" (butterfly) dentro de un warp.
// __shfl_xor_sync(mascara, v, m) entrega a cada lane el valor v del lane
// cuyo índice es (lane XOR m): con m = 16 se intercambian las dos mitades del
// warp, con m = 8 los cuartos, etc. Cada par de lanes intercambia sus valores
// y los dos suman lo mismo, así que al final TODOS los lanes tienen la suma
// (con __shfl_down_sync solo el lane 0 la tiene).
__global__ void reduccion_gpu(int *data) {
  int val = data[threadIdx.x];
  for (int m = 16; m > 0; m >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, m);
  }
  data[threadIdx.x] = val;
}

int reduccion_cpu(int *data, int N) {
  int resultado = 0;
  for (int i = 0; i < N; i++) {
    resultado += data[i];
  }
  return resultado;
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

  int *h_data, *d_data;

  // asignación de memoria en el host
  h_data = (int *)malloc(N * sizeof(int));

  // asignación de memoria en el GPU
  cudaMalloc((void **)&d_data, N * sizeof(int));

  // incializar el array
  inicializar(h_data, N);

  // calcular en el host para comparar
  int resultado_cpu = reduccion_cpu(h_data, N);

  // copiar valores iniciales al GPU
  cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

  // invocar kernel
  reduccion_gpu<<<n_bloques, n_threads>>>(d_data);

  // copiar resultado del GPU al host: un valor por lane
  cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

  printf("Host: %d\n", resultado_cpu);
  for (int i = 0; i < N; i++) {
    printf("GPU, lane %2d: %d\n", i, h_data[i]);
  }

  // liberar espacio de memoria en el GPU
  cudaFree(d_data);

  // liberar memoria del host
  free(h_data);

  return 0;
}
