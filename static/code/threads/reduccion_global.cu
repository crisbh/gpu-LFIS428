#include <stdio.h>
#include <stdlib.h>

// Lee de data_in y escribe en data_out: dos arreglos distintos. Si se
// escribiera en el mismo arreglo (data[idx] += data[idx + stride]), el thread
// idx leería data[idx + stride] mientras el thread idx + stride lo está
// modificando en el mismo lanzamiento: una race condition.
__global__ void reduccion_memoria_global(float *data_out, const float *data_in,
                                         int stride, int N) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx_x + stride < N) {
    data_out[idx_x] = data_in[idx_x] + data_in[idx_x + stride];
  } else if (idx_x < N) {
    data_out[idx_x] = data_in[idx_x];
  }
}

void inicializar_numeros(float *data, int size) {
  for (int i = 0; i < size; i++) {
    // Generamos números aleatorios pequeños
    // El "0xFF" es un número en hexadecimal y está combinado con
    // rand() usando el "bitwise operator" & para obtener números
    // entre 0 y 255.
    data[i] = (float)(rand() & 0xFF) / (float)RAND_MAX;
  }
}

float resultado_cpu(float *data, int N) {
  float resultado = 0.f;
  for (int i = 0; i < N; i++) {
    resultado += data[i];
  }

  return resultado;
}

int main() {
  float *h_array;
  float *d_in, *d_out;

  unsigned int N = 1 << 24;
  unsigned int n_threads = 1024;

  float resultado_host, resultado_gpu;

  srand(2019);

  // Asignar memoria en el host
  h_array = (float *)malloc(N * sizeof(float));

  // Iniciliazar valores de h_array con números aleatorios
  inicializar_numeros(h_array, N);

  // Asignar memoria en el GPU (entrada y salida) y copiar datos
  cudaMalloc((void **)&d_in, N * sizeof(float));
  cudaMalloc((void **)&d_out, N * sizeof(float));
  cudaMemcpy(d_in, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

  // medir el tiempo de los kernels con eventos de CUDA (los veremos en la
  // clase de streams); cudaFuncGetAttributes carga el kernel antes de medir
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, reduccion_memoria_global);
  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);
  float ms;
  cudaEventRecord(inicio);

  // Calcular reducción en el GPU: después de cada lanzamiento la salida pasa
  // a ser la entrada del siguiente (se intercambian los punteros)
  int n_bloques = (N + n_threads - 1) / n_threads;
  for (int stride = 1; stride < N; stride *= 2) {
    reduccion_memoria_global<<<n_bloques, n_threads>>>(d_out, d_in, stride, N);
    float *tmp = d_in;
    d_in = d_out;
    d_out = tmp;
  }

  cudaEventRecord(fin);
  cudaEventSynchronize(fin);
  cudaEventElapsedTime(&ms, inicio, fin);

  // Copiar resultado del GPU: después del último intercambio está en d_in
  cudaMemcpy(&resultado_gpu, &d_in[0], sizeof(float), cudaMemcpyDeviceToHost);

  // Calcular reducción en el CPU (secuencial)
  resultado_host = resultado_cpu(h_array, N);
  printf("host: %f, device %f\n", resultado_host, resultado_gpu);
  printf("tiempo GPU: %.3f ms\n", ms);

  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);

  // Liberar memoria
  cudaFree(d_in);
  cudaFree(d_out);
  free(h_array);

  return 0;
}
