/* Pipelining con streams: solapar copias y cómputo.
 *
 * El mismo trabajo (copiar a y b al GPU, calcular c, copiar c de vuelta) se
 * hace de dos formas:
 *   - con 1 stream: copia todo, calcula todo, copia todo de vuelta;
 *   - con varios streams: los datos se parten en pedazos, y mientras un
 *     stream calcula su pedazo, otro ya está copiando el siguiente.
 *
 * Requisitos para que las copias se solapen con el cómputo:
 *   - memoria del host pinned (cudaMallocHost);
 *   - copias asincrónicas (cudaMemcpyAsync);
 *   - cada pedazo en su propio stream (nada en el default stream).
 *
 * Compilar:  nvcc -arch=sm_75 cuda_pipelining.cu -o cuda_pipelining.x
 * Ejecutar:  ./cuda_pipelining.x [número de streams, por defecto 4]
 * Ver el solapamiento:  nvprof --print-gpu-trace ./cuda_pipelining.x
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOQUE 256
#define ITER 2000 // trabajo por elemento: suficiente para que el cómputo
                  // tarde algo comparable a las copias

// Cálculo con suficiente aritmética para que no lo domine la memoria.
// El compilador no puede eliminar el ciclo: cada iteración depende de la
// anterior.
__global__ void calcular(float *c, const float *a, const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = a[idx];
    for (int i = 0; i < ITER; i++)
      v = v * 0.9999f + b[idx];
    c[idx] = v;
  }
}

// El mismo cálculo en el host, para verificar algunos elementos
float calcularHost(float a, float b) {
  float v = a;
  for (int i = 0; i < ITER; i++)
    v = v * 0.9999f + b;
  return v;
}

// Procesa n elementos repartidos en n_streams pedazos y devuelve el tiempo
// total en ms (desde la primera copia hasta la última)
float procesar(int n_streams, float *h_c, const float *h_a, const float *h_b,
               float *d_c, float *d_a, float *d_b, int n) {
  cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
  for (int s = 0; s < n_streams; s++)
    cudaStreamCreate(&streams[s]);

  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);

  int pedazo = n / n_streams; // n es múltiplo de n_streams
  size_t bytes = pedazo * sizeof(float);

  cudaEventRecord(inicio);
  for (int s = 0; s < n_streams; s++) {
    int off = s * pedazo;
    // copiar host -> device, calcular y copiar device -> host, todo en el
    // stream s: dentro de un stream las operaciones van en orden, entre
    // streams distintos pueden solaparse
    cudaMemcpyAsync(d_a + off, h_a + off, bytes, cudaMemcpyHostToDevice,
                    streams[s]);
    cudaMemcpyAsync(d_b + off, h_b + off, bytes, cudaMemcpyHostToDevice,
                    streams[s]);
    calcular<<<(pedazo + BLOQUE - 1) / BLOQUE, BLOQUE, 0, streams[s]>>>(
        d_c + off, d_a + off, d_b + off, pedazo);
    cudaMemcpyAsync(h_c + off, d_c + off, bytes, cudaMemcpyDeviceToHost,
                    streams[s]);
  }
  // el evento "fin" va en el default stream, que espera a todos los demás
  cudaEventRecord(fin);
  cudaEventSynchronize(fin);

  float ms;
  cudaEventElapsedTime(&ms, inicio, fin);

  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);
  for (int s = 0; s < n_streams; s++)
    cudaStreamDestroy(streams[s]);
  free(streams);
  return ms;
}

int main(int argc, char *argv[]) {
  int n = 1 << 24;
  size_t bytes = n * sizeof(float);
  int n_streams = (argc > 1) ? atoi(argv[1]) : 4;

  // memoria pinned en el host: sin ella, cudaMemcpyAsync no se solapa
  float *h_a, *h_b, *h_c;
  cudaMallocHost((void **)&h_a, bytes);
  cudaMallocHost((void **)&h_b, bytes);
  cudaMallocHost((void **)&h_c, bytes);

  srand(2019);
  for (int i = 0; i < n; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, bytes);
  cudaMalloc((void **)&d_b, bytes);
  cudaMalloc((void **)&d_c, bytes);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("%s: %d motores de copia (asyncEngineCount)\n", prop.name,
         prop.asyncEngineCount);
  printf("%zu MB por arreglo; se copian a y b, y se trae c de vuelta\n\n",
         bytes >> 20);

  // calentamiento: crea el contexto y carga el kernel
  procesar(1, h_c, h_a, h_b, d_c, d_a, d_b, n);

  float t1 = procesar(1, h_c, h_a, h_b, d_c, d_a, d_b, n);
  float tn = procesar(n_streams, h_c, h_a, h_b, d_c, d_a, d_b, n);

  // verificar algunos elementos
  int errores = 0;
  for (int i = 0; i < n; i += n / 16)
    if (fabs(h_c[i] - calcularHost(h_a[i], h_b[i])) > 1e-3f * fabs(h_c[i]))
      errores++;
  if (errores)
    printf("ERROR: %d elementos no coinciden\n", errores);

  printf("streams   tiempo (ms)\n");
  printf("%7d   %11.3f\n", 1, t1);
  printf("%7d   %11.3f   (%.2fx más rápido)\n", n_streams, tn, t1 / tn);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  return 0;
}
