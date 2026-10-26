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
 *
 * Además de los tiempos totales, el programa imprime una línea de tiempo por
 * etapa (copia H->D, kernel, copia D->H), como la de las diapositivas: así se
 * ve qué se solapó sin necesidad de un profiler.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOQUE 256
#define ITER 2000 // trabajo por elemento: suficiente para que el cómputo
                  // tarde algo comparable a las copias
#define ETAPAS 3    // copia H->D, kernel, copia D->H
#define COLUMNAS 60 // ancho de la línea de tiempo impresa

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
// total en ms (desde la primera copia hasta la última). En
// fin[e * n_streams + s] deja cuándo terminó la etapa e del stream s, en ms
// desde el inicio.
float procesar(int n_streams, float *h_c, const float *h_a, const float *h_b,
               float *d_c, float *d_a, float *d_b, int n, float *fin_etapa) {
  cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
  for (int s = 0; s < n_streams; s++)
    cudaStreamCreate(&streams[s]);

  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);

  // un evento al final de cada etapa de cada stream
  cudaEvent_t *ev =
      (cudaEvent_t *)malloc(ETAPAS * n_streams * sizeof(cudaEvent_t));
  for (int i = 0; i < ETAPAS * n_streams; i++)
    cudaEventCreate(&ev[i]);

  int pedazo = n / n_streams; // n es múltiplo de n_streams
  size_t bytes = pedazo * sizeof(float);

  cudaEventRecord(inicio);
  for (int s = 0; s < n_streams; s++) {
    int off = s * pedazo;
    // todo en el stream s: en orden dentro del stream, solapable entre streams
    cudaMemcpyAsync(d_a + off, h_a + off, bytes, cudaMemcpyHostToDevice,
                    streams[s]);
    cudaMemcpyAsync(d_b + off, h_b + off, bytes, cudaMemcpyHostToDevice,
                    streams[s]);
    cudaEventRecord(ev[0 * n_streams + s], streams[s]);
    calcular<<<(pedazo + BLOQUE - 1) / BLOQUE, BLOQUE, 0, streams[s]>>>(
        d_c + off, d_a + off, d_b + off, pedazo);
    cudaEventRecord(ev[1 * n_streams + s], streams[s]);
    cudaMemcpyAsync(h_c + off, d_c + off, bytes, cudaMemcpyDeviceToHost,
                    streams[s]);
    cudaEventRecord(ev[2 * n_streams + s], streams[s]);
  }
  // el evento "fin" va en el default stream, que espera a todos los demás
  cudaEventRecord(fin);
  cudaEventSynchronize(fin);

  float ms;
  cudaEventElapsedTime(&ms, inicio, fin);
  for (int i = 0; i < ETAPAS * n_streams; i++) {
    cudaEventElapsedTime(&fin_etapa[i], inicio, ev[i]);
    cudaEventDestroy(ev[i]);
  }
  free(ev);

  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);
  for (int s = 0; s < n_streams; s++)
    cudaStreamDestroy(streams[s]);
  free(streams);
  return ms;
}

// Imprime una línea por etapa; cada columna son ms_col ms y cada pedazo se
// marca con el número de su stream.
//
// Los eventos solo dicen cuándo TERMINA cada etapa: un evento puesto antes de
// una copia se cumple apenas el stream llega a él, aunque la copia todavía
// espere en la fila del motor de copia. El comienzo se deduce: una etapa
// empieza cuando terminó la etapa anterior de su stream y cuando el motor
// quedó libre, o sea cuando terminó la misma etapa del stream anterior (cada
// motor atiende los pedazos en el orden en que se lanzaron, y cada kernel
// ocupa todo el GPU).
void imprimirLinea(int n_streams, const float *fin, float ms_col) {
  const char *nombre[ETAPAS] = {"Copia H->D", "Kernel    ", "Copia D->H"};
  for (int e = 0; e < ETAPAS; e++) {
    char linea[COLUMNAS + 1];
    memset(linea, '.', COLUMNAS);
    linea[COLUMNAS] = '\0';
    for (int s = 0; s < n_streams; s++) {
      float ini = 0.f;
      if (e > 0)
        ini = fmaxf(ini, fin[(e - 1) * n_streams + s]);
      if (s > 0)
        ini = fmaxf(ini, fin[e * n_streams + s - 1]);
      int c0 = (int)(ini / ms_col + 0.5f);
      int c1 = (int)(fin[e * n_streams + s] / ms_col + 0.5f);
      if (c1 <= c0)
        c1 = c0 + 1; // que hasta el pedazo más corto se vea
      for (int c = c0; c < c1 && c < COLUMNAS; c++)
        linea[c] = '0' + s % 10;
    }
    printf("  %s |%s|\n", nombre[e], linea);
  }
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

  float *fin1 = (float *)malloc(ETAPAS * sizeof(float));
  float *finn = (float *)malloc(ETAPAS * n_streams * sizeof(float));

  // calentamiento: crea el contexto y carga el kernel
  procesar(1, h_c, h_a, h_b, d_c, d_a, d_b, n, fin1);

  float t1 = procesar(1, h_c, h_a, h_b, d_c, d_a, d_b, n, fin1);
  float tn = procesar(n_streams, h_c, h_a, h_b, d_c, d_a, d_b, n, finn);

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

  // las dos líneas con la misma escala, para que se puedan comparar
  float ms_col = t1 / COLUMNAS;
  printf("\nLínea de tiempo (cada columna = %.2f ms):\n", ms_col);
  printf("1 stream:\n");
  imprimirLinea(1, fin1, ms_col);
  printf("%d streams:\n", n_streams);
  imprimirLinea(n_streams, finn, ms_col);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  free(fin1);
  free(finn);
  return 0;
}
