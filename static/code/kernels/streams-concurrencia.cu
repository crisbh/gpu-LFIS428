/* Concurrencia entre streams: qué se ejecuta a la vez y qué no.
 *
 * El kernel esperar() no calcula nada: un solo thread da vueltas hasta que
 * pasa cierto número de ciclos de reloj. Así cada lanzamiento dura un tiempo
 * T conocido (unos 20 ms) y lo único que cambia entre los casos es en qué
 * stream se lanza. Se lanzan K = 5 kernels (numerados 0 a 4) de cinco formas:
 *   1. todos en el default stream;
 *   2. cada uno en su propio stream;
 *   3. cada uno en su propio stream, con cudaStreamSynchronize después de
 *      cada lanzamiento;
 *   4. cada uno en su propio stream, salvo el kernel 3, que va al default
 *      stream;
 *   5. igual que 4, pero con streams creados con cudaStreamNonBlocking.
 *
 * El tiempo se mide con un cronómetro del host, desde el primer lanzamiento
 * hasta cudaDeviceSynchronize(). Acá no sirven los eventos: un evento en el
 * default stream no espera a los streams non-blocking del caso 5. Con
 * kernels de 20 ms, el ruido del host es despreciable.
 *
 * Compilar:  nvcc -arch=sm_75 streams-concurrencia.cu -o streams-concurrencia.x
 */

#include <stdio.h>
#include <sys/time.h>

#define K 5
#define T_MS 20.0f // duración buscada de cada kernel

// Un solo thread espera hasta que pasan 'ciclos' ciclos del reloj del SM
__global__ void esperar(long long ciclos) {
  long long t0 = clock64();
  while (clock64() - t0 < ciclos) {
  }
}

double segundos() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

// Cuántos ciclos hay que esperar para que un kernel dure unos T_MS ms
long long calibrar() {
  long long c = 10000000; // 10^7 ciclos: unos 6 ms a 1.59 GHz

  // calentamiento: carga el kernel y despierta los relojes del GPU
  esperar<<<1, 1>>>(30 * c);
  cudaDeviceSynchronize();

  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);
  cudaEventRecord(inicio);
  esperar<<<1, 1>>>(c);
  cudaEventRecord(fin);
  cudaEventSynchronize(fin);
  float ms;
  cudaEventElapsedTime(&ms, inicio, fin);
  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);

  return (long long)(c * (T_MS / ms));
}

// Lanza los K kernels según el caso y devuelve el tiempo total en ms. En
// t_lanzar deja lo que tardó el host solo en lanzarlos.
double medir(int caso, long long ciclos, cudaStream_t *bloq,
             cudaStream_t *nobloq, double *t_lanzar) {
  double t0 = segundos();
  for (int k = 0; k < K; k++) {
    switch (caso) {
    case 1:
      esperar<<<1, 1>>>(ciclos);
      break;
    case 2:
      esperar<<<1, 1, 0, bloq[k]>>>(ciclos);
      break;
    case 3:
      esperar<<<1, 1, 0, bloq[k]>>>(ciclos);
      cudaStreamSynchronize(bloq[k]); // el host espera a este stream
      break;
    case 4:
      esperar<<<1, 1, 0, k == 3 ? (cudaStream_t)0 : bloq[k]>>>(ciclos);
      break;
    case 5:
      esperar<<<1, 1, 0, k == 3 ? (cudaStream_t)0 : nobloq[k]>>>(ciclos);
      break;
    }
  }
  *t_lanzar = (segundos() - t0) * 1e3;
  cudaDeviceSynchronize();
  return (segundos() - t0) * 1e3;
}

int main() {
  const char *nombre[] = {"",
                          "1. todos en el default stream",
                          "2. un stream por kernel",
                          "3. un stream por kernel + cudaStreamSynchronize",
                          "4. kernel 3 en el default stream",
                          "5. como 4, con streams non-blocking"};

  cudaStream_t bloq[K], nobloq[K];
  for (int k = 0; k < K; k++) {
    cudaStreamCreate(&bloq[k]);
    cudaStreamCreateWithFlags(&nobloq[k], cudaStreamNonBlocking);
  }

  long long ciclos = calibrar();

  // T: un kernel solo, medido igual que los casos
  double t0 = segundos();
  esperar<<<1, 1>>>(ciclos);
  cudaDeviceSynchronize();
  double T = (segundos() - t0) * 1e3;
  printf("T = %.1f ms (un kernel solo); se lanzan %d kernels por caso\n\n", T,
         K);

  printf("%-50s %10s %9s\n", "caso", "tiempo", "tiempo/T");
  double t_lanzar, t_lanzar2 = 0, t_total2 = 0;
  for (int caso = 1; caso <= 5; caso++) {
    double t = medir(caso, ciclos, bloq, nobloq, &t_lanzar);
    printf("%-50s %7.1f ms %9.2f\n", nombre[caso], t, t / T);
    if (caso == 2) {
      t_lanzar2 = t_lanzar;
      t_total2 = t;
    }
  }

  printf("\nEn el caso 2, el host lanzó los %d kernels en %.3f ms y siguió;\n"
         "el GPU terminó %.1f ms después del primer lanzamiento.\n",
         K, t_lanzar2, t_total2);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("ERROR: %s\n", cudaGetErrorString(err));

  for (int k = 0; k < K; k++) {
    cudaStreamDestroy(bloq[k]);
    cudaStreamDestroy(nobloq[k]);
  }
  return 0;
}
