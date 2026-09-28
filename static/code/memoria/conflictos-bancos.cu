/* Costo de un conflicto de bancos de N vías.
 *
 * Es la transpuesta con memoria compartida (transpuestaCompPad de
 * transpuesta_compartida.cu), pero con un padding P variable:
 *
 *   __shared__ float tile[BDIM][BDIM + P];
 *
 * Al leer tile[tx][ty] el thread cae en el banco (tx*(32+P) + ty) % 32,
 * que es (tx*P + ty) % 32: un conflicto de gcd(P, 32) vías. Recorriendo
 * P = 1, 2, 4, 8, 16, 32 se obtienen conflictos de 1, 2, 4, 8, 16 y 32 vías,
 * con el MISMO kernel y el MISMO tráfico de memoria global.
 *
 * Compilar:  nvcc -arch=sm_75 conflictos-bancos.cu -o conflictos-bancos.x
 * (necesita common.h en el mismo directorio)
 */

#include "common.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BDIM 32
#define REPS 20 // lanzamientos por medición, para promediar

void inicializarMatriz(float *entrada, const int Ntot) {
  for (int i = 0; i < Ntot; i++) {
    entrada[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

bool verificarResultado(float *hostRef, float *gpuRef, const int Ntot) {
  for (int i = 0; i < Ntot; i++) {
    if (fabsf(hostRef[i] - gpuRef[i]) > 1.0E-8) {
      printf("diferencia en elemento %d: host %f gpu %f\n", i, hostRef[i],
             gpuRef[i]);
      return false;
    }
  }
  return true;
}

void transpuestaHost(float *salida, float *entrada, const int N) {
  for (int iy = 0; iy < N; ++iy) {
    for (int ix = 0; ix < N; ++ix) {
      salida[ix * N + iy] = entrada[iy * N + ix];
    }
  }
}

// Número de vías del conflicto al leer una columna de tile[BDIM][BDIM + pad]
int vias(int pad) {
  int a = pad, b = 32;
  while (b != 0) { // máximo común divisor (Euclides); gcd(0, 32) = 32
    int r = a % b;
    a = b;
    b = r;
  }
  return a;
}

// Transpuesta con memoria compartida y padding PAD.
// PAD es un parámetro de template porque el tamaño de un arreglo
// __shared__ estático tiene que conocerse al compilar.
template <int PAD>
__global__ void transpuestaPad(float *salida, const float *entrada, int N) {
  __shared__ float tile[BDIM][BDIM + PAD];

  // coordenadas globales en la matriz original
  unsigned int ix, iy, ti, to, ixt, iyt;
  ix = blockDim.x * blockIdx.x + threadIdx.x;
  iy = blockDim.y * blockIdx.y + threadIdx.y;

  // coordenada en memoria lineal
  ti = iy * N + ix;

  // coordenada en matriz transpuesta
  ixt = blockDim.y * blockIdx.y + threadIdx.x;
  iyt = blockDim.x * blockIdx.x + threadIdx.y;

  // coordenada global lineal en matriz transpuesta
  to = iyt * N + ixt;

  if (ixt < N && iyt < N) {
    // escritura por filas: nunca hay conflicto, para cualquier PAD
    tile[threadIdx.y][threadIdx.x] = entrada[ti];

    __syncthreads();

    // lectura por columnas: conflicto de gcd(PAD, 32) vías
    salida[to] = tile[threadIdx.x][threadIdx.y];
  }
}

// Mide el tiempo promedio (en segundos) de transpuestaPad<PAD>
template <int PAD>
double medir(float *d_C, float *d_A, float *gpuRef, float *hostRef, int N,
             dim3 grid, dim3 block) {
  int nBytes = N * N * sizeof(float);

  // calentamiento: el primer lanzamiento incluye costos de inicialización
  transpuestaPad<PAD><<<grid, block>>>(d_C, d_A, N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemset(d_C, 0, nBytes));

  double comienzo = segundos();
  for (int r = 0; r < REPS; r++) {
    transpuestaPad<PAD><<<grid, block>>>(d_C, d_A, N);
  }
  CHECK(cudaDeviceSynchronize());
  double tiempo = (segundos() - comienzo) / REPS;

  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
  if (!verificarResultado(hostRef, gpuRef, N * N))
    printf("PAD = %d: las matrices no coinciden.\n", PAD);

  return tiempo;
}

void imprimir(int pad, double tiempo, double referencia, int N) {
  double gbs = 2.0 * N * N * sizeof(float) / 1e9 / tiempo;
  printf("%7d  %4d  %11.3f  %6.1f  %7.2fx\n", pad, vias(pad), tiempo * 1e3,
         gbs, tiempo / referencia);
}

int main(int argc, char **argv) {
  // matriz de 4096x4096
  int N = 1 << 12;
  int nBytes = N * N * sizeof(float);

  dim3 block(BDIM, BDIM);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  printf("Matriz %d x %d, bloques (%d,%d), promedio de %d lanzamientos\n\n",
         N, N, block.x, block.y, REPS);

  // memoria en el host y transpuesta de referencia
  float *h_A = (float *)malloc(nBytes);
  float *hostRef = (float *)malloc(nBytes);
  float *gpuRef = (float *)malloc(nBytes);
  inicializarMatriz(h_A, N * N);
  transpuestaHost(hostRef, h_A, N);

  // memoria en el device
  float *d_A, *d_C;
  CHECK(cudaMalloc((float **)&d_A, nBytes));
  CHECK(cudaMalloc((float **)&d_C, nBytes));
  CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

  printf("padding  vias  tiempo (ms)    GB/s  lentitud\n");

  // PAD = 1 (sin conflicto) es la referencia para la columna "lentitud"
  double t1 = medir<1>(d_C, d_A, gpuRef, hostRef, N, grid, block);
  imprimir(1, t1, t1, N);

  imprimir(2, medir<2>(d_C, d_A, gpuRef, hostRef, N, grid, block), t1, N);
  imprimir(4, medir<4>(d_C, d_A, gpuRef, hostRef, N, grid, block), t1, N);
  imprimir(8, medir<8>(d_C, d_A, gpuRef, hostRef, N, grid, block), t1, N);
  imprimir(16, medir<16>(d_C, d_A, gpuRef, hostRef, N, grid, block), t1, N);
  imprimir(32, medir<32>(d_C, d_A, gpuRef, hostRef, N, grid, block), t1, N);

  // liberar memoria del host y del device
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_C));
  free(h_A);
  free(hostRef);
  free(gpuRef);

  CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}
