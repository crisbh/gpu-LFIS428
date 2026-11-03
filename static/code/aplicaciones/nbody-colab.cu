/* N-cuerpos sin OpenGL: el mismo kernel de nbody.cu, para correr en Colab.
 *
 * nbody.cu dibuja los cuerpos en una ventana, así que necesita pantalla y un
 * archivo de datos. Este programa usa el mismo cálculo (bodyBodyInteraction,
 * tiles en memoria compartida, un paso leapfrog por lanzamiento), pero genera
 * los cuerpos al azar dentro de una esfera y en vez de dibujar imprime:
 *   - cuánto dura un paso de tiempo;
 *   - cuántas interacciones por segundo calcula el GPU (N^2 por paso);
 *   - los GFLOP/s, contando 20 operaciones por interacción (GPU Gems 3).
 * Antes de medir compara las velocidades tras un paso del GPU con el mismo
 * paso en el host.
 *
 * Compilar:  nvcc -arch=sm_75 nbody-colab.cu -o nbody-colab.x
 * Ejecutar:  ./nbody-colab.x [N, por defecto 16384] [p, por defecto 256]
 *                            [pasos, por defecto 20]
 * N tiene que ser múltiplo de p (el tamaño del bloque y del tile).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPS2 0.01f // softening al cuadrado
#define DT 0.001f  // paso de tiempo
#define FLOPS_POR_INTERACCION 20

// Aceleración que el cuerpo bj produce sobre bi, sumada a ai (G = 1).
// bj.w es la masa de j.
__device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj) {
  float3 r;
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
  float invDist = rsqrtf(distSqr);
  float invDistCube = invDist * invDist * invDist;
  float s = bj.w * invDistCube;
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
  return ai;
}

// Recorre la fila del cuerpo de este thread tile por tile: cada thread del
// bloque copia un cuerpo a memoria compartida y después todos usan los p.
__device__ float3 computeBodyAccel(float4 bodyPos, const float4 *positions,
                                   int numTiles) {
  extern __shared__ float4 sharedPos[];
  float3 acc = {0.0f, 0.0f, 0.0f};

  for (int tile = 0; tile < numTiles; tile++) {
    sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
    __syncthreads(); // el tile completo está en memoria compartida

    for (int j = 0; j < blockDim.x; j++)
      acc = bodyBodyInteraction(acc, bodyPos, sharedPos[j]);

    __syncthreads(); // nadie sobrescribe el tile mientras otro lo usa
  }
  return acc;
}

// Un paso leapfrog: v += a dt, x += v dt. Lee oldPos y escribe newPos, porque
// los demás bloques todavía están leyendo las posiciones viejas.
__global__ void integrarCuerpos(float4 *newPos, const float4 *oldPos,
                                float4 *vel, int numTiles) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 pos = oldPos[i];
  float3 a = computeBodyAccel(pos, oldPos, numTiles);

  float4 v = vel[i];
  v.x += a.x * DT;
  v.y += a.y * DT;
  v.z += a.z * DT;
  pos.x += v.x * DT;
  pos.y += v.y * DT;
  pos.z += v.z * DT;

  newPos[i] = pos;
  vel[i] = v;
}

// La velocidad del cuerpo i tras un paso, en el host y en double
void pasoHost(int i, const float4 *pos, const float4 *vel, int n,
              double nueva[3]) {
  double a[3] = {0.0, 0.0, 0.0};
  for (int j = 0; j < n; j++) {
    double r[3] = {(double)pos[j].x - pos[i].x, (double)pos[j].y - pos[i].y,
                   (double)pos[j].z - pos[i].z};
    double d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + EPS2;
    double s = pos[j].w / (d2 * sqrt(d2));
    for (int k = 0; k < 3; k++)
      a[k] += r[k] * s;
  }
  double v[3] = {vel[i].x, vel[i].y, vel[i].z};
  for (int k = 0; k < 3; k++)
    nueva[k] = v[k] + a[k] * DT;
}

int main(int argc, char *argv[]) {
  int n = (argc > 1) ? atoi(argv[1]) : 16384;
  int p = (argc > 2) ? atoi(argv[2]) : 256;
  int pasos = (argc > 3) ? atoi(argv[3]) : 20;
  if (p < 1 || p > 1024 || n < p || n % p != 0 || pasos < 1) {
    printf("ERROR: se necesita 1 <= p <= 1024, N múltiplo de p y pasos >= 1\n");
    return 1;
  }
  int numTiles = n / p;
  size_t bytes = n * sizeof(float4);
  size_t compartida = p * sizeof(float4); // un float4 por cuerpo del tile

  // cuerpos al azar dentro de una esfera de radio 1, en reposo, masa total 1
  float4 *h_pos = (float4 *)malloc(bytes);
  float4 *h_vel = (float4 *)malloc(bytes);
  float4 *h_nueva = (float4 *)malloc(bytes);
  srand(2019);
  for (int i = 0; i < n; i++) {
    float x, y, z;
    do {
      x = 2.0f * rand() / RAND_MAX - 1.0f;
      y = 2.0f * rand() / RAND_MAX - 1.0f;
      z = 2.0f * rand() / RAND_MAX - 1.0f;
    } while (x * x + y * y + z * z > 1.0f);
    h_pos[i] = make_float4(x, y, z, 1.0f / n);
    h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  float4 *d_pos[2], *d_vel;
  cudaMalloc((void **)&d_pos[0], bytes);
  cudaMalloc((void **)&d_pos[1], bytes);
  cudaMalloc((void **)&d_vel, bytes);
  cudaMemcpy(d_pos[0], h_pos, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vel, h_vel, bytes, cudaMemcpyHostToDevice);

  printf("N = %d cuerpos, p = %d threads por bloque, %d bloques, %d tiles\n",
         n, p, n / p, numTiles);

  // un paso, para verificarlo contra el host (y de paso calentar el GPU)
  integrarCuerpos<<<n / p, p, compartida>>>(d_pos[1], d_pos[0], d_vel,
                                            numTiles);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("ERROR de CUDA: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaMemcpy(h_nueva, d_vel, bytes, cudaMemcpyDeviceToHost);

  // se comparan las velocidades: tras un paso desde el reposo, las posiciones
  // se mueven ~1e-6, por debajo de la precisión de un float cercano a 1
  int errores = 0;
  for (int i = 0; i < n; i += n / 16) {
    double nueva[3];
    pasoHost(i, h_pos, h_vel, n, nueva);
    double vg[3] = {h_nueva[i].x, h_nueva[i].y, h_nueva[i].z};
    double dif = 0.0, norma = 0.0;
    for (int k = 0; k < 3; k++) {
      dif += (vg[k] - nueva[k]) * (vg[k] - nueva[k]);
      norma += nueva[k] * nueva[k];
    }
    if (sqrt(dif) > 1e-3 * sqrt(norma))
      errores++;
  }
  printf("verificación (16 cuerpos contra el host): %s\n",
         errores ? "ERROR" : "correcta");

  // medir 'pasos' pasos, alternando los dos arreglos de posiciones
  cudaEvent_t inicio, fin;
  cudaEventCreate(&inicio);
  cudaEventCreate(&fin);
  int actual = 1;
  cudaEventRecord(inicio);
  for (int s = 0; s < pasos; s++) {
    integrarCuerpos<<<n / p, p, compartida>>>(d_pos[1 - actual], d_pos[actual],
                                              d_vel, numTiles);
    actual = 1 - actual;
  }
  cudaEventRecord(fin);
  cudaEventSynchronize(fin);
  float ms;
  cudaEventElapsedTime(&ms, inicio, fin);

  double ms_paso = ms / pasos;
  double interacciones = (double)n * n / (ms_paso * 1e-3);
  printf("%d pasos: %.3f ms por paso\n", pasos, ms_paso);
  printf("%.2f mil millones de interacciones por segundo, %.1f GFLOP/s\n",
         interacciones * 1e-9, interacciones * FLOPS_POR_INTERACCION * 1e-9);

  cudaEventDestroy(inicio);
  cudaEventDestroy(fin);
  cudaFree(d_pos[0]);
  cudaFree(d_pos[1]);
  cudaFree(d_vel);
  free(h_pos);
  free(h_vel);
  free(h_nueva);
  return 0;
}
