/* N-cuerpos en la CPU: el mismo cálculo que nbody-colab.cu, para comparar.
 *
 * Usa los mismos cuerpos iniciales (misma semilla), el mismo softening y el
 * mismo paso leapfrog, en float. Para que la comparación con el GPU sea justa
 * no es un código ingenuo:
 *   - guarda cada coordenada en su propio arreglo (SoA), para que el
 *     compilador vectorice el lazo interno con instrucciones SIMD;
 *   - reparte los cuerpos entre los cores con OpenMP.
 * Imprime lo mismo que nbody-colab.cu: cuánto dura un paso, cuántas
 * interacciones por segundo calcula (N^2 por paso) y los GFLOP/s, contando
 * 20 operaciones por interacción. Antes de medir compara las velocidades tras
 * un paso con el mismo paso en double.
 *
 * Compilar:  gcc -O3 -march=native -ffast-math -fopenmp nbody-cpu.c \
 *                -o nbody-cpu.x -lm
 * Ejecutar:  ./nbody-cpu.x [N, por defecto 16384] [pasos, por defecto 5]
 * -ffast-math deja vectorizar las sumas y calcular 1/sqrt rápido (como rsqrtf
 * en el GPU). Sin -fopenmp también compila, pero corre en un solo core.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define EPS2 0.01f // softening al cuadrado
#define DT 0.001f  // paso de tiempo
#define FLOPS_POR_INTERACCION 20
#define NVERIF 16 // cuerpos que se comparan contra double

typedef struct {
  float *x, *y, *z, *m;    // posiciones y masas
  float *vx, *vy, *vz;     // velocidades
  float *ax, *ay, *az;     // aceleraciones
} Cuerpos;

// Aceleración de cada cuerpo (G = 1): cada thread de OpenMP toma un grupo de
// cuerpos i, y el lazo en j es el que se vectoriza.
void aceleraciones(Cuerpos *c, int n) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    float xi = c->x[i], yi = c->y[i], zi = c->z[i];
    float axi = 0.0f, ayi = 0.0f, azi = 0.0f;
    for (int j = 0; j < n; j++) {
      float rx = c->x[j] - xi;
      float ry = c->y[j] - yi;
      float rz = c->z[j] - zi;
      float distSqr = rx * rx + ry * ry + rz * rz + EPS2;
      float invDist = 1.0f / sqrtf(distSqr);
      float s = c->m[j] * invDist * invDist * invDist;
      axi += rx * s;
      ayi += ry * s;
      azi += rz * s;
    }
    c->ax[i] = axi;
    c->ay[i] = ayi;
    c->az[i] = azi;
  }
}

// Un paso leapfrog: v += a dt, x += v dt. Primero se calculan todas las
// aceleraciones, así que nadie lee una posición ya actualizada.
void paso(Cuerpos *c, int n) {
  aceleraciones(c, n);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    c->vx[i] += c->ax[i] * DT;
    c->vy[i] += c->ay[i] * DT;
    c->vz[i] += c->az[i] * DT;
    c->x[i] += c->vx[i] * DT;
    c->y[i] += c->vy[i] * DT;
    c->z[i] += c->vz[i] * DT;
  }
}

// La velocidad del cuerpo i tras un paso, en double
void pasoDouble(int i, const Cuerpos *c, int n, double nueva[3]) {
  double a[3] = {0.0, 0.0, 0.0};
  for (int j = 0; j < n; j++) {
    double r[3] = {(double)c->x[j] - c->x[i], (double)c->y[j] - c->y[i],
                   (double)c->z[j] - c->z[i]};
    double d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + EPS2;
    double s = c->m[j] / (d2 * sqrt(d2));
    for (int k = 0; k < 3; k++)
      a[k] += r[k] * s;
  }
  double v[3] = {c->vx[i], c->vy[i], c->vz[i]};
  for (int k = 0; k < 3; k++)
    nueva[k] = v[k] + a[k] * DT;
}

double segundos(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char *argv[]) {
  int n = (argc > 1) ? atoi(argv[1]) : 16384;
  int pasos = (argc > 2) ? atoi(argv[2]) : 5;
  if (n < NVERIF || pasos < 1) {
    printf("ERROR: se necesita N >= %d y pasos >= 1\n", NVERIF);
    return 1;
  }
  int threads = 1;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif

  Cuerpos c;
  float **arreglos[] = {&c.x,  &c.y,  &c.z,  &c.m,  &c.vx,
                        &c.vy, &c.vz, &c.ax, &c.ay, &c.az};
  for (int k = 0; k < 10; k++)
    *arreglos[k] = (float *)malloc(n * sizeof(float));

  // cuerpos al azar dentro de una esfera de radio 1, en reposo, masa total 1
  // (la misma secuencia de rand() que nbody-colab.cu)
  srand(2019);
  for (int i = 0; i < n; i++) {
    float x, y, z;
    do {
      x = 2.0f * rand() / RAND_MAX - 1.0f;
      y = 2.0f * rand() / RAND_MAX - 1.0f;
      z = 2.0f * rand() / RAND_MAX - 1.0f;
    } while (x * x + y * y + z * z > 1.0f);
    c.x[i] = x;
    c.y[i] = y;
    c.z[i] = z;
    c.m[i] = 1.0f / n;
    c.vx[i] = c.vy[i] = c.vz[i] = 0.0f;
  }

  printf("N = %d cuerpos, threads de OpenMP: %d\n", n, threads);

  // la referencia en double se calcula antes del paso, con las posiciones
  // iniciales; después del paso se comparan las velocidades
  double esperada[NVERIF][3];
  for (int k = 0; k < NVERIF; k++)
    pasoDouble(k * (n / NVERIF), &c, n, esperada[k]);
  paso(&c, n); // también calienta la caché y los threads
  int errores = 0;
  for (int k = 0; k < NVERIF; k++) {
    int i = k * (n / NVERIF);
    double v[3] = {c.vx[i], c.vy[i], c.vz[i]};
    double dif = 0.0, norma = 0.0;
    for (int d = 0; d < 3; d++) {
      dif += (v[d] - esperada[k][d]) * (v[d] - esperada[k][d]);
      norma += esperada[k][d] * esperada[k][d];
    }
    if (sqrt(dif) > 1e-3 * sqrt(norma))
      errores++;
  }
  printf("verificación (%d cuerpos contra double): %s\n", NVERIF,
         errores ? "ERROR" : "correcta");

  double t0 = segundos();
  for (int s = 0; s < pasos; s++)
    paso(&c, n);
  double ms_paso = 1e3 * (segundos() - t0) / pasos;

  double interacciones = (double)n * n / (ms_paso * 1e-3);
  printf("%d pasos: %.3f ms por paso\n", pasos, ms_paso);
  printf("%.2f mil millones de interacciones por segundo, %.1f GFLOP/s\n",
         interacciones * 1e-9, interacciones * FLOPS_POR_INTERACCION * 1e-9);

  for (int k = 0; k < 10; k++)
    free(*arreglos[k]);
  return 0;
}
