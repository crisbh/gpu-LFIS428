/*
 * Proyecto: ecuación de Poisson en 2D (versión CPU de referencia)
 *
 * Resuelve  u_xx + u_yy = f  en el cuadrado [0,1]x[0,1] con u = 0 en el borde.
 * Con el laplaciano de 5 puntos, cada punto interior debe cumplir
 *
 *   u[i][j] = (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - h^2 f[i][j]) / 4
 *
 * y el sistema se resuelve iterando esta relación con uno de tres métodos:
 *   0: Jacobi            (usa sólo valores de la iteración anterior)
 *   1: Gauss-Seidel rojo-negro (actualiza primero los puntos con i+j par y
 *      luego los impares, que sólo dependen de los pares)
 *   2: SOR rojo-negro    (Gauss-Seidel con sobrerrelajación,
 *      w = 2 / (1 + sin(pi h)))
 *
 * Fuente: f = -2 pi^2 sin(pi x) sin(pi y), cuya solución exacta es
 *   u(x,y) = sin(pi x) sin(pi y).
 * El error respecto a la solución exacta debe ser O(h^2): al duplicar N
 * debe bajar ~4 veces.
 *
 * Se itera hasta que la norma del residuo r = f - lap(u) cae bajo
 * tol * ||f||. El residuo se calcula cada 'chequeo' iteraciones porque es
 * una reducción sobre toda la malla.
 *
 * Compilar:  gcc -O2 poisson_cpu.c -lm -o poisson_cpu.x
 * Ejecutar:  ./poisson_cpu.x [N] [metodo] [tol]
 *            N: puntos por lado, incluyendo el borde (por defecto 1024)
 *            metodo: 0 Jacobi, 1 Gauss-Seidel rojo-negro, 2 SOR rojo-negro
 *                    (por defecto 2)
 *            tol: tolerancia relativa del residuo (por defecto 1e-8)
 *
 * Ojo: Jacobi necesita O(N^2) iteraciones; con N grande puede tardar mucho.
 *
 * Salida: iteraciones, tiempo de cómputo del ciclo principal (incluido el
 * cálculo del residuo), error respecto a la solución exacta y el campo final
 * en salida_cpu.bin (formato en escribir_salida).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

// Cambiar a float es una optimización posible, pero limita la tolerancia
// alcanzable: hay que justificarla
typedef double real;

double segundos(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Escribe una cabecera (nx, ny, ncampos, bytes por real) seguida de los
// campos uno tras otro, cada uno con nx*ny valores en orden [y][x]
void escribir_salida(const char *nombre, int nx, int ny, int ncampos,
                     real **campos) {
  FILE *f = fopen(nombre, "wb");
  int cabecera[4] = {nx, ny, ncampos, (int)sizeof(real)};
  fwrite(cabecera, sizeof(int), 4, f);
  for (int c = 0; c < ncampos; c++)
    fwrite(campos[c], sizeof(real), (size_t)nx * ny, f);
  fclose(f);
}

void inicializar_fuente(real *f, int N, real h) {
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      f[j * N + i] = -2.0 * PI * PI * sin(PI * i * h) * sin(PI * j * h);
}

// Una iteración de Jacobi: lee u, escribe u_new
void iteracion_jacobi(const real *u, real *u_new, const real *f, int N,
                      real h2) {
  for (int j = 1; j < N - 1; j++)
    for (int i = 1; i < N - 1; i++) {
      int idx = j * N + i;
      u_new[idx] =
          0.25 * (u[idx + 1] + u[idx - 1] + u[idx + N] + u[idx - N] -
                  h2 * f[idx]);
    }
}

// Media iteración rojo-negro in situ sobre los puntos con (i + j) % 2 == color.
// Con w = 1 es Gauss-Seidel; con 1 < w < 2 es SOR
void media_iteracion_rb(real *u, const real *f, int N, real h2, real w,
                        int color) {
  for (int j = 1; j < N - 1; j++)
    for (int i = 1 + (j + 1 + color) % 2; i < N - 1; i += 2) {
      int idx = j * N + i;
      real gs = 0.25 * (u[idx + 1] + u[idx - 1] + u[idx + N] + u[idx - N] -
                        h2 * f[idx]);
      u[idx] += w * (gs - u[idx]);
    }
}

// Norma L2 del residuo r = f - lap(u) sobre los puntos interiores
real norma_residuo(const real *u, const real *f, int N, real h2) {
  real suma = 0.0;
  for (int j = 1; j < N - 1; j++)
    for (int i = 1; i < N - 1; i++) {
      int idx = j * N + i;
      real lap =
          (u[idx + 1] + u[idx - 1] + u[idx + N] + u[idx - N] - 4.0 * u[idx]) /
          h2;
      real r = f[idx] - lap;
      suma += r * r;
    }
  return sqrt(suma);
}

real norma(const real *f, int N) {
  real suma = 0.0;
  for (int k = 0; k < N * N; k++)
    suma += f[k] * f[k];
  return sqrt(suma);
}

real error_maximo(const real *u, int N, real h) {
  real err = 0.0;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      real e = fabs(u[j * N + i] - sin(PI * i * h) * sin(PI * j * h));
      if (e > err)
        err = e;
    }
  return err;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  int metodo = (argc > 2) ? atoi(argv[2]) : 2;
  real tol = (argc > 3) ? atof(argv[3]) : 1e-8;

  const char *nombres[3] = {"Jacobi", "Gauss-Seidel rojo-negro",
                            "SOR rojo-negro"};
  if (metodo < 0 || metodo > 2) {
    fprintf(stderr, "metodo debe ser 0, 1 o 2\n");
    return 1;
  }
  if (N < 3) {
    fprintf(stderr, "Se necesita N >= 3\n");
    return 1;
  }

  real h = 1.0 / (N - 1);
  real h2 = h * h;
  real w = (metodo == 2) ? 2.0 / (1.0 + sin(PI * h)) : 1.0;
  int chequeo = 100;
  int max_iter = 10000000;

  real *u = (real *)calloc((size_t)N * N, sizeof(real));
  real *u_new = (real *)calloc((size_t)N * N, sizeof(real));
  real *f = (real *)malloc((size_t)N * N * sizeof(real));
  inicializar_fuente(f, N, h);
  real norma_f = norma(f, N);

  printf("Poisson 2D: N = %d, método = %s", N, nombres[metodo]);
  if (metodo == 2)
    printf(" (w = %.6f)", w);
  printf(", tol = %.1e\n", tol);

  double t0 = segundos();
  int iter = 0;
  real res = norma_f;
  while (iter < max_iter) {
    for (int k = 0; k < chequeo; k++) {
      if (metodo == 0) {
        iteracion_jacobi(u, u_new, f, N, h2);
        real *tmp = u;
        u = u_new;
        u_new = tmp;
      } else {
        media_iteracion_rb(u, f, N, h2, w, 0);
        media_iteracion_rb(u, f, N, h2, w, 1);
      }
    }
    iter += chequeo;
    res = norma_residuo(u, f, N, h2) / norma_f;
    if (res < tol)
      break;
  }
  double t_total = segundos() - t0;

  printf("Iteraciones: %d, residuo relativo = %.3e\n", iter, res);
  printf("Tiempo de cómputo: %.3f s (%.3f ms por iteración)\n", t_total,
         1e3 * t_total / iter);
  printf("Error máximo vs. solución exacta: %.3e (h^2 = %.3e)\n",
         error_maximo(u, N, h), h2);

  real *campos[1] = {u};
  escribir_salida("salida_cpu.bin", N, N, 1, campos);

  free(u);
  free(u_new);
  free(f);
  return 0;
}
