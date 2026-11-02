/*
 * Proyecto: ecuación del calor en 2D (versión CPU de referencia)
 *
 * Resuelve  u_t = alpha (u_xx + u_yy)  en el cuadrado [0,1]x[0,1] con
 * u = 0 en el borde, usando diferencias finitas explícitas (FTCS):
 *
 *   u_new[i][j] = u[i][j] + r (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]
 *                              - 4 u[i][j]),        r = alpha dt / h^2
 *
 * El esquema es estable si r <= 1/4; aquí se usa r = 0.2.
 *
 * Condición inicial: u(x,y,0) = sin(pi x) sin(pi y), cuya solución exacta es
 *   u(x,y,t) = exp(-2 pi^2 alpha t) sin(pi x) sin(pi y),
 * que se usa para validar el resultado.
 *
 * Compilar:  gcc -O2 calor_cpu.c -lm -o calor_cpu.x
 * Ejecutar:  ./calor_cpu.x [N] [pasos]
 *            N: puntos por lado, incluyendo el borde (por defecto 2048)
 *            pasos: número de pasos de tiempo (por defecto 5000)
 *
 * Salida: tiempo de cómputo, error respecto a la solución exacta y el campo
 * final en salida_cpu.bin (ver comparar.py y graficar.py).
 *
 * Con los valores por defecto la placa apenas se enfría (t es muy pequeño),
 * así que para validar conviene una malla menor con más pasos, p. ej.
 * ./calor_cpu.x 256 10000, donde la amplitud baja a ~0.55.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

// Cambiar a float es una optimización posible: hay que justificarla con el
// error respecto a la solución exacta
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

void condicion_inicial(real *u, int N, real h) {
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      u[j * N + i] = sin(PI * i * h) * sin(PI * j * h);
}

// Un paso de FTCS sobre los puntos interiores; el borde queda en cero
void paso_calor(const real *u, real *u_new, int N, real r) {
  for (int j = 1; j < N - 1; j++)
    for (int i = 1; i < N - 1; i++) {
      int idx = j * N + i;
      u_new[idx] = u[idx] + r * (u[idx + 1] + u[idx - 1] + u[idx + N] +
                                 u[idx - N] - 4.0 * u[idx]);
    }
}

// Error máximo relativo respecto a la solución exacta en el tiempo t
real error_maximo(const real *u, int N, real h, real alpha, real t) {
  real decaimiento = exp(-2.0 * PI * PI * alpha * t);
  real err = 0.0;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      real exacta = decaimiento * sin(PI * i * h) * sin(PI * j * h);
      real e = fabs(u[j * N + i] - exacta);
      if (e > err)
        err = e;
    }
  return err / decaimiento;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 2048;
  int pasos = (argc > 2) ? atoi(argv[2]) : 5000;

  real alpha = 1.0;
  real h = 1.0 / (N - 1);
  real r = 0.2;
  real dt = r * h * h / alpha;

  size_t bytes = (size_t)N * N * sizeof(real);
  real *u = (real *)malloc(bytes);
  real *u_new = (real *)malloc(bytes);

  condicion_inicial(u, N, h);
  condicion_inicial(u_new, N, h); // para que el borde de u_new también sea 0

  printf("Ecuación del calor 2D: N = %d, pasos = %d, dt = %.3e\n", N, pasos,
         dt);

  double t0 = segundos();
  for (int n = 0; n < pasos; n++) {
    paso_calor(u, u_new, N, r);
    // Intercambio de punteros: evita copiar el arreglo completo
    real *tmp = u;
    u = u_new;
    u_new = tmp;
  }
  double t_total = segundos() - t0;

  real t = pasos * dt;
  printf("Tiempo de cómputo: %.3f s (%.3f ms por paso)\n", t_total,
         1e3 * t_total / pasos);
  printf("Tiempo físico t = %.4e, amplitud exacta = %.6f\n", t,
         exp(-2.0 * PI * PI * alpha * t));
  printf("Error máximo relativo vs. solución exacta: %.3e\n",
         error_maximo(u, N, h, alpha, t));
  // Referencia: un código que no actualiza u tendría este error
  printf("(sin evolucionar, el error sería %.3e)\n",
         (1.0 - exp(-2.0 * PI * PI * alpha * t)) /
             exp(-2.0 * PI * PI * alpha * t));

  real *campos[1] = {u};
  escribir_salida("salida_cpu.bin", N, N, 1, campos);

  free(u);
  free(u_new);
  return 0;
}
