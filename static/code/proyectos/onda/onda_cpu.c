/*
 * Proyecto: ecuación de onda en 2D, membrana fija (versión CPU de referencia)
 *
 * Resuelve  u_tt = c^2 (u_xx + u_yy)  en el cuadrado [0,1]x[0,1] con u = 0 en
 * el borde, usando el esquema leapfrog (diferencias centradas en t y en x):
 *
 *   u_sig = 2 u - u_ant + r^2 (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]
 *                              - 4 u[i][j]),        r = c dt / h
 *
 * El esquema es estable si r <= 1/sqrt(2); aquí se usa r = 0.5.
 *
 * Condición inicial: el modo normal (1,1), u(x,y,0) = sin(pi x) sin(pi y),
 * con velocidad inicial cero. La solución exacta es
 *   u(x,y,t) = cos(omega t) sin(pi x) sin(pi y),   omega = sqrt(2) pi c.
 * Por defecto se simula un período y un cuarto: ahí la solución exacta es
 * u = 0 y cambia lo más rápido posible, así que un error en la frecuencia se
 * nota entero en el error (al cabo de un período justo casi no se nota).
 *
 * Además se mide la energía discreta (cinética + elástica), que el esquema
 * leapfrog conserva hasta el error de redondeo.
 *
 * Compilar:  gcc -O2 onda_cpu.c -lm -o onda_cpu.x
 * Ejecutar:  ./onda_cpu.x [N] [pasos]
 *            N: puntos por lado, incluyendo el borde (por defecto 2048)
 *            pasos: número de pasos de tiempo (por defecto, los que cubren
 *                   1.25 períodos con r ~ 0.5); si se da, se usa r = 0.5
 *                   justo y el tiempo final cambia
 *
 * Salida: tiempo de cómputo del ciclo principal, error respecto a la solución
 * exacta, variación de la energía y el campo final en salida_cpu.bin
 * (formato en escribir_salida).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

// Cambiar a float es una optimización posible: hay que justificarla con el
// error y la conservación de la energía
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

// Primer paso (velocidad inicial cero): u1 = u0 + (r^2 / 2) * laplaciano
void primer_paso(const real *u0, real *u1, int N, real r2) {
  for (int j = 1; j < N - 1; j++)
    for (int i = 1; i < N - 1; i++) {
      int idx = j * N + i;
      u1[idx] = u0[idx] + 0.5 * r2 *
                              (u0[idx + 1] + u0[idx - 1] + u0[idx + N] +
                               u0[idx - N] - 4.0 * u0[idx]);
    }
}

// Un paso leapfrog sobre los puntos interiores; el borde queda en cero
void paso_onda(const real *u_ant, const real *u, real *u_sig, int N, real r2) {
  for (int j = 1; j < N - 1; j++)
    for (int i = 1; i < N - 1; i++) {
      int idx = j * N + i;
      u_sig[idx] = 2.0 * u[idx] - u_ant[idx] +
                   r2 * (u[idx + 1] + u[idx - 1] + u[idx + N] + u[idx - N] -
                         4.0 * u[idx]);
    }
}

// Energía discreta entre los niveles n y n+1, conservada por leapfrog:
// E = (h^2/2) sum [ ((u1 - u0)/dt)^2 + c^2 grad(u1) . grad(u0) ]
real energia(const real *u0, const real *u1, int N, real h, real dt, real c) {
  real cinetica = 0.0, elastica = 0.0;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      int idx = j * N + i;
      real v = (u1[idx] - u0[idx]) / dt;
      cinetica += v * v;
      if (i < N - 1)
        elastica += (u1[idx + 1] - u1[idx]) * (u0[idx + 1] - u0[idx]);
      if (j < N - 1)
        elastica += (u1[idx + N] - u1[idx]) * (u0[idx + N] - u0[idx]);
    }
  return 0.5 * h * h * (cinetica + c * c * elastica / (h * h));
}

// Error máximo respecto a la solución exacta en el tiempo t
real error_maximo(const real *u, int N, real h, real omega, real t) {
  real err = 0.0;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      real exacta = cos(omega * t) * sin(PI * i * h) * sin(PI * j * h);
      real e = fabs(u[j * N + i] - exacta);
      if (e > err)
        err = e;
    }
  return err;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 2048;

  real c = 1.0;
  real h = 1.0 / (N - 1);
  real omega = sqrt(2.0) * PI * c;
  real periodo = 2.0 * PI / omega;
  real t_final = 1.25 * periodo;

  // Por defecto, un número entero de pasos que llega a t_final con r ~ 0.5
  int pasos = (argc > 2) ? atoi(argv[2]) : (int)ceil(t_final / (0.5 * h / c));
  real dt = (argc > 2) ? 0.5 * h / c : t_final / pasos;
  if (N < 3 || pasos < 1) {
    fprintf(stderr, "Se necesita N >= 3 y pasos >= 1\n");
    return 1;
  }
  real r = c * dt / h;
  real r2 = r * r;

  size_t bytes = (size_t)N * N * sizeof(real);
  real *u_ant = (real *)calloc((size_t)N * N, sizeof(real));
  real *u = (real *)calloc((size_t)N * N, sizeof(real));
  real *u_sig = (real *)calloc((size_t)N * N, sizeof(real));

  condicion_inicial(u_ant, N, h);
  primer_paso(u_ant, u, N, r2);

  printf("Ecuación de onda 2D: N = %d, pasos = %d, dt = %.3e, r = %.4f\n", N,
         pasos, dt, r);
  printf("Memoria por arreglo: %.1f MB\n", bytes / 1e6);

  real E0 = energia(u_ant, u, N, h, dt, c);

  // El primer paso ya está hecho, faltan pasos - 1
  double t0 = segundos();
  for (int n = 1; n < pasos; n++) {
    paso_onda(u_ant, u, u_sig, N, r2);
    // Rotación de punteros: u_ant <- u <- u_sig
    real *tmp = u_ant;
    u_ant = u;
    u = u_sig;
    u_sig = tmp;
  }
  double t_total = segundos() - t0;

  real E1 = energia(u_ant, u, N, h, dt, c);
  real t = pasos * dt;

  printf("Tiempo de cómputo: %.3f s (%.3f ms por paso)\n", t_total,
         1e3 * t_total / pasos);
  printf("Tiempo físico t = %.6f (período = %.6f)\n", t, periodo);
  printf("Error máximo vs. solución exacta: %.3e\n",
         error_maximo(u, N, h, omega, t));
  // Referencia: un código que no actualiza u tendría este error
  printf("(sin evolucionar, el error sería %.3e)\n",
         fabs(1.0 - cos(omega * t)));
  printf("Energía: inicial = %.12e, final = %.12e, variación relativa = %.3e\n",
         E0, E1, fabs(E1 - E0) / E0);

  real *campos[1] = {u};
  escribir_salida("salida_cpu.bin", N, N, 1, campos);

  free(u_ant);
  free(u);
  free(u_sig);
  return 0;
}
