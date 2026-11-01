/*
 * Proyecto: reacción-difusión de Gray-Scott en 2D (versión CPU de referencia)
 *
 * Dos especies químicas U y V reaccionan (U + 2V -> 3V) y difunden:
 *
 *   u_t = Du lap(u) - u v^2 + F (1 - u)
 *   v_t = Dv lap(v) + u v^2 - (F + k) v
 *
 * en una malla N x N con condiciones de borde periódicas, usando el laplaciano
 * de 5 puntos (dx = 1) y Euler explícito (dt = 1). Según F y k aparecen
 * manchas, laberintos o manchas que se dividen ("mitosis").
 *
 * Condición inicial: u = 1, v = 0, más varias semillas cuadradas con
 * u = 0.5, v = 0.25 y un poco de ruido. Los números aleatorios se generan en
 * el host con una semilla fija, así que la condición inicial es reproducible.
 *
 * No hay solución exacta: para validar la versión GPU se compara el campo
 * final con salida_cpu.bin (para pocos pasos debe coincidir casi exactamente)
 * y, para corridas largas, los promedios <u> y <v> que se imprimen.
 *
 * Compilar:  gcc -O2 reaccion-difusion_cpu.c -lm -o reaccion-difusion_cpu.x
 * Ejecutar:  ./reaccion-difusion_cpu.x [N] [pasos] [F] [k]
 *            N: puntos por lado (por defecto 1024)
 *            pasos: número de pasos de tiempo (por defecto 10000)
 *            F, k: parámetros de la reacción (por defecto 0.0367, 0.0649)
 *
 * Salida: tiempo de cómputo, promedios de u y v y los campos finales u, v en
 * salida_cpu.bin (ver comparar.py y graficar.py).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Cambiar a float es una optimización posible: hay que comprobar que los
// patrones y los promedios no cambian apreciablemente
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

// Generador congruencial simple con semilla fija: uniforme en [0, 1)
unsigned long long estado_rng = 12345ULL;
double aleatorio(void) {
  estado_rng = estado_rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return (estado_rng >> 11) * (1.0 / 9007199254740992.0);
}

void condicion_inicial(real *u, real *v, int N) {
  for (int k = 0; k < N * N; k++) {
    u[k] = 1.0;
    v[k] = 0.0;
  }
  // Una semilla de 20 x 20 por cada ~ 100 x 100 puntos
  int semillas = N * N / 10000 + 1;
  for (int s = 0; s < semillas; s++) {
    int x0 = (int)(aleatorio() * N);
    int y0 = (int)(aleatorio() * N);
    for (int dy = 0; dy < 20; dy++)
      for (int dx = 0; dx < 20; dx++) {
        int idx = ((y0 + dy) % N) * N + (x0 + dx) % N;
        u[idx] = 0.5;
        v[idx] = 0.25;
      }
  }
  for (int k = 0; k < N * N; k++) {
    u[k] += 0.02 * (aleatorio() - 0.5);
    v[k] += 0.02 * (aleatorio() - 0.5);
  }
}

// Un paso de Euler explícito con bordes periódicos
void paso_gray_scott(const real *u, const real *v, real *u_new, real *v_new,
                     int N, real Du, real Dv, real F, real k) {
  for (int j = 0; j < N; j++) {
    int jm = (j - 1 + N) % N, jp = (j + 1) % N;
    for (int i = 0; i < N; i++) {
      int im = (i - 1 + N) % N, ip = (i + 1) % N;
      int idx = j * N + i;
      real lap_u = u[j * N + ip] + u[j * N + im] + u[jp * N + i] +
                   u[jm * N + i] - 4.0 * u[idx];
      real lap_v = v[j * N + ip] + v[j * N + im] + v[jp * N + i] +
                   v[jm * N + i] - 4.0 * v[idx];
      real uvv = u[idx] * v[idx] * v[idx];
      u_new[idx] = u[idx] + Du * lap_u - uvv + F * (1.0 - u[idx]);
      v_new[idx] = v[idx] + Dv * lap_v + uvv - (F + k) * v[idx];
    }
  }
}

real promedio(const real *a, int N) {
  real suma = 0.0;
  for (int k = 0; k < N * N; k++)
    suma += a[k];
  return suma / ((real)N * N);
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  int pasos = (argc > 2) ? atoi(argv[2]) : 10000;
  real F = (argc > 3) ? atof(argv[3]) : 0.0367;
  real k = (argc > 4) ? atof(argv[4]) : 0.0649;
  real Du = 0.16, Dv = 0.08;

  size_t bytes = (size_t)N * N * sizeof(real);
  real *u = (real *)malloc(bytes);
  real *v = (real *)malloc(bytes);
  real *u_new = (real *)malloc(bytes);
  real *v_new = (real *)malloc(bytes);

  condicion_inicial(u, v, N);

  printf("Gray-Scott 2D: N = %d, pasos = %d, F = %.4f, k = %.4f\n", N, pasos,
         F, k);

  double t0 = segundos();
  for (int n = 0; n < pasos; n++) {
    paso_gray_scott(u, v, u_new, v_new, N, Du, Dv, F, k);
    real *tmp = u;
    u = u_new;
    u_new = tmp;
    tmp = v;
    v = v_new;
    v_new = tmp;
  }
  double t_total = segundos() - t0;

  printf("Tiempo de cómputo: %.3f s (%.3f ms por paso)\n", t_total,
         1e3 * t_total / pasos);
  printf("Promedios finales: <u> = %.10f, <v> = %.10f\n", promedio(u, N),
         promedio(v, N));

  real *campos[2] = {u, v};
  escribir_salida("salida_cpu.bin", N, N, 2, campos);

  free(u);
  free(v);
  free(u_new);
  free(v_new);
  return 0;
}
