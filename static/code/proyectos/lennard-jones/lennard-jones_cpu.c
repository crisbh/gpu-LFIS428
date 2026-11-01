/*
 * Proyecto: dinámica molecular 2D con potencial de Lennard-Jones
 * (versión CPU de referencia)
 *
 * N partículas en una caja cuadrada de lado Lc con bordes periódicos
 * interactúan con el potencial de Lennard-Jones (unidades reducidas,
 * epsilon = sigma = m = 1), cortado en r_c = 2.5 y desplazado para que valga
 * cero en el corte:
 *
 *   U(r) = 4 (r^-12 - r^-6) - U(r_c),   F(r) = 24 (2 r^-14 - r^-8) r_vec
 *
 * Como la interacción tiene alcance finito, no hace falta revisar los N^2
 * pares (a diferencia del ejemplo de N-cuerpos del curso): la caja se divide
 * en celdas de lado >= r_c y cada partícula sólo mira las partículas de su
 * celda y de las 8 vecinas. Las celdas se guardan como listas enlazadas
 * (primera[celda], siguiente[partícula]) y se reconstruyen en cada paso.
 *
 * Integración con velocity Verlet:
 *   v += (dt/2) F;  x += dt v;  recalcular celdas y F;  v += (dt/2) F
 *
 * Condición inicial: red cuadrada con densidad rho y velocidades aleatorias
 * escaladas a la temperatura T0, con momento total cero.
 *
 * Validación: la energía total (cinética + potencial) debe conservarse; su
 * variación relativa debe ser pequeña y no crecer sistemáticamente. Como la
 * dinámica es caótica, las posiciones de la versión GPU sólo coincidirán con
 * las de CPU durante pocos pasos: para corridas largas se comparan la energía
 * y la temperatura.
 *
 * Compilar:  gcc -O2 lennard-jones_cpu.c -lm -o lennard-jones_cpu.x
 * Ejecutar:  ./lennard-jones_cpu.x [n] [pasos] [rho] [T0]
 *            n: partículas por lado de la red inicial, N = n^2 (por defecto
 *               256)
 *            pasos: pasos de tiempo con dt = 0.005 (por defecto 1000)
 *            rho: densidad (por defecto 0.6)
 *            T0: temperatura inicial (por defecto 1.0)
 *
 * Salida: energía y temperatura durante la simulación, tiempo de cómputo y
 * los campos x, y, vx, vy (nx = N, ny = 1) en salida_cpu.bin (ver
 * comparar.py y graficar.py).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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

// Reconstruye las listas de celdas: primera[c] es una partícula de la celda c
// (o -1) y siguiente[i] la próxima partícula en la misma celda (o -1)
void construir_celdas(const real *x, const real *y, int N, int ncel, real lcel,
                      int *primera, int *siguiente) {
  for (int c = 0; c < ncel * ncel; c++)
    primera[c] = -1;
  for (int i = 0; i < N; i++) {
    int ci = (int)(x[i] / lcel) % ncel;
    int cj = (int)(y[i] / lcel) % ncel;
    int c = cj * ncel + ci;
    siguiente[i] = primera[c];
    primera[c] = i;
  }
}

// Fuerzas sobre cada partícula y energía potencial total. Cada partícula
// recorre su celda y las 8 vecinas; cada par se visita dos veces (una desde
// cada partícula), así que la energía se divide por 2
real fuerzas(const real *x, const real *y, real *fx, real *fy, int N, real Lc,
             int ncel, real lcel, const int *primera, const int *siguiente,
             real rc2, real U_rc) {
  real U = 0.0;
  for (int i = 0; i < N; i++) {
    real fxi = 0.0, fyi = 0.0;
    int ci = (int)(x[i] / lcel) % ncel;
    int cj = (int)(y[i] / lcel) % ncel;
    for (int dj = -1; dj <= 1; dj++)
      for (int di = -1; di <= 1; di++) {
        int c = ((cj + dj + ncel) % ncel) * ncel + (ci + di + ncel) % ncel;
        for (int j = primera[c]; j != -1; j = siguiente[j]) {
          if (j == i)
            continue;
          // Imagen mínima
          real dx = x[i] - x[j], dy = y[i] - y[j];
          dx -= Lc * round(dx / Lc);
          dy -= Lc * round(dy / Lc);
          real r2 = dx * dx + dy * dy;
          if (r2 < rc2) {
            real ir2 = 1.0 / r2;
            real ir6 = ir2 * ir2 * ir2;
            real f = 24.0 * ir2 * ir6 * (2.0 * ir6 - 1.0);
            fxi += f * dx;
            fyi += f * dy;
            U += 0.5 * (4.0 * ir6 * (ir6 - 1.0) - U_rc);
          }
        }
      }
    fx[i] = fxi;
    fy[i] = fyi;
  }
  return U;
}

real energia_cinetica(const real *vx, const real *vy, int N) {
  real K = 0.0;
  for (int i = 0; i < N; i++)
    K += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
  return K;
}

int main(int argc, char **argv) {
  int n = (argc > 1) ? atoi(argv[1]) : 256;
  int pasos = (argc > 2) ? atoi(argv[2]) : 1000;
  real rho = (argc > 3) ? atof(argv[3]) : 0.6;
  real T0 = (argc > 4) ? atof(argv[4]) : 1.0;

  int N = n * n;
  real dt = 0.005;
  real rc = 2.5, rc2 = rc * rc;
  real U_rc = 4.0 * (pow(rc, -12) - pow(rc, -6));
  real Lc = n / sqrt(rho);
  int ncel = (int)(Lc / rc);
  real lcel = Lc / ncel;
  if (ncel < 3) {
    fprintf(stderr, "La caja es muy pequeña: se necesitan al menos 3 celdas "
                    "por lado (aumentar n)\n");
    return 1;
  }
  int muestras = 10;

  real *x = (real *)malloc(N * sizeof(real));
  real *y = (real *)malloc(N * sizeof(real));
  real *vx = (real *)malloc(N * sizeof(real));
  real *vy = (real *)malloc(N * sizeof(real));
  real *fx = (real *)malloc(N * sizeof(real));
  real *fy = (real *)malloc(N * sizeof(real));
  int *primera = (int *)malloc(ncel * ncel * sizeof(int));
  int *siguiente = (int *)malloc(N * sizeof(int));

  // Red cuadrada y velocidades aleatorias con momento total cero
  real a = Lc / n;
  real px = 0.0, py = 0.0;
  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++) {
      int k = j * n + i;
      x[k] = (i + 0.5) * a;
      y[k] = (j + 0.5) * a;
      vx[k] = aleatorio() - 0.5;
      vy[k] = aleatorio() - 0.5;
      px += vx[k];
      py += vy[k];
    }
  for (int k = 0; k < N; k++) {
    vx[k] -= px / N;
    vy[k] -= py / N;
  }
  // En 2D: K = N T, así que T = K / N
  real escala = sqrt(T0 / (energia_cinetica(vx, vy, N) / N));
  for (int k = 0; k < N; k++) {
    vx[k] *= escala;
    vy[k] *= escala;
  }

  printf("Lennard-Jones 2D: N = %d, caja = %.2f, %d x %d celdas, pasos = %d\n",
         N, Lc, ncel, ncel, pasos);
  printf("%7s %16s %16s %10s\n", "paso", "E total", "U", "T");

  construir_celdas(x, y, N, ncel, lcel, primera, siguiente);
  real U = fuerzas(x, y, fx, fy, N, Lc, ncel, lcel, primera, siguiente, rc2,
                   U_rc);
  real K = energia_cinetica(vx, vy, N);
  real E0 = K + U;
  printf("%7d %16.8f %16.8f %10.6f\n", 0, E0 / N, U / N, K / N);

  double t0 = segundos();
  int cada = (pasos >= muestras) ? pasos / muestras : 1;
  for (int p = 1; p <= pasos; p++) {
    for (int i = 0; i < N; i++) {
      vx[i] += 0.5 * dt * fx[i];
      vy[i] += 0.5 * dt * fy[i];
      x[i] += dt * vx[i];
      y[i] += dt * vy[i];
      // Bordes periódicos: devolver a [0, Lc)
      x[i] -= Lc * floor(x[i] / Lc);
      y[i] -= Lc * floor(y[i] / Lc);
    }
    construir_celdas(x, y, N, ncel, lcel, primera, siguiente);
    U = fuerzas(x, y, fx, fy, N, Lc, ncel, lcel, primera, siguiente, rc2,
                U_rc);
    for (int i = 0; i < N; i++) {
      vx[i] += 0.5 * dt * fx[i];
      vy[i] += 0.5 * dt * fy[i];
    }

    if (p % cada == 0 || p == pasos) {
      K = energia_cinetica(vx, vy, N);
      printf("%7d %16.8f %16.8f %10.6f\n", p, (K + U) / N, U / N, K / N);
    }
  }
  double t_total = segundos() - t0;

  K = energia_cinetica(vx, vy, N);
  printf("Tiempo de cómputo: %.3f s (%.3f ms por paso)\n", t_total,
         1e3 * t_total / pasos);
  printf("Variación relativa de la energía total: %.3e\n",
         fabs(K + U - E0) / fabs(E0));

  real *campos[4] = {x, y, vx, vy};
  escribir_salida("salida_cpu.bin", N, 1, 4, campos);

  free(x);
  free(y);
  free(vx);
  free(vy);
  free(fx);
  free(fy);
  free(primera);
  free(siguiente);
  return 0;
}
