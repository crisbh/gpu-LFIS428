/*
 * Proyecto: modelo de Ising 2D con Monte Carlo (versión CPU de referencia)
 *
 * Red cuadrada L x L de espines s = +-1 con condiciones de borde periódicas y
 * energía  E = -J sum_<ij> s_i s_j  (J = 1, k_B = 1, sin campo externo).
 *
 * Algoritmo de Metropolis con actualización en tablero de ajedrez: primero se
 * proponen volteos en todos los sitios "negros" ((i + j) par) y luego en los
 * "blancos". Los sitios de un mismo color no son vecinos entre sí, así que
 * pueden actualizarse en paralelo sin condiciones de carrera.
 *
 * Al voltear s_i el cambio de energía es dE = 2 s_i h_i, con h_i la suma de
 * los 4 vecinos; s_i h_i sólo toma los valores -4, -2, 0, 2, 4, así que las
 * probabilidades de aceptación exp(-dE / T) se precalculan en una tabla.
 *
 * Para cada temperatura se parte de todos los espines +1, se termaliza y luego
 * se mide <|m|>, <e>, el calor específico C y la susceptibilidad chi por
 * espín. Se comparan con la solución exacta de Onsager (red infinita):
 *   m(T) = (1 - sinh(2/T)^-4)^(1/8) para T < Tc = 2/ln(1 + sqrt(2)) ~ 2.269
 * y la energía exacta por espín, que usa una integral elíptica.
 * Cerca de Tc los efectos de tamaño finito son grandes: en una red finita
 * <|m|> no llega a 0 sobre Tc.
 *
 * Barras de error (método de bloques): las medidas se dividen en NBLOQUES
 * bloques consecutivos; el error estándar de <|m|> y <e> es la desviación
 * estándar de los promedios de los bloques dividida por sqrt(NBLOQUES - 1).
 * Medidas sucesivas están correlacionadas, y cerca de Tc el tiempo de
 * correlación crece mucho (ralentización crítica): ahí los bloques no son
 * independientes y el error queda subestimado.
 *
 * Compilar:  gcc -O2 ising_cpu.c -lm -o ising_cpu.x
 * Ejecutar:  ./ising_cpu.x [L] [T_min] [T_max] [nT] [term] [medidas] [semilla]
 *            L: lado de la red, par (por defecto 512)
 *            T_min, T_max, nT: barrido de temperaturas (por defecto 1.5, 3.5, 11)
 *            term: barridos de termalización por temperatura (por defecto 1000)
 *            medidas: barridos de medición por temperatura (por defecto 1000)
 *            semilla: semilla del generador (por defecto 12345)
 *
 * Salida: tabla de observables vs. T con sus errores, tiempo de cómputo de
 * todo el barrido de temperaturas (termalización y medidas) y la
 * configuración final (de la última temperatura) en salida_cpu.bin (formato
 * en escribir_salida). Como la versión GPU usará otros números aleatorios,
 * se valida comparando los observables dentro de sus barras de error, no la
 * configuración.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323846
#define NBLOQUES 20

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

// Generador congruencial simple: uniforme en [0, 1). La semilla se puede
// cambiar desde la línea de comandos.
// En la GPU cada thread necesita su propio generador (p. ej. cuRAND)
unsigned long long estado_rng = 12345ULL;
double aleatorio(void) {
  estado_rng = estado_rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return (estado_rng >> 11) * (1.0 / 9007199254740992.0);
}

// Media iteración de Metropolis sobre los sitios con (i + j) % 2 == color.
// prob[(s h + 4) / 2] es la probabilidad de aceptar el volteo
void metropolis_color(int *s, int L, const real *prob, int color) {
  for (int j = 0; j < L; j++) {
    int jm = (j - 1 + L) % L, jp = (j + 1) % L;
    for (int i = (j + color) % 2; i < L; i += 2) {
      int im = (i - 1 + L) % L, ip = (i + 1) % L;
      int idx = j * L + i;
      int h = s[j * L + ip] + s[j * L + im] + s[jp * L + i] + s[jm * L + i];
      int sh = s[idx] * h;
      if (sh <= 0 || aleatorio() < prob[(sh + 4) / 2])
        s[idx] = -s[idx];
    }
  }
}

// Magnetización total y energía total (cada enlace se cuenta una vez)
void medir(const int *s, int L, real *M, real *E) {
  long long m = 0, e = 0;
  for (int j = 0; j < L; j++)
    for (int i = 0; i < L; i++) {
      int idx = j * L + i;
      m += s[idx];
      e -= s[idx] * (s[j * L + (i + 1) % L] + s[((j + 1) % L) * L + i]);
    }
  *M = (real)m;
  *E = (real)e;
}

// Integral elíptica completa de primera especie K(k), vía la media
// aritmético-geométrica: K(k) = pi / (2 AGM(1, sqrt(1 - k^2)))
real elliptic_K(real k) {
  real a = 1.0, b = sqrt(1.0 - k * k);
  for (int n = 0; n < 50 && fabs(a - b) > 1e-15; n++) {
    real an = 0.5 * (a + b);
    b = sqrt(a * b);
    a = an;
  }
  return PI / (2.0 * a);
}

real m_onsager(real T) {
  real Tc = 2.0 / log(1.0 + sqrt(2.0));
  if (T >= Tc)
    return 0.0;
  return pow(1.0 - pow(sinh(2.0 / T), -4.0), 0.125);
}

real e_onsager(real T) {
  real b = 1.0 / T;
  real kappa = 2.0 * sinh(2.0 * b) / (cosh(2.0 * b) * cosh(2.0 * b));
  real t2 = tanh(2.0 * b) * tanh(2.0 * b);
  return -(1.0 / tanh(2.0 * b)) *
         (1.0 + (2.0 / PI) * (2.0 * t2 - 1.0) * elliptic_K(kappa));
}

int main(int argc, char **argv) {
  int L = (argc > 1) ? atoi(argv[1]) : 512;
  real T_min = (argc > 2) ? atof(argv[2]) : 1.5;
  real T_max = (argc > 3) ? atof(argv[3]) : 3.5;
  int nT = (argc > 4) ? atoi(argv[4]) : 11;
  int term = (argc > 5) ? atoi(argv[5]) : 1000;
  int medidas = (argc > 6) ? atoi(argv[6]) : 1000;
  estado_rng = (argc > 7) ? strtoull(argv[7], NULL, 10) : 12345ULL;

  // El tablero de ajedrez con bordes periódicos necesita L par: con L impar
  // los sitios (0, j) y (L-1, j) son vecinos y tienen el mismo color
  if (L < 2 || L % 2 != 0) {
    fprintf(stderr, "L debe ser par\n");
    return 1;
  }
  if (medidas < NBLOQUES || medidas % NBLOQUES != 0) {
    fprintf(stderr, "medidas debe ser múltiplo de %d\n", NBLOQUES);
    return 1;
  }
  int por_bloque = medidas / NBLOQUES;

  int *s = (int *)malloc((size_t)L * L * sizeof(int));
  real Nsitios = (real)L * L;

  printf("Ising 2D: L = %d, %d temperaturas en [%.3f, %.3f], %d + %d "
         "barridos\n",
         L, nT, T_min, T_max, term, medidas);
  printf("%7s %10s %9s %10s %10s %9s %10s %10s %10s\n", "T", "<|m|>", "error",
         "m exacta", "<e>", "error", "e exacta", "C", "chi");

  double t0 = segundos();
  for (int t = 0; t < nT; t++) {
    real T = (nT > 1) ? T_min + t * (T_max - T_min) / (nT - 1) : T_min;
    real prob[5];
    for (int k = 0; k < 5; k++)
      prob[k] = exp(-2.0 * (2 * k - 4) / T);

    for (int k = 0; k < L * L; k++)
      s[k] = 1;

    for (int n = 0; n < term; n++) {
      metropolis_color(s, L, prob, 0);
      metropolis_color(s, L, prob, 1);
    }

    real suma_m = 0, suma_m2 = 0, suma_e = 0, suma_e2 = 0;
    real bloque_m[NBLOQUES], bloque_e[NBLOQUES];
    for (int b = 0; b < NBLOQUES; b++) {
      bloque_m[b] = 0;
      bloque_e[b] = 0;
      for (int n = 0; n < por_bloque; n++) {
        metropolis_color(s, L, prob, 0);
        metropolis_color(s, L, prob, 1);
        real M, E;
        medir(s, L, &M, &E);
        real m = fabs(M) / Nsitios, e = E / Nsitios;
        suma_m += m;
        suma_m2 += m * m;
        suma_e += e;
        suma_e2 += e * e;
        bloque_m[b] += m / por_bloque;
        bloque_e[b] += e / por_bloque;
      }
    }
    real m = suma_m / medidas, m2 = suma_m2 / medidas;
    real e = suma_e / medidas, e2 = suma_e2 / medidas;
    real C = Nsitios * (e2 - e * e) / (T * T);
    real chi = Nsitios * (m2 - m * m) / T;

    // Error estándar a partir de la dispersión de los promedios por bloque
    real var_m = 0, var_e = 0;
    for (int b = 0; b < NBLOQUES; b++) {
      var_m += (bloque_m[b] - m) * (bloque_m[b] - m);
      var_e += (bloque_e[b] - e) * (bloque_e[b] - e);
    }
    real err_m = sqrt(var_m / (NBLOQUES * (NBLOQUES - 1.0)));
    real err_e = sqrt(var_e / (NBLOQUES * (NBLOQUES - 1.0)));

    printf("%7.4f %10.6f %9.6f %10.6f %10.6f %9.6f %10.6f %10.4f %10.4f\n", T,
           m, err_m, m_onsager(T), e, err_e, e_onsager(T), C, chi);
  }
  double t_total = segundos() - t0;

  long long volteos = (long long)nT * (term + medidas) * L * L;
  printf("Tiempo de cómputo: %.3f s (%.2f ns por intento de volteo)\n",
         t_total, 1e9 * t_total / volteos);

  real *config = (real *)malloc((size_t)L * L * sizeof(real));
  for (int k = 0; k < L * L; k++)
    config[k] = s[k];
  real *campos[1] = {config};
  escribir_salida("salida_cpu.bin", L, L, 1, campos);

  free(s);
  free(config);
  return 0;
}
