/*
 * Proyecto: ecuación de Schrödinger 2D dependiente del tiempo
 * (versión CPU de referencia)
 *
 * Resuelve  i psi_t = -(1/2) lap(psi) + V(x,y) psi  (hbar = m = 1) en la caja
 * [-Lc/2, Lc/2)^2 con bordes periódicos y el potencial de oscilador armónico
 * V = (1/2) omega^2 (x^2 + y^2), usando el método split-step de Fourier
 * (Strang):
 *
 *   psi <- exp(-i V dt/2) psi            (en el espacio x)
 *   psi <- FFT(psi)
 *   psi <- exp(-i k^2 dt/2) psi          (en el espacio k)
 *   psi <- IFFT(psi)
 *   psi <- exp(-i V dt/2) psi
 *
 * La FFT 2D se hace con FFTs 1D radix-2 por filas y luego por columnas
 * (N debe ser potencia de 2). En la GPU se reemplaza por cuFFT.
 *
 * Condición inicial: el estado fundamental desplazado a (x0, 0), un estado
 * coherente. Su centro oscila como una partícula clásica:
 *   <x>(t) = x0 cos(omega t),  <y>(t) = 0,
 * sin cambiar de forma, y la norma se conserva. El código imprime <x> y la
 * norma para validar.
 *
 * psi se guarda como reales intercalados (re, im, re, im, ...), el mismo
 * formato que cufftDoubleComplex.
 *
 * Compilar:  gcc -O2 schrodinger_cpu.c -lm -o schrodinger_cpu.x
 * Ejecutar:  ./schrodinger_cpu.x [N] [pasos]
 *            N: puntos por lado, potencia de 2 (por defecto 512)
 *            pasos: pasos de tiempo con dt = 0.01 (por defecto 628, ~ un
 *                   período de oscilación)
 *
 * Salida: <x>, <y> y la norma a lo largo de la simulación, tiempo de cómputo
 * del ciclo principal (incluidas esas medidas) y las partes real e
 * imaginaria de psi final en salida_cpu.bin (formato en
 * escribir_salida_compleja).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

// Cambiar a float (cufftComplex en vez de cufftDoubleComplex) es una
// optimización posible: hay que justificarla con la norma y <x>
typedef double real;

double segundos(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Escribe una cabecera (nx, ny, ncampos, bytes por real) seguida de los
// campos uno tras otro, cada uno con nx*ny valores en orden [y][x].
// Aquí se escriben dos campos: Re(psi) e Im(psi)
void escribir_salida_compleja(const char *nombre, int N, const real *psi) {
  FILE *f = fopen(nombre, "wb");
  int cabecera[4] = {N, N, 2, (int)sizeof(real)};
  fwrite(cabecera, sizeof(int), 4, f);
  for (int parte = 0; parte < 2; parte++)
    for (int k = 0; k < N * N; k++)
      fwrite(&psi[2 * k + parte], sizeof(real), 1, f);
  fclose(f);
}

// FFT 1D radix-2 in situ sobre n complejos intercalados.
// signo = -1: transformada directa; signo = +1: inversa (sin normalizar)
void fft1d(real *a, int n, int signo) {
  // Permutación por inversión de bits
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      real tr = a[2 * i], ti = a[2 * i + 1];
      a[2 * i] = a[2 * j];
      a[2 * i + 1] = a[2 * j + 1];
      a[2 * j] = tr;
      a[2 * j + 1] = ti;
    }
  }
  // Mariposas de Cooley-Tukey
  for (int len = 2; len <= n; len <<= 1) {
    real ang = signo * 2.0 * PI / len;
    for (int k = 0; k < len / 2; k++) {
      real wr = cos(ang * k), wi = sin(ang * k);
      for (int i = k; i < n; i += len) {
        int p = i, q = i + len / 2;
        real xr = a[2 * q] * wr - a[2 * q + 1] * wi;
        real xi = a[2 * q] * wi + a[2 * q + 1] * wr;
        a[2 * q] = a[2 * p] - xr;
        a[2 * q + 1] = a[2 * p + 1] - xi;
        a[2 * p] += xr;
        a[2 * p + 1] += xi;
      }
    }
  }
}

// FFT 2D: FFTs 1D por filas y luego por columnas (copiando cada columna a un
// arreglo auxiliar). La inversa se normaliza por 1/N^2. Ojo: cuFFT no
// normaliza la inversa; hay que multiplicar por 1/N^2 en un kernel (p. ej. el
// mismo que aplica la fase del potencial)
void fft2d(real *psi, int N, int signo, real *col) {
  for (int j = 0; j < N; j++)
    fft1d(&psi[2 * j * N], N, signo);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      col[2 * j] = psi[2 * (j * N + i)];
      col[2 * j + 1] = psi[2 * (j * N + i) + 1];
    }
    fft1d(col, N, signo);
    for (int j = 0; j < N; j++) {
      psi[2 * (j * N + i)] = col[2 * j];
      psi[2 * (j * N + i) + 1] = col[2 * j + 1];
    }
  }
  if (signo > 0) {
    real norma = 1.0 / ((real)N * N);
    for (int k = 0; k < 2 * N * N; k++)
      psi[k] *= norma;
  }
}

// psi <- fase * psi, punto a punto (ambos complejos intercalados)
void multiplicar(real *psi, const real *fase, int n) {
  for (int k = 0; k < n; k++) {
    real pr = psi[2 * k], pi = psi[2 * k + 1];
    real fr = fase[2 * k], fi = fase[2 * k + 1];
    psi[2 * k] = pr * fr - pi * fi;
    psi[2 * k + 1] = pr * fi + pi * fr;
  }
}

// Norma, <x> y <y> de psi
void observables(const real *psi, int N, real dx, real Lc, real *norma,
                 real *xm, real *ym) {
  real s = 0, sx = 0, sy = 0;
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      int k = j * N + i;
      real rho = psi[2 * k] * psi[2 * k] + psi[2 * k + 1] * psi[2 * k + 1];
      real x = -0.5 * Lc + i * dx, y = -0.5 * Lc + j * dx;
      s += rho;
      sx += x * rho;
      sy += y * rho;
    }
  *norma = s * dx * dx;
  *xm = sx / s;
  *ym = sy / s;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 512;
  int pasos = (argc > 2) ? atoi(argv[2]) : 628;
  if (N < 2 || (N & (N - 1)) != 0) {
    fprintf(stderr, "N debe ser potencia de 2\n");
    return 1;
  }
  if (pasos < 1) {
    fprintf(stderr, "Se necesita pasos >= 1\n");
    return 1;
  }

  real Lc = 20.0;     // lado de la caja
  real omega = 1.0;   // frecuencia del oscilador
  real x0 = 3.0;      // desplazamiento inicial
  real dt = 0.01;
  real dx = Lc / N;
  real dk = 2.0 * PI / Lc;
  int muestras = 10;  // cuántas veces imprimir observables

  size_t n = (size_t)N * N;
  real *psi = (real *)malloc(2 * n * sizeof(real));
  real *fase_V = (real *)malloc(2 * n * sizeof(real));
  real *fase_K = (real *)malloc(2 * n * sizeof(real));
  real *col = (real *)malloc(2 * N * sizeof(real));

  // Condición inicial y factores de fase precalculados
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++) {
      int k = j * N + i;
      real x = -0.5 * Lc + i * dx, y = -0.5 * Lc + j * dx;
      real r2 = (x - x0) * (x - x0) + y * y;
      psi[2 * k] = sqrt(omega / PI) * exp(-0.5 * omega * r2);
      psi[2 * k + 1] = 0.0;

      real V = 0.5 * omega * omega * (x * x + y * y);
      fase_V[2 * k] = cos(-V * 0.5 * dt);
      fase_V[2 * k + 1] = sin(-V * 0.5 * dt);

      // Números de onda en el orden de la FFT: 0, 1, ..., N/2-1, -N/2, ..., -1
      real kx = ((i < N / 2) ? i : i - N) * dk;
      real ky = ((j < N / 2) ? j : j - N) * dk;
      real T = 0.5 * (kx * kx + ky * ky);
      fase_K[2 * k] = cos(-T * dt);
      fase_K[2 * k + 1] = sin(-T * dt);
    }

  printf("Schrödinger 2D (split-step Fourier): N = %d, pasos = %d, dt = %.3f\n",
         N, pasos, dt);
  printf("%9s %12s %12s %12s %12s\n", "t", "norma", "<x>", "x0 cos(wt)",
         "<y>");

  real norma0, xm, ym;
  observables(psi, N, dx, Lc, &norma0, &xm, &ym);
  printf("%9.4f %12.9f %12.8f %12.8f %12.2e\n", 0.0, norma0, xm, x0, ym);

  double t0 = segundos();
  int cada = (pasos >= muestras) ? pasos / muestras : 1;
  for (int p = 1; p <= pasos; p++) {
    multiplicar(psi, fase_V, (int)n);
    fft2d(psi, N, -1, col);
    multiplicar(psi, fase_K, (int)n);
    fft2d(psi, N, +1, col);
    multiplicar(psi, fase_V, (int)n);

    if (p % cada == 0 || p == pasos) {
      real norma;
      observables(psi, N, dx, Lc, &norma, &xm, &ym);
      real t = p * dt;
      printf("%9.4f %12.9f %12.8f %12.8f %12.2e\n", t, norma, xm,
             x0 * cos(omega * t), ym);
    }
  }

  double t_total = segundos() - t0;

  real norma;
  observables(psi, N, dx, Lc, &norma, &xm, &ym);
  real t = pasos * dt;
  printf("Tiempo de cómputo: %.3f s (%.3f ms por paso)\n", t_total,
         1e3 * t_total / pasos);
  printf("Error final en <x>: %.3e, variación relativa de la norma: %.3e\n",
         fabs(xm - x0 * cos(omega * t)), fabs(norma - norma0) / norma0);

  escribir_salida_compleja("salida_cpu.bin", N, psi);

  free(psi);
  free(fase_V);
  free(fase_K);
  free(col);
  return 0;
}
