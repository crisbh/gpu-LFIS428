/*
 * Proyecto: fluido con el método de Lattice Boltzmann D2Q9
 * (versión CPU de referencia)
 *
 * En cada nodo de una malla nx x ny se guardan 9 funciones de distribución
 * f_i, una por cada velocidad discreta c_i (reposo, 4 ejes, 4 diagonales).
 * Cada paso tiene dos etapas:
 *
 *   1. Colisión (BGK), local a cada nodo:
 *        rho = sum f_i,   rho u = sum f_i c_i + F/2
 *        f_i <- f_i - (f_i - f_i^eq(rho, u)) / tau + S_i
 *      con f_i^eq = w_i rho (1 + 3 c.u + 9/2 (c.u)^2 - 3/2 u^2) y S_i el
 *      término de fuerza de Guo. La viscosidad es nu = (tau - 1/2) / 3.
 *   2. Propagación: cada f_i se mueve al nodo vecino en la dirección c_i.
 *      Si el vecino es sólido, rebota (bounce-back), lo que impone u = 0 a
 *      medio camino entre el nodo fluido y el sólido.
 *
 * El flujo es impulsado por una fuerza uniforme F = rho g en x, con bordes
 * periódicos en x y paredes sólidas en y = 0 e y = ny - 1.
 *
 * Modos:
 *   0: canal (flujo de Poiseuille). El estado estacionario es
 *        u_x(y) = g / (2 nu) y' (H - y'),  y' = y - 1/2,  H = ny - 2,
 *      que se usa para validar. Partiendo del reposo, el perfil se acerca a
 *      la parábola como exp(-t / t_rel), con t_rel = H^2 / (pi^2 nu) pasos:
 *      el tiempo crece como el ancho del canal al cuadrado. Por defecto se
 *      simulan ~12 t_rel; con menos pasos la validación falla aunque el
 *      código esté bien.
 *   1: canal con un cilindro. El fluido parte con velocidad uniforme y, con
 *      tau cercano a 1/2 (viscosidad baja), se forma una calle de vórtices
 *      de von Kármán detrás del cilindro. Es cualitativo (la estela vuelve a
 *      entrar por el borde periódico) y no tiene solución exacta: se valida
 *      contra la salida de CPU.
 *
 * Las f_i se guardan en formato AoS: f[(y * nx + x) * 9 + i].
 *
 * Compilar:  gcc -O2 lattice-boltzmann_cpu.c -lm -o lattice-boltzmann_cpu.x
 * Ejecutar:  ./lattice-boltzmann_cpu.x [modo] [pasos] [nx] [ny]
 *            modo 0 (por defecto): pasos = 12 t_rel (~30000), nx = 512, ny = 66
 *            modo 1:               pasos = 10000, nx = 512, ny = 128
 *
 * Salida: tiempo de cómputo (y MLUPS: millones de nodos actualizados por
 * segundo, la métrica usual de LBM), error respecto a Poiseuille (modo 0) y
 * los campos u_x, u_y, rho en salida_cpu.bin (ver comparar.py y graficar.py).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

typedef double real;

// Velocidades discretas, pesos y dirección opuesta de D2Q9
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const real w[9] = {4.0 / 9,  1.0 / 9,  1.0 / 9,  1.0 / 9, 1.0 / 9,
                   1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};
const int opuesta[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

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

// Densidad y velocidad (con la corrección de fuerza de Guo) en un nodo
void macroscopicas(const real *fn, real gx, real *rho, real *ux, real *uy) {
  real r = 0, jx = 0, jy = 0;
  for (int i = 0; i < 9; i++) {
    r += fn[i];
    jx += fn[i] * cx[i];
    jy += fn[i] * cy[i];
  }
  *rho = r;
  *ux = (jx + 0.5 * r * gx) / r;
  *uy = jy / r;
}

// Colisión BGK con fuerza de Guo en todos los nodos fluidos
void colision(real *f, const int *solido, int nx, int ny, real tau, real gx) {
  for (int y = 0; y < ny; y++)
    for (int x = 0; x < nx; x++) {
      int nodo = y * nx + x;
      if (solido[nodo])
        continue;
      real *fn = &f[9 * nodo];
      real rho, ux, uy;
      macroscopicas(fn, gx, &rho, &ux, &uy);
      real u2 = ux * ux + uy * uy;
      real Fx = rho * gx;
      for (int i = 0; i < 9; i++) {
        real cu = cx[i] * ux + cy[i] * uy;
        real feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        real S = (1.0 - 0.5 / tau) * w[i] *
                 (3.0 * (cx[i] - ux) + 9.0 * cu * cx[i]) * Fx;
        fn[i] += -(fn[i] - feq) / tau + S;
      }
    }
}

// Propagación "pull": cada nodo fluido toma f_i del vecino en -c_i. Si ese
// vecino es sólido, la distribución rebota: f_i <- f_opuesta del propio nodo
void propagacion(const real *f, real *f_new, const int *solido, int nx,
                 int ny) {
  for (int y = 0; y < ny; y++)
    for (int x = 0; x < nx; x++) {
      int nodo = y * nx + x;
      if (solido[nodo])
        continue;
      for (int i = 0; i < 9; i++) {
        int xs = (x - cx[i] + nx) % nx; // periódico en x
        int ys = y - cy[i];             // las paredes evitan salir en y
        int vecino = ys * nx + xs;
        if (solido[vecino])
          f_new[9 * nodo + i] = f[9 * nodo + opuesta[i]];
        else
          f_new[9 * nodo + i] = f[9 * vecino + i];
      }
    }
}

int main(int argc, char **argv) {
  int modo = (argc > 1) ? atoi(argv[1]) : 0;
  int nx = (argc > 3) ? atoi(argv[3]) : 512;
  int ny = (argc > 4) ? atoi(argv[4]) : ((modo == 0) ? 66 : 128);

  // Canal: tau = 1 converge rápido a Poiseuille. Cilindro: tau cercano a
  // 1/2 da viscosidad baja y Reynolds suficiente para desprender vórtices
  real tau = (modo == 0) ? 1.0 : 0.55;
  real nu = (tau - 0.5) / 3.0;
  real H = ny - 2;

  // Tiempo de relajación del modo más lento del canal, en pasos
  real t_rel = H * H / (PI * PI * nu);
  int pasos_canal = 100 * (int)ceil(12.0 * t_rel / 100.0);
  int pasos = (argc > 2) ? atoi(argv[2]) : ((modo == 0) ? pasos_canal : 10000);
  // Velocidad máxima de Poiseuille (modo 0) o velocidad inicial (modo 1);
  // la fuerza es la que mantendría un Poiseuille con esa velocidad máxima
  real u_max = (modo == 0) ? 0.05 : 0.08;
  real gx = 8.0 * nu * u_max / (H * H);

  size_t nodos = (size_t)nx * ny;
  real *f = (real *)malloc(9 * nodos * sizeof(real));
  real *f_new = (real *)malloc(9 * nodos * sizeof(real));
  int *solido = (int *)calloc(nodos, sizeof(int));

  // Geometría: paredes arriba y abajo y, en el modo 1, un cilindro
  real xc = nx / 4.0, yc = ny / 2.0, radio = ny / 10.0;
  for (int y = 0; y < ny; y++)
    for (int x = 0; x < nx; x++) {
      int pared = (y == 0 || y == ny - 1);
      int cilindro = (modo == 1) && ((x - xc) * (x - xc) + (y - yc) * (y - yc) <
                                     radio * radio);
      solido[y * nx + x] = pared || cilindro;
    }

  // Densidad 1. Modo 0: fluido en reposo. Modo 1: velocidad uniforme u_max
  // más una pequeña perturbación en u_y que rompe la simetría para que el
  // cilindro desprenda vórtices antes
  for (size_t n = 0; n < nodos; n++) {
    int y = (int)(n / nx);
    real ux = (modo == 1 && !solido[n]) ? u_max : 0.0;
    real uy = (modo == 1) ? 1e-3 * sin(6.283185307179586 * y / ny) : 0.0;
    for (int i = 0; i < 9; i++) {
      real cu = cx[i] * ux + cy[i] * uy;
      real u2 = ux * ux + uy * uy;
      f[9 * n + i] = w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
      f_new[9 * n + i] = f[9 * n + i];
    }
  }

  printf("Lattice Boltzmann D2Q9, modo %d (%s): %d x %d, pasos = %d\n", modo,
         (modo == 0) ? "canal" : "cilindro", nx, ny, pasos);
  printf("tau = %.3f, nu = %.4f, g = %.3e\n", tau, nu, gx);
  if (modo == 0) {
    printf("Tiempo de relajación t_rel = H^2 / (pi^2 nu) = %.0f pasos; se "
           "simulan %.1f t_rel\n",
           t_rel, pasos / t_rel);
    if (pasos < pasos_canal)
      printf("Aviso: con menos de %d pasos el perfil aún no llega al estado "
             "estacionario; el error vs. Poiseuille incluye una parte "
             "transitoria ~exp(-pasos / t_rel) = %.1e\n",
             pasos_canal, exp(-pasos / t_rel));
  }
  if (modo == 1)
    printf("Reynolds estimado (u_max D / nu) = %.0f\n",
           u_max * 2.0 * radio / nu);

  double t0 = segundos();
  for (int p = 0; p < pasos; p++) {
    colision(f, solido, nx, ny, tau, gx);
    propagacion(f, f_new, solido, nx, ny);
    real *tmp = f;
    f = f_new;
    f_new = tmp;
  }
  double t_total = segundos() - t0;

  // MLUPS cuenta todos los nodos, también los sólidos
  printf("Tiempo de cómputo: %.3f s (%.1f MLUPS)\n", t_total,
         (double)nodos * pasos / t_total / 1e6);

  // Campos macroscópicos
  real *ux = (real *)calloc(nodos, sizeof(real));
  real *uy = (real *)calloc(nodos, sizeof(real));
  real *rho = (real *)calloc(nodos, sizeof(real));
  for (size_t n = 0; n < nodos; n++)
    if (!solido[n])
      macroscopicas(&f[9 * n], gx, &rho[n], &ux[n], &uy[n]);

  if (modo == 0) {
    // Perfil promediado en x vs. Poiseuille
    real err = 0.0;
    for (int y = 1; y < ny - 1; y++) {
      real prom = 0.0;
      for (int x = 0; x < nx; x++)
        prom += ux[y * nx + x];
      prom /= nx;
      real yp = y - 0.5;
      real exacta = gx / (2.0 * nu) * yp * (H - yp);
      if (fabs(prom - exacta) > err)
        err = fabs(prom - exacta);
    }
    printf("Error máximo del perfil vs. Poiseuille: %.3e (relativo a u_max: "
           "%.3e)\n",
           err, err / u_max);
  } else {
    real suma = 0.0;
    int fluidos = 0;
    for (size_t n = 0; n < nodos; n++)
      if (!solido[n]) {
        suma += ux[n];
        fluidos++;
      }
    printf("Velocidad media <u_x> = %.6f\n", suma / fluidos);
  }

  real *campos[3] = {ux, uy, rho};
  escribir_salida("salida_cpu.bin", nx, ny, 3, campos);

  free(f);
  free(f_new);
  free(solido);
  free(ux);
  free(uy);
  free(rho);
  return 0;
}
