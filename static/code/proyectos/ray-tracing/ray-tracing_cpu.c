/*
 * Proyecto: ray tracing (versión CPU de referencia)
 *
 * Genera una imagen siguiendo rayos de luz desde la cámara hacia la escena
 * (path tracing), al estilo de "Ray Tracing in One Weekend" de P. Shirley,
 * pero en C: las esferas y materiales son structs y el tipo de material se
 * elige con un switch (sin funciones virtuales, que son incómodas en la GPU).
 *
 * Óptica geométrica de cada material:
 *   - difuso (Lambert): el rayo sale en una dirección aleatoria alrededor de
 *     la normal y se atenúa por el albedo;
 *   - metal: reflexión especular r = d - 2 (d.n) n, con un poco de ruido
 *     ("fuzz") para metales opacos;
 *   - dieléctrico (vidrio): refracción con la ley de Snell
 *     n1 sin(t1) = n2 sin(t2), reflexión total interna, y la probabilidad de
 *     reflexión de Fresnel en la aproximación de Schlick.
 *
 * Cada píxel promedia 'muestras' rayos con pequeñas desviaciones aleatorias
 * (antialiasing y desenfoque de la cámara); cada rayo rebota hasta
 * 'max_rebotes' veces y, si escapa, toma el color del cielo.
 *
 * Números aleatorios: cada píxel tiene su propio generador, inicializado a
 * partir de su índice. Así el resultado no depende del orden en que se
 * calculan los píxeles: una versión GPU que use la misma función aleatorio()
 * debe reproducir la imagen de CPU (salvo en los pocos rayos donde el
 * redondeo cambia un rebote).
 *
 * Modos:
 *   0: la escena final del libro (~490 esferas al azar).
 *   1: prueba del "horno blanco": esferas difusas y de vidrio de albedo 1
 *      iluminadas por un cielo uniforme de intensidad 1. Como ningún
 *      material absorbe luz, todos los píxeles deben valer 1 (conservación
 *      de la energía). En este modo se permiten 1000 rebotes: con 50,
 *      algunos rayos atrapados entre el vidrio y el suelo se pierden.
 *
 * Compilar:  gcc -O2 ray-tracing_cpu.c -lm -o ray-tracing_cpu.x
 * Ejecutar:  ./ray-tracing_cpu.x [modo] [ancho] [alto] [muestras]
 *            por defecto: modo 0, 400 x 225 píxeles, 100 muestras por píxel
 *
 * Salida: tiempo de cómputo, la imagen en imagen_cpu.ppm (se abre con
 * cualquier visor de imágenes) y los canales r, g, b en salida_cpu.bin (ver
 * comparar.py y graficar.py).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323846

// Cambiar a float es una optimización importante en la T4: hay que mostrar
// que la imagen no cambia apreciablemente
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

// ---------------------------------------------------------------------------
// Números aleatorios: hash PCG. Cada píxel guarda su propio estado de 32 bits
// ---------------------------------------------------------------------------

uint32_t hash_pcg(uint32_t v) {
  uint32_t s = v * 747796405u + 2891336453u;
  uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// Uniforme en [0, 1). Se usan los 24 bits altos, que caben exactos en un
// float: así el resultado nunca se redondea a 1 aunque real sea float
real aleatorio(uint32_t *estado) {
  *estado = hash_pcg(*estado);
  return (real)(*estado >> 8) * (real)(1.0 / 16777216.0);
}

// ---------------------------------------------------------------------------
// Vectores
// ---------------------------------------------------------------------------

typedef struct {
  real x, y, z;
} vec3;

vec3 v3(real x, real y, real z) {
  vec3 v = {x, y, z};
  return v;
}
vec3 suma(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
vec3 resta(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
vec3 escalar(real s, vec3 a) { return v3(s * a.x, s * a.y, s * a.z); }
vec3 producto(vec3 a, vec3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
real punto(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
vec3 cruz(vec3 a, vec3 b) {
  return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x);
}
vec3 unitario(vec3 a) { return escalar(1.0 / sqrt(punto(a, a)), a); }

// Punto aleatorio dentro de la esfera unitaria (por rechazo)
vec3 aleatorio_en_esfera(uint32_t *estado) {
  vec3 p;
  do {
    p = v3(2.0 * aleatorio(estado) - 1.0, 2.0 * aleatorio(estado) - 1.0,
           2.0 * aleatorio(estado) - 1.0);
  } while (punto(p, p) >= 1.0);
  return p;
}

// Punto aleatorio dentro del disco unitario en el plano xy (por rechazo)
vec3 aleatorio_en_disco(uint32_t *estado) {
  vec3 p;
  do {
    p = v3(2.0 * aleatorio(estado) - 1.0, 2.0 * aleatorio(estado) - 1.0, 0.0);
  } while (punto(p, p) >= 1.0);
  return p;
}

// ---------------------------------------------------------------------------
// Escena: esferas con material
// ---------------------------------------------------------------------------

enum { DIFUSO, METAL, DIELECTRICO };

typedef struct {
  vec3 centro;
  real radio;
  int material;
  vec3 albedo;  // color (difuso y metal)
  real param;   // fuzz (metal) o índice de refracción (dieléctrico)
} esfera;

typedef struct {
  vec3 origen, esquina, horizontal, vertical, u, v;
  real radio_lente;
} camara;

camara crear_camara(vec3 desde, vec3 hacia, vec3 arriba, real vfov,
                    real aspecto, real apertura, real dist_foco) {
  camara c;
  real theta = vfov * PI / 180.0;
  real alto = 2.0 * tan(theta / 2.0);
  real ancho = aspecto * alto;
  vec3 w = unitario(resta(desde, hacia));
  c.u = unitario(cruz(arriba, w));
  c.v = cruz(w, c.u);
  c.origen = desde;
  c.horizontal = escalar(dist_foco * ancho, c.u);
  c.vertical = escalar(dist_foco * alto, c.v);
  c.esquina = resta(resta(resta(desde, escalar(0.5, c.horizontal)),
                          escalar(0.5, c.vertical)),
                    escalar(dist_foco, w));
  c.radio_lente = apertura / 2.0;
  return c;
}

// Escena final de "Ray Tracing in One Weekend": suelo, una grilla de esferas
// pequeñas al azar y tres esferas grandes. Devuelve el número de esferas
int escena_libro(esfera *e) {
  uint32_t estado = 2024u; // semilla fija: la escena es siempre la misma
  int n = 0;
  esfera suelo = {v3(0, -1000, 0), 1000, DIFUSO, v3(0.5, 0.5, 0.5), 0};
  e[n++] = suelo;
  for (int a = -11; a < 11; a++)
    for (int b = -11; b < 11; b++) {
      real elegir = aleatorio(&estado);
      vec3 centro = v3(a + 0.9 * aleatorio(&estado), 0.2,
                       b + 0.9 * aleatorio(&estado));
      if (sqrt(punto(resta(centro, v3(4, 0.2, 0)),
                     resta(centro, v3(4, 0.2, 0)))) <= 0.9)
        continue;
      esfera s = {centro, 0.2, DIFUSO, v3(0, 0, 0), 0};
      if (elegir < 0.8) {
        s.albedo = v3(aleatorio(&estado) * aleatorio(&estado),
                      aleatorio(&estado) * aleatorio(&estado),
                      aleatorio(&estado) * aleatorio(&estado));
      } else if (elegir < 0.95) {
        s.material = METAL;
        s.albedo = v3(0.5 * (1 + aleatorio(&estado)),
                      0.5 * (1 + aleatorio(&estado)),
                      0.5 * (1 + aleatorio(&estado)));
        s.param = 0.5 * aleatorio(&estado);
      } else {
        s.material = DIELECTRICO;
        s.param = 1.5;
      }
      e[n++] = s;
    }
  esfera vidrio = {v3(0, 1, 0), 1.0, DIELECTRICO, v3(1, 1, 1), 1.5};
  esfera difusa = {v3(-4, 1, 0), 1.0, DIFUSO, v3(0.4, 0.2, 0.1), 0};
  esfera metal = {v3(4, 1, 0), 1.0, METAL, v3(0.7, 0.6, 0.5), 0.0};
  e[n++] = vidrio;
  e[n++] = difusa;
  e[n++] = metal;
  return n;
}

// Escena del "horno blanco": nada absorbe luz
int escena_horno(esfera *e) {
  int n = 0;
  esfera suelo = {v3(0, -1000, 0), 1000, DIFUSO, v3(1, 1, 1), 0};
  esfera vidrio = {v3(0, 1, 0), 1.0, DIELECTRICO, v3(1, 1, 1), 1.5};
  esfera difusa1 = {v3(-4, 1, 0), 1.0, DIFUSO, v3(1, 1, 1), 0};
  esfera difusa2 = {v3(4, 1, 0), 1.0, DIFUSO, v3(1, 1, 1), 0};
  e[n++] = suelo;
  e[n++] = vidrio;
  e[n++] = difusa1;
  e[n++] = difusa2;
  return n;
}

// ---------------------------------------------------------------------------
// Trazado de rayos
// ---------------------------------------------------------------------------

// Busca la intersección más cercana del rayo o + t d con t en (t_min, t_max).
// Devuelve el índice de la esfera (o -1) y escribe t
int interseccion(const esfera *e, int n, vec3 o, vec3 d, real t_min,
                 real t_max, real *t_hit) {
  int cual = -1;
  real t_cerca = t_max;
  for (int k = 0; k < n; k++) {
    vec3 oc = resta(o, e[k].centro);
    real a = punto(d, d);
    real b = punto(oc, d);
    real c = punto(oc, oc) - e[k].radio * e[k].radio;
    real disc = b * b - a * c;
    if (disc > 0) {
      real raiz = sqrt(disc);
      real t = (-b - raiz) / a;
      if (t <= t_min || t >= t_cerca)
        t = (-b + raiz) / a;
      if (t > t_min && t < t_cerca) {
        t_cerca = t;
        cual = k;
      }
    }
  }
  *t_hit = t_cerca;
  return cual;
}

real schlick(real coseno, real indice) {
  real r0 = (1 - indice) / (1 + indice);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow(1 - coseno, 5);
}

// Color de un rayo: sigue sus rebotes de forma iterativa (sin recursión)
vec3 color(vec3 o, vec3 d, const esfera *e, int n, int max_rebotes,
           int cielo_uniforme, uint32_t *estado) {
  vec3 atenuacion = v3(1, 1, 1);
  for (int rebote = 0; rebote < max_rebotes; rebote++) {
    real t;
    int k = interseccion(e, n, o, d, 0.001, 1e30, &t);
    if (k < 0) {
      // El rayo escapa: color del cielo
      if (cielo_uniforme)
        return atenuacion;
      vec3 du = unitario(d);
      real s = 0.5 * (du.y + 1.0);
      vec3 cielo = suma(escalar(1.0 - s, v3(1, 1, 1)),
                        escalar(s, v3(0.5, 0.7, 1.0)));
      return producto(atenuacion, cielo);
    }

    vec3 p = suma(o, escalar(t, d));
    vec3 normal = escalar(1.0 / e[k].radio, resta(p, e[k].centro));
    vec3 nueva;

    switch (e[k].material) {
    case DIFUSO:
      nueva = suma(normal, aleatorio_en_esfera(estado));
      atenuacion = producto(atenuacion, e[k].albedo);
      break;
    case METAL: {
      vec3 du = unitario(d);
      vec3 reflejado = resta(du, escalar(2.0 * punto(du, normal), normal));
      nueva = suma(reflejado, escalar(e[k].param, aleatorio_en_esfera(estado)));
      if (punto(nueva, normal) <= 0)
        return v3(0, 0, 0); // el ruido lo mandó hacia dentro: absorbido
      atenuacion = producto(atenuacion, e[k].albedo);
      break;
    }
    default: { // DIELECTRICO
      vec3 du = unitario(d);
      real dn = punto(du, normal);
      vec3 n_fuera;
      real razon, coseno;
      if (dn > 0) { // el rayo sale del vidrio
        n_fuera = escalar(-1.0, normal);
        razon = e[k].param;
        coseno = sqrt(1.0 - e[k].param * e[k].param * (1.0 - dn * dn));
      } else { // el rayo entra al vidrio
        n_fuera = normal;
        razon = 1.0 / e[k].param;
        coseno = -dn;
      }
      vec3 reflejado = resta(du, escalar(2.0 * dn, normal));
      // Ley de Snell: discriminante < 0 es reflexión total interna
      real dt = punto(du, n_fuera);
      real disc = 1.0 - razon * razon * (1.0 - dt * dt);
      if (disc > 0 && aleatorio(estado) >= schlick(coseno, e[k].param))
        nueva = resta(escalar(razon, resta(du, escalar(dt, n_fuera))),
                      escalar(sqrt(disc), n_fuera));
      else
        nueva = reflejado;
      break; // el vidrio no absorbe
    }
    }
    o = p;
    d = nueva;
  }
  return v3(0, 0, 0); // demasiados rebotes
}

int main(int argc, char **argv) {
  int modo = (argc > 1) ? atoi(argv[1]) : 0;
  int ancho = (argc > 2) ? atoi(argv[2]) : 400;
  int alto = (argc > 3) ? atoi(argv[3]) : 225;
  int muestras = (argc > 4) ? atoi(argv[4]) : 100;
  // En el horno blanco la luz puede quedar atrapada muchos rebotes (p. ej. por
  // reflexión total interna cerca del punto de contacto vidrio-suelo)
  int max_rebotes = (modo == 1) ? 1000 : 50;

  esfera *e = (esfera *)malloc(500 * sizeof(esfera));
  int n = (modo == 1) ? escena_horno(e) : escena_libro(e);

  vec3 desde = v3(13, 2, 3), hacia = v3(0, 0, 0);
  camara cam = crear_camara(desde, hacia, v3(0, 1, 0), 20.0,
                            (real)ancho / alto, 0.1, 10.0);

  size_t npix = (size_t)ancho * alto;
  real *r = (real *)malloc(npix * sizeof(real));
  real *g = (real *)malloc(npix * sizeof(real));
  real *b = (real *)malloc(npix * sizeof(real));

  printf("Ray tracing, modo %d (%s): %d x %d píxeles, %d muestras, %d "
         "esferas\n",
         modo, (modo == 1) ? "horno blanco" : "escena del libro", ancho, alto,
         muestras, n);

  double t0 = segundos();
  for (int j = 0; j < alto; j++)
    for (int i = 0; i < ancho; i++) {
      int pix = j * ancho + i;
      uint32_t estado = hash_pcg((uint32_t)pix + 1u);
      vec3 suma_color = v3(0, 0, 0);
      for (int s = 0; s < muestras; s++) {
        // Punto aleatorio dentro del píxel y en la lente (desenfoque)
        real u = (i + aleatorio(&estado)) / ancho;
        real v = (j + aleatorio(&estado)) / alto;
        vec3 rd = escalar(cam.radio_lente, aleatorio_en_disco(&estado));
        vec3 desplaz = suma(escalar(rd.x, cam.u), escalar(rd.y, cam.v));
        vec3 o = suma(cam.origen, desplaz);
        vec3 d = resta(suma(cam.esquina, suma(escalar(u, cam.horizontal),
                                              escalar(v, cam.vertical))),
                       o);
        suma_color = suma(suma_color, color(o, d, e, n, max_rebotes,
                                            modo == 1, &estado));
      }
      r[pix] = suma_color.x / muestras;
      g[pix] = suma_color.y / muestras;
      b[pix] = suma_color.z / muestras;
    }
  double t_total = segundos() - t0;

  long long rayos = (long long)npix * muestras;
  printf("Tiempo de cómputo: %.3f s (%.2f millones de rayos de cámara por "
         "segundo)\n",
         t_total, rayos / t_total / 1e6);

  real prom = 0.0, minimo = 1e30;
  for (size_t k = 0; k < npix; k++) {
    real m = (r[k] + g[k] + b[k]) / 3.0;
    prom += m;
    if (m < minimo)
      minimo = m;
  }
  prom /= npix;
  printf("Brillo promedio = %.6f, mínimo = %.6f", prom, minimo);
  if (modo == 1)
    printf(" (horno blanco: ambos deben ser ~1)");
  printf("\n");

  // Imagen PPM con corrección gamma 2; la fila 0 es la de abajo
  FILE *f = fopen("imagen_cpu.ppm", "w");
  fprintf(f, "P3\n%d %d\n255\n", ancho, alto);
  for (int j = alto - 1; j >= 0; j--)
    for (int i = 0; i < ancho; i++) {
      int pix = j * ancho + i;
      real c[3] = {r[pix], g[pix], b[pix]};
      for (int k = 0; k < 3; k++) {
        real x = sqrt(c[k] > 1.0 ? 1.0 : c[k]);
        fprintf(f, "%d ", (int)(255.99 * x));
      }
      fprintf(f, "\n");
    }
  fclose(f);

  real *campos[3] = {r, g, b};
  escribir_salida("salida_cpu.bin", ancho, alto, 3, campos);

  free(e);
  free(r);
  free(g);
  free(b);
  return 0;
}
