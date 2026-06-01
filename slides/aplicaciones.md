---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

# **Programación en GPUs**
## Aplicaciones

N-cuerpos · OpenGL · Ray tracing

---

## **Códigos**

Los códigos de esta clase están disponibles para descargar:

- N-cuerpos: [nbody.cu](../code/aplicaciones/nbody.cu)
- OpenGL: [simpleGL.cu](../code/aplicaciones/simpleGL.cu)
- Ray tracing (serie completa, capítulos 1–12): [ch01_rt.cu](../code/aplicaciones/ray_tracing/c01_salida_basica/ch01_rt.cu), [ch02_rt.cu](../code/aplicaciones/ray_tracing/c02_vectores/ch02_rt.cu), [ch03_rt.cu](../code/aplicaciones/ray_tracing/c03_rayos/ch03_rt.cu), [ch04_rt.cu](../code/aplicaciones/ray_tracing/c04_esferas/ch04_rt.cu), …

---

# Simulaciones: N-cuerpos

Ejemplo del CUDA Toolkit

---

## **Simulaciones: N-cuerpos**

- Una simulación de N-cuerpos trata de $N$ masas, representadas como partículas.
- La fuerza gravitacional entre estas masas se calcula en la simulación.
- De la fuerza obtenemos la **aceleración** que actúa en cada partícula.
- De la aceleración actualizamos la **velocidad** de cada partícula, y su **posición**.
- Repetimos este proceso cada *timestep* para evolucionar las partículas.

---

## **Simulaciones: N-cuerpos**

Hay varios métodos para obtener la fuerza gravitacional. El más directo es sumar todas las fuerzas entre cada par de partículas (**sumación directa**):

- Requiere una expresión para la fuerza entre un par de partículas (ley universal de la gravedad de Newton).
- Desventaja: complejidad algorítmica $\mathcal{O}(N^2)$.

---

## **Simulaciones: N-cuerpos**

Fuerza entre $2$ cuerpos:

$$\mathbf{F}_{ij} = G \frac{m_i m_j}{r_{ij}^2}\,\frac{\mathbf{r}_{ij}}{r_{ij}}$$

donde $m_i$, $m_j$ son las masas $i$ y $j$, y $\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i$.

La fuerza total $\mathbf{F}_i$ en la partícula $i$:

$$\mathbf{F}_i = \sum_{j \neq i}^N \mathbf{F}_{ij} = G m_i \sum_{j \neq i}^N \frac{m_j\, \mathbf{r}_{ij}}{r_{ij}^3}$$

---

## **Simulaciones: N-cuerpos**

La fuerza crece arbitrariamente cuando dos partículas están muy cercanas. Para galaxias (sistemas no colisionales) no es necesario resolver la dinámica entre dos partículas cercanas:

$$\mathbf{F}_i \approx G m_i \sum_{j}^N \frac{m_j\, \mathbf{r}_{ij}}{(r_{ij}^2 + \epsilon^2)^{3/2}}$$

El factor $\epsilon$ se llama *softening length*. En la práctica necesitamos la aceleración: $\mathbf{a}_i = \mathbf{F}_i / m_i$.

---

## **Simulaciones: N-cuerpos**

Para actualizar las velocidades y posiciones usamos un integrador *leapfrog-Verlet*:

$$v_{i+1/2} = v_{i-1/2} + a_i \, \Delta t$$
$$x_{i+1} = x_i + v_{i+1/2} \, \Delta t$$

---

## **Paralelización del cálculo**

- El algoritmo calcula cada elemento de una matriz $\mathbf{F}_{ij}$ de $N \times N$ de fuerzas entre pares de partículas.
- La fuerza total en la partícula $i$ es la suma de la fila $i$ de la matriz.
- Se puede calcular cada elemento independientemente (paralelismo $\mathcal{O}(N^2)$), pero requeriría memoria $\mathcal{O}(N^2)$ y estaría muy limitado por *bandwidth*.
- En esta implementación se utiliza un *tile*: una sub-matriz de $p \times p$ elementos.

---

## **Paralelización del cálculo**

![w:520px](images/aplicaciones/p_tile.png)

Usamos los datos de $2p$ partículas para calcular $p^2$ valores de la matriz $\mathbf{F}_{ij}$. Las filas se calculan en paralelo, las columnas en forma secuencial.

<p class="credit">Fuente: GPU Gems Vol. 3</p>

---

## **Cálculo de la fuerza entre un par de partículas**

```cuda
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
  float3 r;
  // r_ij  [3 FLOPS]
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
  // invDistCube = 1/distSqr^(3/2)  [4 FLOPS]
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f/sqrtf(distSixth);
  // s = m_j * invDistCube  [1 FLOP]
  float s = bj.w * invDistCube;
  // a_i = a_i + s * r_ij  [6 FLOPS]
  ai.x += r.x * s;  ai.y += r.y * s;  ai.z += r.z * s;
  return ai;
}
```

---

## **Cálculo de la fuerza entre un par de partículas**

- Se usa `float4` para los datos en la memoria del *device*, para ayudar con el acceso contiguo. El valor `w` corresponde a la masa.
- El uso de `float3` en la función es para variables locales, donde hay que reducir el espacio usado en registros.

---

## **Cálculo de los tiles**

- Los datos de $p$ partículas se cargan de la memoria global a la memoria compartida para un bloque.
- Cada *thread* en el bloque calcula $p$ interacciones.
- El resultado del cálculo de un *tile* son $p$ aceleraciones parcialmente actualizadas.

---

## **Cálculo de los tiles**

```cuda
__device__ float3 tile_calculation(float4 myPosition, float3 accel)
{
  int i;
  extern __shared__ float4[] shPosition;
  for (i = 0; i < blockDim.x; i++){
    accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
  }
  return accel;
}
```

Cada *thread* ejecuta esta función para las mismas $p$ partículas. El acceso a memoria compartida es de tipo *broadcast*: no hay conflictos de bancos.

---

## **Varios tiles en un bloque**

Cada bloque tiene $p$ *threads* y calcula $N/p$ *tiles* en forma secuencial. Antes de calcular un *tile*, cada *thread* copia los datos de una partícula a la memoria compartida; al terminar, hay sincronización.

![w:460px](images/aplicaciones/multiple_tiles.png)

<p class="credit">Fuente: GPU Gems Vol. 3</p>

---

## **Cálculo de todas las fuerzas**

```cuda
__global__ void calculate_forces(void *devX, void *devA)
{
  extern __shared__ float4[] shPosition;
  float4 *globalX = (float4 *)devX;
  float4 *globalA = (float4 *)devA;
  float4 myPosition;
  int i, tile;
  float3 acc = {0.0f, 0.0f, 0.0f};
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  myPosition = globalX[gtid];
  for (i = 0, tile = 0; i < N; i += p, tile++) {
    int idx = tile * blockDim.x + threadIdx.x;
    shPosition[threadIdx.x] = globalX[idx];
    __syncthreads();
    acc = tile_calculation(myPosition, acc);
    __syncthreads();
  }
  float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
  globalA[gtid] = acc4;  // guardar para la integración temporal
}
```

---

## **Todas las partículas**

- Con $p$ *threads* por bloque y $N$ partículas, necesitamos $N/p$ bloques en la grilla.
- Esto resulta en $N$ *threads* calculando $N$ interacciones ($N^2$ en total).

---

## **Todas las partículas**

![w:520px](images/aplicaciones/full_calc_nbody.png)

Ejemplo 1: [nbody.cu](../code/aplicaciones/nbody.cu) (basado en un ejemplo del *CUDA Toolkit*).

<p class="credit">Fuente: GPU Gems Vol. 3</p>

---

# OpenGL

---

## **OpenGL**

- Creado en 1992 por Silicon Graphics.
- Un API para estandarizar el uso de gráficos en 3D.
- Actualmente el estándar lo mantienen el *Khronos Group* y el *OpenGL Architectural Review Board* (ARB).
- Basado en el proceso de **rasterización**.

---

## **Rasterización**

- Los objetos 3D se especifican por triángulos, vértices y líneas.
- El programador puede definir la posición y el ángulo de una cámara.
- Rasterización: para cada triángulo en el espacio 3D, identificar en cuáles *pixeles* de la pantalla se proyectaría.

![w:520px](images/aplicaciones/raytracing-raster.png)

---

## **OpenGL**

Interfaz para el programador al *hardware* del GPU. El API especifica:
- **primitivos**: puntos, líneas, polígonos.
- **propiedades**: colores, luces, texturas, etc.
- **vista**: posición de la cámara y perspectiva.

Conexión CUDA ↔ OpenGL: se puede referir a datos en la memoria del GPU (generados por CUDA) directamente desde OpenGL.

---

## **Interoperabilidad**

- **CUDA C**: gestión de la memoria con `malloc`, punteros, etc.
- **OpenGL**: guarda sus datos en *buffers* abstractos y genéricos, los *buffer objects*.
- Concepto principal: *map*/*unmap* de un *buffer* de OpenGL en el espacio de memoria de CUDA.
- Hay que incluir `cuda_gl_interop.h`.

---

## **Un ejemplo en detalle**

Ejemplo 2: [simpleGL.cu](../code/aplicaciones/simpleGL.cu) (basado en un ejemplo del *CUDA Toolkit*).

Pasos:
1. Crear un *buffer object* de vértices (VBO) vacío.
2. Registrar el VBO con CUDA.
3. *Map* del VBO para que acepte datos de CUDA.
4. Invocar un *kernel* para modificar las posiciones de los vértices.
5. *Unmap* del VBO.
6. Dibujar los resultados con OpenGL.

---

## **Un ejemplo en detalle**

El uso de OpenGL difiere entre sistemas operativos. Aquí lo vemos en Linux.

```cuda
#define GL_GLEXT_PROTOTYPES
#include <GL/freeglut.h>
```

Hay que tener instalado OpenGL, OpenGL Utility y FreeGLUT (instalar FreeGLUT instala los otros). Para compilar:

```sh
nvcc nombre_del_programa.cu -lglut -lGL -lGLU
```

---

## **Un ejemplo en detalle**

Variables globales:

```cuda
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
```

- La primera es un `unsigned int` definido en OpenGL, para referir a un VBO.
- La segunda es un puntero a un tipo definido en `cuda_gl_interop.h`, para apuntar al VBO desde CUDA.

---

## **Un ejemplo en detalle**

```cuda
int main(int argc, char **argv)
{
  setenv("DISPLAY", ":0", 0);  // define una variable del entorno
  runTest(argc, argv);
}
```

---

## **Un ejemplo en detalle**

```cuda
bool runTest(int argc, char **argv)
{
  if (false == initGL(&argc, argv)) return false;  // inicializar OpenGL
  // registrar los *callbacks*
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutCloseFunc(cleanup);
  // crear el VBO
  createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
  runCuda(&cuda_vbo_resource);  // ejecutar la parte de CUDA
  glutMainLoop();  // OpenGL dibuja la pantalla y escucha eventos
  return true;
}
```

---

## **Un ejemplo en detalle**

```cuda
bool initGL(int *argc, char **argv)
{
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(window_width, window_height);
  glutCreateWindow("Cuda GL Interop (VBO)");
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMotionFunc(motion);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glDisable(GL_DEPTH_TEST);
  glViewport(0, 0, window_width, window_height);              // viewport
  glMatrixMode(GL_PROJECTION);                                // projection
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);
  return true;
}
```

---

## **Un ejemplo en detalle**

```cuda
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
  glGenBuffers(1, vbo);                       // crear el buffer object
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);  // registrar con CUDA
}
```

---

## **Un ejemplo en detalle**

```cuda
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
  // mapear el buffer object de OpenGL para escribir datos desde CUDA
  float4 *dptr;
  cudaGraphicsMapResources(1, vbo_resource, 0);
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource);
  launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);
  cudaGraphicsUnmapResources(1, vbo_resource, 0);  // unmap
}
```

---

## **Un ejemplo en detalle**

```cuda
void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time)
{
  dim3 block(8, 8, 1);
  dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  simple_vbo_kernel<<< grid, block >>>(pos, mesh_width, mesh_height, time);
}
```

---

## **Un ejemplo en detalle**

```cuda
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width,
                                  unsigned int height, float time)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  // coordenadas uv
  float u = x / (float) width;
  float v = y / (float) height;
  u = u*2.0f - 1.0f;
  v = v*2.0f - 1.0f;
  // patrón de onda sinusoidal
  float freq = 4.0f;
  float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;
  pos[y*width+x] = make_float4(u, v, w, 1.0f);  // vértice de salida
}
```

---

## **Un ejemplo en detalle**

```cuda
void display()
{
  runCuda(&cuda_vbo_resource);  // generar posiciones de los vértices
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);   // matriz de la vista
  glLoadIdentity();
  glTranslatef(0.0, 0.0, translate_z);
  glRotatef(rotate_x, 1.0, 0.0, 0.0);
  glRotatef(rotate_y, 0.0, 1.0, 0.0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);              // render del VBO
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glColor3f(1.0, 0.0, 0.0);
  glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
  glDisableClientState(GL_VERTEX_ARRAY);
  glutSwapBuffers();
  g_fAnim += 0.01f;
}
```

---

## **Un ejemplo en detalle**

Hay otras funciones que se llaman *event handlers*:
- `timerEvent`: corre después de cierto tiempo.
- `keyboard`: responde si el usuario toca el teclado.
- `mouse`: responde a los botones del *mouse*/*trackpad*.
- `motion`: responde al movimiento del *mouse*/*trackpad*.

---

## **Visualización: *snapshots* de simulaciones de N-cuerpos**

Ejemplo 3: `nbody_viewer.cu`.

---

# Ray tracing

Peter Shirley: *Ray Tracing In One Weekend* · Roger Allen: *RTIOW In CUDA*

---

## **Ray tracing: introducción**

Escribiremos un programa que aplica la técnica de *ray-tracing*: la luz en una escena se calcula según las trayectorias de los rayos de luz.

![w:480px](images/aplicaciones/rtx.png)

<p class="credit">Fuente: IGN</p>

---

## **Ray tracing: introducción**

![w:620px](images/aplicaciones/ray_tracing.jpeg)

---

## **Ray tracing: introducción**

- El programa estará escrito en **C++**.
- Por lo tanto, aprenderemos un poco de **programación orientada a objetos**.
- Este paradigma es muy útil para aplicaciones como *ray-tracing*.

---

## **Ray tracing — Capítulo 1**

- Usaremos un formato de imagen llamado *ppm*: se puede guardar la imagen en un archivo de texto.
- El primer programa crea una imagen con gradientes de colores.

---

## **Ray tracing — Capítulo 1**

Ejemplo de una imagen en formato *ppm*:

```text
P3
1200 600
255
0 255 51
0 255 51
0 255 51
...
```

---

## **Ray tracing — Capítulo 1**

- Primera línea: `P3` define que el formato es ASCII.
- Segunda línea: número de *pixeles* en $x$ e $y$.
- Tercera línea: valor máximo del color por canal. $255$ corresponde a colores de $8$-bit (se puede usar hasta $65535$, $16$-bit).

---

## **Ray tracing — Capítulo 1**

- Librería de entrada/salida en C++: `iostream`.
- Para imprimir al terminal/archivo se usa el operador de *stream*: `std::cout`.
- `std::cout` es *standard out* (el terminal), `std::cerr` es *standard error*.

---

## **Ray tracing — Capítulo 1**

```cuda
int main() {
  int nx = 1200, ny = 600, tx = 8, ty = 8;
  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";
  int num_pixels = nx*ny;
  size_t fb_size = 3*num_pixels*sizeof(float);
  float *fb;
  cudaMallocManaged((void **)&fb, fb_size);   // "frame buffer"
  clock_t start = clock();
  dim3 blocks(nx/tx+1, ny/ty+1);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, nx, ny);
  cudaGetLastError();
  cudaDeviceSynchronize();
  double timer_seconds = ((double)(clock() - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--)
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j*3*nx + i*3;
      int ir = int(255.99*fb[pixel_index+0]);
      int ig = int(255.99*fb[pixel_index+1]);
      int ib = int(255.99*fb[pixel_index+2]);
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  cudaFree(fb);
}
```

---

## **Ray tracing — Capítulo 1**

```cuda
__global__ void render(float *fb, int max_x, int max_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x*3 + i*3;
  fb[pixel_index + 0] = float(i) / max_x;
  fb[pixel_index + 1] = float(j) / max_y;
  fb[pixel_index + 2] = 0.2;
}
```

---

## **Ray tracing — Capítulo 1**

```sh
nvcc -arch=sm_50 ch01_rt.cu -o ch01_rt.x
./ch01_rt.x > out.ppm
```

![w:360px](images/aplicaciones/ch01_rt.png)

---

## **Ray tracing — Capítulo 2**

- Comenzamos con aspectos de **programación orientada a objetos**.
- Para visualizaciones tridimensionales necesitamos vectores en 3D.
- No existe un tipo de datos básico en C/C++ que corresponda a un vector tridimensional.

---

## **Ray tracing — Capítulo 2**

- Definimos una clase de vectores en 3D, que llamamos `vec3`.
- Podemos crear **objetos** que pertenecen a esa clase (un vector en 3D).
- Estos objetos tendrán **atributos** (propiedades, p. ej. longitud) y **métodos** (funciones, p. ej. producto punto).

---

## **Ray tracing — Capítulo 2** (`vec3.h`)

```cuda
#ifndef VEC3H
#define VEC3H
#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {
public:
  __host__ __device__ vec3() {}
  __host__ __device__ vec3(float e0, float e1, float e2) {
    e[0] = e0; e[1] = e1; e[2] = e2;
  }
```

---

## **Ray tracing — Capítulo 2** (`vec3.h`)

```cuda
  __host__ __device__ inline float x() const { return e[0]; }
  __host__ __device__ inline float y() const { return e[1]; }
  __host__ __device__ inline float z() const { return e[2]; }
  __host__ __device__ inline float r() const { return e[0]; }
  __host__ __device__ inline float g() const { return e[1]; }
  __host__ __device__ inline float b() const { return e[2]; }

  __host__ __device__ inline const vec3& operator+() const { return *this; }
  __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  __host__ __device__ inline float operator[](int i) const { return e[i]; }
  __host__ __device__ inline float& operator[](int i) { return e[i]; }
```

---

## **Ray tracing — Capítulo 2** (`vec3.h`)

```cuda
  __host__ __device__ inline vec3& operator+=(const vec3 &v2);
  __host__ __device__ inline vec3& operator-=(const vec3 &v2);
  __host__ __device__ inline vec3& operator*=(const vec3 &v2);
  __host__ __device__ inline vec3& operator/=(const vec3 &v2);
  __host__ __device__ inline vec3& operator*=(const float t);
  __host__ __device__ inline vec3& operator/=(const float t);

  __host__ __device__ inline float length() const {
    return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
  }
  __host__ __device__ inline float squared_length() const {
    return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
  }
  __host__ __device__ inline void make_unit_vector();
  float e[3];
};
```

---

## **Ray tracing — Capítulo 2** (`vec3.h`)

```cuda
__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}
__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0]-v2.e[0], v1.e[1]-v2.e[1], v1.e[2]-v2.e[2]);
}
__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
__host__ __device__ inline vec3 operator/(vec3 v, float t) {
  return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}
```

(El archivo completo `vec3.h` incluye también `dot`, `cross`, los `operator*=`/`/=`, `make_unit_vector` y `unit_vector`.)

---

## **Ray tracing — Capítulo 2**

De `float*` a `vec3*` en el *kernel*:

```cuda
__global__ void render(vec3 *fb, int max_x, int max_y) {
  ...
  int pixel_index = j*max_x + i;
  fb[pixel_index] = vec3( float(i)/max_x, float(j)/max_y, 0.2f );
}
```

![w:300px](images/aplicaciones/ch02_out.png)

---

## **Ray tracing — Capítulo 3**

Ahora definimos un **rayo**. Matemáticamente: $\vec{P}(t) = \vec{A} + t\,\vec{b}$.

$\vec{A}$ es el origen del rayo, $\vec{b}$ la dirección, y $t$ un parámetro que define dónde estamos en el rayo.

![w:440px](images/aplicaciones/ray.png)

<p class="credit">Fuente: <em>Ray Tracing In One Weekend</em></p>

---

## **Ray tracing — Capítulo 3** (`ray.h`)

```cuda
#ifndef RAYH
#define RAYH
#include "vec3.h"
class ray
{
public:
  __device__ ray() {}
  __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
  __device__ vec3 origin() const { return A; }
  __device__ vec3 direction() const { return B; }
  __device__ vec3 point_at_parameter(float t) const { return A + t*B; }
  vec3 A;
  vec3 B;
};
#endif
```

---

## **Ray tracing — Capítulo 3**

Mandamos los rayos a la escena...

![w:480px](images/aplicaciones/camera_geometry.png)

<p class="credit">Fuente: <em>Ray Tracing In One Weekend</em></p>

---

## **Ray tracing — Capítulo 3**

```cuda
__global__ void render(vec3 *fb, int max_x, int max_y,
    vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  ray r(origin, lower_left_corner + u*horizontal + v*vertical);
  fb[pixel_index] = color(r);
}
```

---

## **Ray tracing — Capítulo 3**

```cuda
__device__ vec3 color(const ray& r) {
  vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5f*(unit_direction.y() + 1.0f);
  return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}
```

```cuda
render<<<blocks, threads>>>(fb, nx, ny,
    vec3(-2.0, -1.0, -1.0),
    vec3( 4.0,  0.0,  0.0),
    vec3( 0.0,  2.0,  0.0),
    vec3( 0.0,  0.0,  0.0));
```

![w:300px](images/aplicaciones/ch03_out.png)

---

## **Ray tracing — Capítulo 4**

Hay que poner objetos en la escena. El más fácil de representar es una **esfera**.

La ecuación de una esfera en forma vectorial: $(\vec{P}-\vec{C}) \cdot (\vec{P}-\vec{C}) = r^2$.

Un rayo choca con la esfera si existe un $t$ para el cual $(\vec{P}(t)-\vec{C}) \cdot (\vec{P}(t)-\vec{C}) = r^2$.

---

## **Ray tracing — Capítulo 4**

$$(\vec{A} + t\vec{b} - \vec{C}) \cdot (\vec{A} + t\vec{b} - \vec{C}) = r^2$$
$$t^2 b + 2t\,\vec{b} \cdot (\vec{A}-\vec{C}) + (\vec{A}-\vec{C})\cdot(\vec{A}-\vec{C}) - r^2 = 0$$

El único desconocido es $t$: es una ecuación **cuadrática** en $t$.

![w:360px](images/aplicaciones/sphere.png)

<p class="credit">Fuente: <em>Ray Tracing In One Weekend</em></p>

---

## **Ray tracing — Capítulo 4**

```cuda
__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0f * dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - 4.0f*a*c;
  return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) {
  if (hit_sphere(vec3(0,0,-1), 0.5, r))
    return vec3(1,0,0);
  vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5f*(unit_direction.y() + 1.0f);
  return (1.0f-t)*vec3(1.0,1.0,1.0) + t*vec3(0.5,0.7,1.0);
}
```

![w:280px](images/aplicaciones/ch04_out.png)

---

## **Redes neuronales: cuDNN**

Un ejemplo de convoluciones con cuDNN:

<http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/>

---

# ¡Gracias!
