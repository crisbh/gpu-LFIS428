---
marp: true
paginate: true
math: katex
html: true
style: |
  section {
    font-size: 2.2em !important;
    font-family: 'Arial', sans-serif;
    overflow: hidden;
  }
  img {
    display: block;
    margin: auto;
    width: 70%; /* Increase size */
    max-width: 70%;
  }
  .video-container {
    position: relative;
    padding-bottom: 56.25%; /* 16:9 aspect ratio */
    height: 0;
    overflow: hidden;
    max-width: 100%;
    background: black;
  }
  .video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }
---

# **Programación en GPUs**
## Clase 1
## Introducción a CUDA

Lenguaje: **CUDA/C** (GPUs de NVIDIA)

---

## **Información sobre el curso**

Libros de referencia:
- *Learn CUDA Programming* — Han, Sharma
- *Professional CUDA C Programming* — Cheng, Grossman, McKercher
- *Parallel Programming: Concepts and Practice* — Schmidt, González-Domínguez, Hundt, Schlarb
- *Hands-On GPU Programming with Python and CUDA* — Tuomanen

El lenguaje del curso es **CUDA/C**, pero veremos un poco sobre cómo interactuar con CUDA a través de **Python**.

---

## **Programa del curso**

1. Introducción a CUDA
2. **Prueba** (Evaluación 1)
3. El uso de la memoria del GPU
4. Control de los *threads*
5. **Tarea** (Evaluación 2)
6. Invocación de los *kernels*
7. Librerías de CUDA y Python
8. **Tarea** (Evaluación 3)
9. Aplicaciones (N-body, ray-tracing, OpenGL)
10. **Proyecto final** (Deep Learning, Evaluación 4)

---

## **Códigos**

Cada capítulo tiene una carpeta con programas de ejemplo.

Los códigos de esta clase están disponibles para descargar:
- [hola_mundo.cu](../code/intro/hola_mundo.cu)
- [suma_vectores_host.c](../code/intro/suma_vectores_host.c)
- [suma_vectores_gpu.cu](../code/intro/suma_vectores_gpu.cu)
- [mostrarIndices.cu](../code/intro/mostrarIndices.cu)
- [simpleDeviceQuery.cu](../code/intro/simpleDeviceQuery.cu)

---

# Introducción a CUDA

---

## **¿Por qué GPUs?**

![w:55%](images/use_of_gpu.png)

Fuente: nvidia.com

---

## **Programación heterogénea**

![](images/hetero_arch.png)

El cómputo se reparte entre el **host** (CPU) y el **device** (GPU).

Fuente: *Professional CUDA C Programming*

---

## **GPU Hardware**

![w:60%](images/modern_gpu_performance.png)

Fuente: NVIDIA Developer Blog

---

## **CPU vs GPU**

![w:60%](images/cpu_vs_gpu.png)

Fuente: *Professional CUDA C Programming*

---

## **Thread del GPU vs thread del CPU**

- *Threads* en el **CPU** son "pesados": el *context switching* es costoso. Los *cores* del CPU minimizan la *latency* para uno o dos *threads*.
  - Un CPU de 4 procesadores *quad-core* puede ejecutar 16 *threads* a la vez (32 con *hyper-threading*).
- *Threads* en el **GPU** son "livianos": hay miles disponibles y el *context switching* es rápido. Los *cores* manejan muchos *threads* para maximizar el *throughput*.
  - Ejemplo: un GPU con 16 multiprocesadores y 1536 *threads* activos por multiprocesador alcanza $> 24000$ *threads* activos simultáneamente.

---

## **Un poco de jerga**

- **Thread**: secuencia de instrucciones, manejada por un *scheduler* (reparte el tiempo del procesador entre *threads*/procesos).
- **Context switching**: parar la operación de un *thread* para permitir la de otro.
- **Latency** (latencia): retraso entre emitir una instrucción y recibir los datos que pide.
- **Throughput**: cantidad de datos que pasan por una red de comunicación por unidad de tiempo (típicamente GB/s).
- **Bandwidth**: máximo teórico del *throughput* de una red de comunicación.

---

## **El compilador NVCC**

![w:55%](images/nvcc_compiler.png)

- Código del **host**: corre en el CPU.
- Código del **device**: corre en el GPU.

Fuente: *Professional CUDA C Programming*

---

## **¿Tengo un GPU?**

En el *shell* de Linux:

```sh
nvidia-smi
```

También se puede usar:

```sh
lspci | grep NVIDIA
```

---

# Primer programa de CUDA

---

## **¡Hola Mundo! con CUDA**

Ejemplo 1: [hola_mundo.cu](../code/intro/hola_mundo.cu)

```cpp
#include <stdio.h>
#include <stdlib.h>

__global__ void imprimir_desde_el_gpu() {
  printf("Hola Mundo! desde el thread [%d,%d] del device\n",
         threadIdx.x, blockIdx.x);
}

int main() {
  printf("Hola Mundo desde el host!\n");
  imprimir_desde_el_gpu<<<1, 10>>>();
  cudaDeviceSynchronize();
  return 0;
}
```

Compilar con `nvcc -arch=sm_50 hola_mundo.cu -o hola_mundo.x` (el valor de `-arch` depende del GPU).

---

# Un programa más útil

---

## **Suma de vectores**

![w:60%](images/vector_addition.png)

Fuente: *Professional CUDA C Programming*

---

## **Suma de vectores: host**

Ejemplo 2a: [suma_vectores_host.c](../code/intro/suma_vectores_host.c)

```c
#define N 512

void suma_host(int *a, int *b, int *c) {
  for (int idx = 0; idx < N; idx++)
    c[idx] = a[idx] + b[idx];
}
```

Compilar con `gcc suma_vectores_host.c -o suma_vectores_host.x`.

---

## **Suma de vectores: device**

Ejemplo 2b: [suma_vectores_gpu.cu](../code/intro/suma_vectores_gpu.cu)

```cpp
__global__ void suma_device(int *a, int *b, int *c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  c[idx] = a[idx] + b[idx];
}
```

- No hay ciclo `for`: las coordenadas de los *threads* reemplazan el índice del ciclo.
- `N` queda definida implícitamente al lanzar el *kernel* con `N` *threads*.

Compilar con `nvcc -arch=sm_50 suma_vectores_gpu.cu -o suma_vectores_gpu.x`.

---

## **Suma de vectores: manejo de memoria**

```cpp
cudaMalloc((void **)&d_a, size);            // asignar memoria en el device
cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  // host -> device
suma_device<<<2, N / 2>>>(d_a, d_b, d_c);   // invocar kernel
cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);  // device -> host
```

- `cudaMalloc`: asignar memoria en el *device*.
- `cudaMemcpy`: copiar datos entre el *host* y el *device* (en ambas direcciones).

Más funciones en la documentación del **CUDA Runtime API**.

---

# Kernels

---

## **Utilizando el GPU**

- Para realizar un trabajo en el GPU hay que invocar un **kernel**.
- Un *kernel* es una función que corre en el GPU, con ciertas restricciones.

```cpp
__global__ void nombre_kernel(...) {
  // cuerpo de la función
}
```

Para invocarlo:

```cpp
nombre_kernel<<< N, M >>>(...);
```

Los valores de $N$ y $M$ controlan el número de *threads* que usa el *kernel*.

---

## **Restricciones para los kernels**

- Acceso a la memoria del *device* solamente.
- El tipo de retorno debe ser `void`.
- No se puede usar un número variable de argumentos.
- No se puede usar variables estáticas.
- No se puede usar punteros a funciones.
- Corren asincrónicamente.

---

## **Organización de los threads**

![w:55%](images/cuda_indexing.png)

- Todos los *threads* de un *kernel* forman un **grid** y comparten la memoria **global** del GPU.
- Un *grid* se compone de **bloques** de *threads*; cada bloque tiene su memoria **compartida**.
- Coordenadas únicas: `blockIdx` (índice del bloque en el *grid*) y `threadIdx` (índice del *thread* en el bloque).

---

## **Organización de los threads (dimensiones)**

Se puede organizar los *threads* en 1D, 2D o 3D. Las coordenadas son del tipo `uint3` (device):

- `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
- `threadIdx.x`, `threadIdx.y`, `threadIdx.z`

Dimensiones del *grid* y los bloques:

- `blockDim.x/y/z` (en *threads*)
- `gridDim.x/y/z` (en *bloques*)

---

## **Organización de los threads (host)**

En el *host* las dimensiones se especifican con el tipo `dim3`:

```cpp
dim3 bloques(bx, by, bz);
dim3 grid(gx, gy, gz);
nombre_kernel<<< grid, bloques >>>(...);
```

Para una distribución 2D, basta con dar dos valores (o poner $1$ en $z$):

```cpp
dim3 bloques(bx, by);
dim3 grid(gx, gy);
```

---

## **¡Importante!**

- Hay un límite de **$1024$ *threads* por bloque**, sin importar si es 1D, 2D o 3D.
  - 1D: hasta $1024$ en $x$.
  - 2D: $32 \times 32 = 1024$.
  - 3D: por ejemplo $16 \times 16 \times 4 = 1024$.
- Es fácil pasarse del límite y el error es difícil de detectar (más sobre esto en un momento).

---

## **Organización de los threads: ejemplo**

Ejemplo 3: [mostrarIndices.cu](../code/intro/mostrarIndices.cu)

```cpp
__global__ void mostrarIndices(void) {
  printf("threadIdx: (%d,%d,%d)  blockIdx: (%d,%d,%d)  "
         "blockDim: (%d,%d,%d)  gridDim: (%d,%d,%d)\n",
         threadIdx.x, threadIdx.y, threadIdx.z,
         blockIdx.x, blockIdx.y, blockIdx.z,
         blockDim.x, blockDim.y, blockDim.z,
         gridDim.x, gridDim.y, gridDim.z);
}

int main() {
  dim3 block(3, 3);
  dim3 grid(2, 2);
  mostrarIndices<<<grid, block>>>();
  cudaDeviceReset();
}
```

Compilar con `nvcc -arch=sm_50 mostrarIndices.cu -o mostrarIndices.x`.

---

## **Warps, bloques, grids**

- Los *threads* trabajan en grupos de $32$ llamados **warps**.
- Según el número de *threads* por bloque, cada bloque tendrá múltiples *warps*.
- Los *threads* de un *warp* están sincronizados implícitamente.
- Todos los *threads* de un bloque comparten un espacio de memoria compartida.
- **No** hay comunicación entre *threads* de distintos bloques.

---

## **Diseño de los kernels**

- Los *kernels* siguen el modelo **SPMD** (*single program, multiple data*).
- Un *kernel* es **código escalar** para un solo *thread*.
- Al invocarlo, muchos *threads* realizan la misma operación definida en el *kernel*.

---

## **Variedades de funciones en CUDA**

- `__global__`: ejecuta en el *device*; se llama desde el *host* (y desde el *device* para *compute capability* $> 3$).
- `__host__`: ejecuta en el *host*; se llama desde el *host* (normalmente no hay que especificarlo).
- `__device__`: ejecuta en el *device*; se llama desde el *device*.

Una función se puede compilar para *host* y *device* combinando `__host__` y `__device__`.

---

# Errores

---

## **Manejando errores**

- Siempre hay errores en un programa... y en CUDA son un poco difíciles de detectar.
- Todas las funciones del API de CUDA devuelven un `enum` (`cudaError_t`) con el tipo de error.

```cpp
cudaError_t err = cudaMemcpy(...);
cudaGetErrorString(err);
```

---

## **Manejando errores: macro**

Una forma mejor es usar un *macro*:

```cpp
#define CHECK(call)                                                   \
{                                                                     \
  const cudaError_t err = call;                                       \
  if (err != cudaSuccess) {                                           \
    printf("Error: %s:%d, ", __FILE__, __LINE__);                     \
    printf("codigo:%d, mensaje: %s\n", err, cudaGetErrorString(err)); \
    exit(1);                                                          \
  }                                                                   \
}
```

---

## **Manejando errores: kernels**

- ¡La invocación de un *kernel* **no devuelve nada**! Si falla, no aparece ningún mensaje de error.
- Ejemplo: invocar con demasiados *threads*.

```cpp
suma_device<<<1, 2048>>>(d_a, d_b, d_c);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
  printf("Error: %s\n", cudaGetErrorString(err));
```

Usamos `cudaGetLastError` para capturar el error. Más información en la documentación del API.

---

# Profiling

---

## **Profiling (perfilaje)**

Los *profilers* dan información sobre la ejecución (tiempo por función, uso de memoria, etc.). Para CUDA:

- **nvprof**: GPUs de *compute capability* $< 7$. Uso de recursos y tiempo de las funciones del API.
- **ncu** (NSight Compute): GPUs de CC $\geq 7$. Uso de recursos del GPU, transferencias de memoria, etc.
- **nsys** (NSight Systems): GPUs de CC $\geq 7$. Tiempo de ejecución de las funciones.

---

## **nvprof**

![w:60%](images/nvprof_example.png)

- Opciones: `nvprof --help`.
- Para el uso de recursos se usan **métricas**: `nvprof --query-metrics`.

---

## **Profilers visuales: NVVP**

![w:55%](images/nvvp.png)

```sh
nvprof --export-profile profile.nvvp --analysis-metrics ./programa
```

El archivo `profile.nvvp` se abre con NVVP (NVIDIA Visual Profiler).

---

## **Profilers visuales: NSight Compute**

![w:55%](images/ncu_example.png)

```sh
ncu -o informacion ./programa.x      # guarda informacion.ncu-rep
ncu --metrics <metrica> ./programa.x # información en pantalla
```

Se abre el `.ncu-rep` con NSight Compute (`ncu-ui`). Métricas: `ncu --query-metrics`.

---

## **Profilers visuales: NSight Systems**

![w:55%](images/nsys_example.png)

```sh
nsys profile ./programa.x              # guarda report.qdrep
nsys profile --stats=true ./programa.x # información en pantalla
```

El `.qdrep` se abre con NSight Systems (`nsys-ui`).

---

## **¿Acotado por el cómputo o por la memoria?**

- **Compute bound**: el rendimiento lo limita la rapidez de las operaciones aritméticas del GPU.
- **Memory bound**: el rendimiento lo limita la rapidez de la comunicación con la memoria del GPU.
- Casi **siempre** los programas de cómputo científico son *memory bound*.

En el próximo capítulo veremos cómo mejorar el uso de la memoria...

---

## **Información del GPU en el sistema**

Con el API de CUDA: `cudaGetDeviceProperties` — Ejemplo 4: [simpleDeviceQuery.cu](../code/intro/simpleDeviceQuery.cu)

```cpp
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
printf("Device %d: \"%s\"\n", dev, deviceProp.name);
```

En el *shell* de Linux: `nvidia-smi` o `lspci | grep NVIDIA`.

Más información en la documentación sobre *device management*.

---

# ¡Gracias!

## Próxima clase: el uso de la memoria del GPU
