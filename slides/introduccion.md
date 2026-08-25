---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

# **Programación en GPUs**
## Introducción a CUDA

---

## **Información sobre el curso**

El lenguaje del curso es **CUDA/C**, pero veremos un poco sobre cómo interactuar con CUDA a través de **Python**.

Libros de referencia:
- *Learn CUDA Programming* — Han, Sharma
- *Professional CUDA C Programming* — Cheng, Grossman, McKercher
- *Parallel Programming: Concepts and Practice* — Schmidt, González-Domínguez, Hundt, Schlarb
- *Hands-On GPU Programming with Python and CUDA* — Tuomanen


---

## **Programa del curso**

1. Introducción a CUDA
2. **Quiz** (Evaluación 1)
3. El uso de la memoria del GPU
4. Control de los *threads*
5. **Quiz** (Evaluación 2)
6. Invocación de los *kernels*
7. Librerías de CUDA y Python
8. Aplicaciones (N-body, ray-tracing)
9. **Proyecto final** (Evaluaciones 3 y 4)

---

## **Códigos**

Cada capítulo tiene asociado programas de ejemplo.

Durante las primeras clases usaremos:
- [hola_mundo.cu](../code/intro/hola_mundo.cu)
- [suma_vectores_host.c](../code/intro/suma_vectores_host.c)
- [suma_vectores_gpu.cu](../code/intro/suma_vectores_gpu.cu)
- [saxpy.cu](../code/intro/saxpy.cu)
- [mostrarIndices.cu](../code/intro/mostrarIndices.cu)
- [simpleDeviceQuery.cu](../code/intro/simpleDeviceQuery.cu)

---

<!-- _class: hook -->

## **Introducción a CUDA**

<p class="destacado">¿Por qué programar con GPUs?</p>

---

## **¿Por qué GPUs?**

![w:630px](images/use_of_gpu.png)

<p class="credit">Fuente: nvidia.com</p>

El cómputo se reparte entre el **host** (CPU) y el **device** (GPU).


---

## **Programación heterogénea**


![](images/hetero_arch.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **GPU Hardware**

![w:680px](images/modern_gpu_performance.png)

<p class="credit">Fuente: NVIDIA Developer Blog</p>

---

## **CPU vs GPU**

![w:680px](images/cpu_vs_gpu.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

<!-- _class: hook -->

## **Introducción a CUDA**

<p class="destacado">¿Qué entendemos por threads (hilos)?</p>

---

## **¿Qué es un thread?**

- Un *thread* es un **contexto virtual de ejecución** que es **asignado** a un core (núcleo) forma independiente por un *scheduler* para su ejecución.
  - Los CPU pueden tener **hyper-threading**, lo que permite que un *core* ejecute varios *threads* al mismo tiempo.
- Varios *threads* pueden avanzar de forma concurrente, ya sea repartiéndose el tiempo de un mismo *core* o corriendo en paralelo en distintos *cores*.

---

## **Thread del GPU vs thread del CPU**

- *Threads* del **CPU** son "pesados": el *context switching* es costoso. Los *cores* del CPU minimizan la *latency* para uno o dos *threads*.
  - Un CPU de 4 procesadores *quad-core* puede ejecutar 16 *threads* a la vez (32 con *hyper-threading*).

---

## **Thread del GPU vs thread del CPU**

- *Threads* del **GPU** son "livianos": el *context switching* es rápido y hay miles disponibles. Los *cores* manejan muchos *threads* para maximizar el *throughput*.
  - Ejemplo: un GPU con 16 multiprocesadores y 1536 *threads* activos por multiprocesador alcanza $> 24000$ *threads* activos simultáneamente.

---

## **Un poco de jerga**

- **Thread** (hilo): *contexto virtual de ejecución* que es *asignado* a un core (núcleo) forma independiente por un *scheduler* para su ejecución.
- **Context switching**: guardar el estado de ejecución de un thread (registros, program counter, stack) y cargar el de otro, para que el core pase de ejecutar uno a ejecutar el otro.
- **Latency** (latencia): retraso entre emitir una instrucción y recibir los datos que pide.

---

## **Un poco de jerga**

- **Throughput**: cantidad de trabajo (datos, operaciones, instrucciones) completado por unidad de tiempo.
  - Ejemplos: ancho de banda de memoria (GB/s), operaciones de punto flotante por segundo (FLOP/s)
- **Bandwidth** (ancho de banda): capacidad máxima de transferencia de un canal de comunicación (memoria, PCIe, NVLink, red), típicamente en GB/s. Es el máximo teórico del throughput, y es una propiedad del hardware.
- **Arithmetic intensity** (intensidad aritmética): número de operaciones aritméticas realizadas por byte transferido desde memoria. Determina si un kernel está limitado por cómputo o por ancho de banda.

---

## **El compilador NVCC**

![w:630px](images/nvcc_compiler.png)
<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

- Código del **host**: corre en el CPU.
- Código del **device**: corre en el GPU.


---

## **¿Tengo un GPU de NVIDIA?**

En el *shell* de Linux:

```sh
nvidia-smi
```

También se puede usar:

```sh
lspci | grep NVIDIA
```

---

## **Google Colab: acceso a la GPU T4 gratis**

[Google Colab](https://colab.research.google.com) ofrece GPUs **NVIDIA T4** gratuitas en la nube.

1. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución** (arriba a la derecha, al lado de *conectar*.
2. **Acelerador por hardware → GPU (T4)**.
3. Verificar el GPU asignado:

```sh
!nvidia-smi
```

<!-- El T4 tiene *compute capability* $7.5$. -->

---

# Primer programa de CUDA

---

## **¡Hola Mundo! con CUDA**

Ejemplo 1: [hola_mundo.cu](../code/intro/hola_mundo.cu)

@include[cuda]{static/code/intro/hola_mundo.cu}

Compilar con `nvcc -arch=sm_50 hola_mundo.cu -o hola_mundo.x` (el valor de `-arch` depende del GPU).


---

## **Compilar CUDA en Colab**

- En Colab, se pueden crear y editar archivos desde una termial (e.g. con `vi`)
- También se puede a través de una celda de Colab, se puede escribir un archivo `.cu` con `%%writefile`, bajo lo cual se pone el contenido

- 1. Crear un archivo `.cu` con el código de `hola_mundo.cu`.

```sh
%%writefile hola_mundo.cu
// ... código CUDA ...
```

2. En otra celda, compilar y ejecutar con `nvcc`.
<!-- (el T4 es CC $7.5$ → `-arch=sm_75`): -->

```sh
!nvcc -arch=sm_75 hola_mundo.cu -o hola_mundo.x && ./hola_mundo.x
```

---
# Un programa más útil

---

## **Suma de vectores**

![w:440px](images/vector_addition.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Suma de vectores: host**

Ejemplo 2a: [suma_vectores_host.c](../code/intro/suma_vectores_host.c)

@include[c]{static/code/intro/suma_vectores_host.c:5-10}

Compilar con `gcc suma_vectores_host.c -o suma_vectores_host.x`.

---

## **Suma de vectores: device**

Ejemplo 2b: [suma_vectores_gpu.cu](../code/intro/suma_vectores_gpu.cu)

@include[cuda]{static/code/intro/suma_vectores_gpu.cu:12-15}

- No hay ciclo `for`: cada *thread* calcula su índice global (`threadIdx.x + blockIdx.x * blockDim.x`) y procesa un elemento.
- El *kernel* **no verifica límites**: hay que lanzar exactamente `N` *threads*, o agregar `if (idx < N)` para evitar accesos fuera del arreglo.

Compilar con `nvcc -arch=sm_50 suma_vectores_gpu.cu -o suma_vectores_gpu.x`.

---

## **Suma de vectores: manejo de memoria**

```cuda
cudaMalloc((void **)&d_a, size);            // asignar memoria en el device
cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  // host -> device
suma_device<<<2, N / 2>>>(d_a, d_b, d_c);   // invocar kernel
cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);  // device -> host
```

- `cudaMalloc`: asignar memoria en el *device*.
  - Retorna un puntero en el *device*.
- `cudaMemcpy`: copiar datos entre el *host* y el *device* (en ambas direcciones).
  - 1er argumento: puntero destino. 2do argumento: puntero origen.

Más funciones en la documentación del **CUDA Runtime API**.

---

## **SAXPY**

La operación **SAXPY** (*single-precision A·X plus Y*): $Y = a \cdot X + Y$.

Ejemplo 3: [saxpy.cu](../code/intro/saxpy.cu)

@include[cuda]{static/code/intro/saxpy.cu:5-9}

- Patrón muy común en cómputo científico (BLAS nivel 1).
- El `if (i < n)` protege del exceso de *threads*; se lanza con `blocks = (N + threads - 1) / threads`.

---

# Los Kernels

---

## **Los Kernels: funciones para el GPU**

- Para realizar un trabajo en el GPU hay que invocar un **kernel**.
- Un *kernel* es una función que corre en el GPU, con ciertas restricciones.

```cuda
__global__ void nombre_kernel(...) {
  // cuerpo de la función
}
```

Para invocarlo:

```cuda
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

![w:430px](images/threads_hierarchy.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

- Los *threads* se organizan en un **grid** y comparten la memoria **global** del GPU.


---

## **Organización de los threads**

![w:630px](images/cuda_indexing.png)

- El *grid* se compone de **bloques** de *threads*; cada bloque tiene su memoria **compartida**.
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

```cuda
dim3 bloques(bx, by, bz);
dim3 grid(gx, gy, gz);
nombre_kernel<<< grid, bloques >>>(...);
```

Para una distribución 2D, basta con dar dos valores (o poner $1$ en $z$):

```cuda
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

## **Índices de los threads: el kernel**

Ejemplo 4: [mostrarIndices.cu](../code/intro/mostrarIndices.cu)

@include[cuda]{static/code/intro/mostrarIndices.cu:1-10}

Cada *thread* imprime sus coordenadas y las dimensiones del *grid* y del bloque.

---

## **Índices de los threads: el lanzamiento**

@include[cuda]{static/code/intro/mostrarIndices.cu:12-31}

---

## **Diseño de los kernels**

- Los *kernels* siguen el modelo **SPMD** (*single program, multiple data*).
- Un *kernel* es **código escalar** para un solo *thread*.
- Al invocarlo, muchos *threads* realizan la misma operación definida en el *kernel*.

---

## **Warps, bloques, grids**

- Los *threads* trabajan en grupos de $32$ llamados **warps**.
  - Los warps no son visibles para el programador, pero es importante para el rendimiento.
- Los *threads* de un *warp* avanzan sincronizados (*lock-step*) en GPUs anteriores a Volta.
  - Desde *compute capability* $7.0$ hay que usar `__syncwarp()` para garantizarlo.
- Cada bloque puede tener múltiples *warps*, según cuantos *threads* hay.
- Todos los *threads* de un bloque comparten un espacio de memoria compartida.
- **No** hay comunicación entre *threads* de distintos bloques.

---

## **Variedades de funciones en CUDA**

- `__global__`: ejecuta en el *device*; se llama desde el *host* (y desde el *device* para *compute capability* $\geq 3.5$).
- `__host__`: ejecuta en el *host*; se llama desde el *host* (normalmente no hay que especificarlo).
- `__device__`: ejecuta en el *device*; se llama desde el *device*.

Una función se puede compilar para *host* y *device* combinando `__host__` y `__device__`.

---

## **Ejercicio: el bug de los límites**

Descargar: [suma_vectores_limites.cu](../code/intro/ejercicios/suma_vectores_limites.cu)

- Se lanzan $4 \times 256 = 1024$ *threads*, pero $N = 1000$.
1. Ejecutar el código tal cual. ¿El resultado parece correcto?
2. Arreglar el *kernel* con una línea.
3. Probar ahora con $N = 1050$. ¿Qué problema hay ahora?

---

## **Ejercicio: un kernel más robusto (grid-stride)**

Un *grid-stride loop* hace que el *kernel* sea correcto para **cualquier** configuración de `<<<blocks, threads>>>`:

```cuda
__global__ void suma_device(int *a, int *b, int *c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int paso = blockDim.x * gridDim.x;   // total de threads
  for (int i = idx; i < n; i += paso)
    c[i] = a[i] + b[i];
}
```

4. Reescribir el *kernel* así y comprobar que funciona bien con $N = 1050$ y pocos bloques (e.g. `blocks = 1`).

---

# Errores

---

## **Manejando errores**

- Siempre hay errores en un programa... y en CUDA son un poco difíciles de detectar.
- Las funciones del API de CUDA devuelven un `enum` (`cudaError_t`) con el tipo de error.

```cuda
cudaError_t err = cudaMemcpy(...);
cudaGetErrorString(err);
```

---

## **Manejando errores: un macro útil**

Una forma conveniente es usar un *macro* — Ejemplo completo: [errores.cu](../code/intro/errores.cu)

@include[cuda]{static/code/intro/errores.cu:4-13}

Se envuelve cada llamada del API, e.g. `CHECK(cudaMalloc(...));`. Si falla, imprime el archivo, la línea y el mensaje, y termina el programa.

---

## **Manejando errores: kernels**

- Dado que la invocación de un *kernel* **no devuelve nada**, no aparece ningún mensaje de error si este falla!
- Ejemplo: invocar con demasiados *threads*.

```cuda
suma_device<<<1, 2048>>>(d_a, d_b, d_c);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
  printf("Error: %s\n", cudaGetErrorString(err));
```

- Usamos `cudaGetLastError` para capturar el error. 
- El ejemplo completo con el macro `CHECK` está en [errores.cu](../code/intro/errores.cu).

---

# Profiling

---

## **Profiling (perfilación)**

Los *profilers* dan información sobre la ejecución (tiempo por función, uso de memoria, etc.). Para CUDA:

- **nvprof** (y **nvvp**, su interfaz gráfica): herramienta *legacy*. Funciona hasta *compute capability* $7.x$ (incluida la **T4**, CC $7.5$); **no** soporta CC $\geq 8$ y fue **eliminada** en CUDA 13. Combina métricas de recursos y trazas del API — hoy ese rol se reparte entre `ncu` y `nsys`.
- **ncu** (Nsight Compute): análisis **por kernel** (CC $\geq 6.1$, Pascal en adelante): *occupancy*, uso de recursos, *memory workload* y modelo roofline.
- **nsys** (Nsight Systems): análisis a nivel de **sistema**: línea de tiempo de las llamadas del API, los *kernels* y las transferencias de memoria (host ↔ device).

---

## **nvprof**

![w:680px](images/nvprof_example.png)

- Opciones: `nvprof --help`.
- Para el uso de recursos se usan **métricas**: `nvprof --query-metrics`.

---

## **Profilers visuales: NVVP**

![w:630px](images/nvvp.png)

```sh
nvprof --export-profile profile.nvvp --analysis-metrics ./programa
```

El archivo `profile.nvvp` se abre con NVVP (NVIDIA Visual Profiler).

---

## **Profilers visuales: NSight Compute**

![w:630px](images/ncu_example.png)

---

## **Profilers visuales: NSight Compute**
```sh
ncu -o informacion ./programa.x      # guarda informacion.ncu-rep
ncu --metrics <metrica> ./programa.x # información en pantalla
```

Se abre el `.ncu-rep` con NSight Compute (`ncu-ui`). Métricas: `ncu --query-metrics`.

---

## **Profilers visuales: NSight Systems**

![w:630px](images/nsys_example.png)

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

## **Modelo Roofline**

<svg viewBox="0 0 620 250" style="width:52%" xmlns="http://www.w3.org/2000/svg">
  <!-- ejes -->
  <line x1="70" y1="205" x2="590" y2="205" stroke="#8895a7" stroke-width="2"/>
  <line x1="70" y1="205" x2="70" y2="25" stroke="#8895a7" stroke-width="2"/>
  <!-- roofline -->
  <polyline points="70,205 285,75 580,75" fill="none" stroke="#2a7ae2" stroke-width="4"/>
  <!-- linea del ridge point -->
  <line x1="285" y1="75" x2="285" y2="205" stroke="#8895a7" stroke-width="1.5" stroke-dasharray="5,5"/>
  <circle cx="285" cy="75" r="6" fill="#e8603c"/>
  <!-- etiquetas -->
  <text x="215" y="240" fill="#35495e" font-size="16">Intensidad aritmética (FLOP/byte)</text>
  <text x="30" y="140" fill="#35495e" font-size="16" transform="rotate(-90 30 140)">Rendimiento</text>
  <text x="120" y="135" fill="#2a7ae2" font-size="15" transform="rotate(-31 120 135)">ancho de banda</text>
  <text x="360" y="62" fill="#2a7ae2" font-size="15">peak de cómputo</text>
  <text x="105" y="190" fill="#6b7785" font-size="13">memory bound</text>
  <text x="390" y="190" fill="#6b7785" font-size="13">compute bound</text>
</svg>

- Rendimiento $\leq \min(\text{peak de cómputo},\ AI \times \text{ancho de banda})$, con $AI$ la intensidad aritmética (FLOP/byte).
- El **ridge point** (punto naranja) separa las regiones *memory bound* y *compute bound*.

---

## **Información del GPU en el sistema**

Con el API de CUDA: `cudaGetDeviceProperties` — Ejemplo 5: [simpleDeviceQuery.cu](../code/intro/simpleDeviceQuery.cu)

```cuda
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
printf("Device %d: \"%s\"\n", dev, deviceProp.name);
```

En el *shell* de Linux: `nvidia-smi` o `lspci | grep NVIDIA`.

Más información en la documentación sobre *device management*.

---

## Fin capítulo 1

Próxima clase: el uso de la memoria del GPU
