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

## **Diseño de los kernels**

- Los *kernels* siguen el modelo **SPMD** (*single program, multiple data*).
- Un *kernel* es **código escalar** para un solo *thread*.
- Al invocarlo, muchos *threads* realizan la misma operación definida en el *kernel*.

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

- Los *threads* se organizan de forma jerárquica en **grids** y **bloques**.
- El *grid* tiene **bloques** de *threads*, los que comparten memoria.

---

## **Organización de los threads (índices)**

![w:630px](images/cuda_indexing.png)

- `blockIdx`: índice del *bloque* en el *grid*.
- `threadIdx`: índice del *thread* en el *bloque* .
- `index`: combinación que sirve de índice global del *thread* en el *grid*.

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
  - Bloque 1D: hasta $1024$ en $x$.
  - Bloque 2D: e.g. $32 \times 32 = 1024$.
  - Bloque 3D: e.g. $16 \times 16 \times 4 = 1024$.

Ojo: Es fácil pasarse del límite y **es un error difícil de detectar** (más sobre esto en un momento).

---

## **Índices de los threads: el kernel**

Ejemplo 4: [mostrarIndices.cu](../code/intro/mostrarIndices.cu)

@include[cuda]{static/code/intro/mostrarIndices.cu:1-9}

Cada *thread* imprime sus coordenadas y las dimensiones del *grid* y del bloque.

---

## **Índices de los threads: el lanzamiento**

@include[cuda]{static/code/intro/mostrarIndices.cu:13-31}


---

## **Warps, bloques, grids**

- Los *threads* trabajan en grupos de $32$ llamados **warps**.
  - Los warps no son visibles para el programador, pero es importante para el rendimiento.
- Los *threads* de un *warp* avanzan sincronizados (*lock-step*) en GPUs anteriores a Volta.
  - Desde *compute capability* $7.0$ hay que usar `__syncwarp()` para garantizarlo.
- Cada bloque puede tener múltiples *warps*, según cuantos *threads* hay.
- Los *threads* de un bloque tienen un espacio de memoria compartida.
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

- Vectores de largo $N = 1000$. Se lanzan $4 \times 256 = 1024$ *threads*.
1. Ejecutar el código tal cual. ¿Qué problema existe?
2. Arreglar el *kernel* con una línea.
3. Probar ahora con $N = 1050$. ¿Qué problema hay ahora?

---

## **Ejercicio: un kernel más robusto (grid-stride)**

Un *grid-stride loop* es un patrón que hace que un *kernel* sea correcto para **cualquier** configuración de `<<<blocks, threads>>>`:

```cuda
__global__ void suma_device(int *a, int *b, int *c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int paso = blockDim.x * gridDim.x;   // total de threads
  for (int i = idx; i < n; i += paso)
    c[i] = a[i] + b[i];
}
```

4. Reescribir el *kernel* así y comprobar que funciona bien con $N = 1050$ y pocos bloques (e.g. `blocks = 1`).
5. ¿Como está operando el kernel en este caso?
---

# Errores

---

## **Manejando errores**

- Siempre hay errores cuando estamos implementando un programa... y en CUDA son un poco difíciles de detectar.
- **Todas** las funciones del API de CUDA devuelven un `enum` (`cudaError_t`) con el tipo de error.

Ejemplo:
```cuda
cudaError_t err = cudaMemcpy(...);
cudaGetErrorString(err);
```

- Notar que funciones como `cudaMemcpy` no necesitan retornar nada.

---

## **Nota: punteros dobles**

- Dado que el valor de retorno del API está reservado para `cudaError_t`: ¿Qué pasa con las funciones que necesitan retornar algo?
- Por ejemplo: `cudaMalloc` necesita retornar la **dirección** de la memoria asignada en el *device*. Según la documentación del API de CUDA:

```cuda
cudaError_t cudaMalloc(void **devPtr, size_t size);
```

- Es decir, pasamos la **dirección** del puntero: un *puntero a puntero*.
- Recordar que en C los argumentos se pasan **por valor**. 
  - Para que la función pueda modificar nuestro puntero `d_a`, le pasamos su dirección `&d_a` (**pasar por referencia**).

---

## **Nota: punteros dobles**

```cuda
int *d_a;                          // puntero (aún sin dirección válida)
cudaMalloc((void **)&d_a, size);   // pasamos la dirección de d_a para ser escrita
```

- En la práctica, gracias a C++ *overloading* y *templates*, es equivalente escribir:

```cuda
cudaMalloc(&d_a, size)
```

- Luego el compilador hace el *cast* del doble puntero a `(void **)` automáticamente para respetar la sintaxis.

---

## **Manejando errores: un macro útil**

Una forma conveniente es usar un *macro*:

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
- Revisar el ejemplo completo [errores.cu](../code/intro/errores.cu).

---

# Profilers

---

## **Profilers (perfiladores)**

- Una vez que nuestro programa funciona *correctamente*, queremos **optimizarlo**.

- Los *profilers* son herramientas que dan información sobre la ejecución (tiempo por función, uso de memoria, etc.). Para CUDA:

- **nvprof** (y **nvvp**, su interfaz gráfica): herramienta *legacy*. Funciona hasta *compute capability* $7.x$ (incluida la **T4**, CC $7.5$).

---

## **nvprof**

![w:680px](images/nvprof_example.png)

- Opciones: `nvprof --help`.
- Para el uso de recursos se usan **métricas**: `nvprof --query-metrics`.

---

## **Ejemplo: perfilar SAXPY con nvprof**

Podemos utilizar `nvprof` para ver un resumen rápido de **dónde se gasta el tiempo**:

```sh
!nvcc -arch=sm_75 saxpy.cu -o saxpy.x
!nvprof ./saxpy.x
```

- **GPU activities**: tiempo del *kernel* `saxpy` vs. las copias `[CUDA memcpy HtoD]` / `[DtoH]`.
- **API calls**: tiempo en `cudaMalloc`, `cudaMemcpy`, etc.

¿Qué domina: el *kernel* o las transferencias de memoria?

---

## **Profilers visuales: NVVP**

![w:500px](images/nvvp.png)

```sh
nvprof --export-profile profile.nvvp --analysis-metrics ./programa
```

El archivo `profile.nvvp` se abre con NVVP (NVIDIA Visual Profiler).

**No disponible en Colab (requiere interfaz gráfica)** 

---

## **Profilers: más allá de nvprof**

Para GPUs más modernos, se suele usar las herramientas de **Nsight**:

- **ncu** (Nsight Compute): análisis **por kernel** (CC $\geq 6.1$). 
  - *Occupancy*, uso de recursos, *memory workload*.
  - Modelo *roofline*.
- **nsys** (Nsight Systems): análisis a nivel de **sistema**.
  - Línea de tiempo de las llamadas del API y *kernels*.
  - Transferencias de memoria (host <-> device).

---

## **Profilers visuales: NSight Compute**

![w:630px](images/ncu_example.png)

---

## **Profilers visuales: NSight Compute**
```sh
ncu -o informacion ./programa.x      # guarda informacion.ncu-rep
ncu --metrics <metrica> ./programa.x # información en pantalla
```

Se abre el `.ncu-rep` con NSight Compute (`ncu-ui`). 
- Métricas: `ncu --query-metrics`.

---

## **Profilers visuales: NSight Systems**

![w:630px](images/nsys_example.png)

```sh
nsys profile ./programa.x              # guarda report.qdrep
nsys profile --stats=true ./programa.x # información en pantalla
```

El `.qdrep` se abre con NSight Systems (`nsys-ui`).

---

## **Profiling en Colab (sin interfaz gráfica)**

- Dado que en Colab no disponemos de GUI para visualizar los archivos `.ncu-rep` / `.nsys-rep`, preferimos usar la terminal.

- En Colab están `nvprof` (clásico) y `ncu`; `nsys` normalmente **no**.
- `nvprof ./prog.x` → resumen de tiempos (*kernel*, copias de memoria, llamadas del API).
- `ncu --set basic ./prog.x` → métricas **por kernel** (*Speed Of Light*, *occupancy*, uso de recursos, etc.).

También es posible descargar el `.ncu-rep` y abrirlo con `ncu-ui` localmente con GUI.

---

## **Ejemplo: perfilar SAXPY con ncu**

Usar [saxpy.cu](../code/intro/saxpy.cu) en Colab:

```sh
!nvcc -arch=sm_75 saxpy.cu -o saxpy.x
!ncu --set basic ./saxpy.x
```

Ver la sección **GPU Speed Of Light Throughput**: mide qué fracción del **máximo teórico** del hardware ("límite físico") alcanza el *kernel*:

- **Memory Throughput [%]**: % del *peak* de ancho de banda de memoria.
- **Compute (SM) Throughput [%]**: % del *peak* de cómputo.

El **mayor** de los dos indica el **cuello de botella** del *kernel*.

---
## Actividad

<p class="destacado">Actividad: Sistema Procesador + Memoria.</p>

---

## **¿Acotado por el cómputo o por la memoria?**

En general, el rendimiento de un programa de cómputo científico está **limitado por una de las siguientes razones** :

- **Compute bound**: el rendimiento lo limita la rapidez de las operaciones aritméticas del GPU (*FLOPS*: floating point operations).
- **Memory bound**: el rendimiento lo limita la rapidez de la comunicación con la memoria del GPU.

**La mayoría** de los programas de cómputo científico son *memory bound*.

En el próximo Capítulo veremos cómo mejorar el uso de la memoria.

---

## **La intensidad aritmética (AI)**

**Cómputo** realizado por cada **byte** movido desde/hacia la memoria global:

$$AI = \frac{\text{FLOP realizadas}}{\text{bytes leídos} + \text{bytes escritos}}$$

Se cuenta **por elemento** (el cuociente no depende de $N$):
- **FLOP**: cada `+`, `-`, `*`, `/` cuenta como $1$ operación (aunque `/` es técnicamente más costosa a nivel de hardware).
- **Bytes**: número de accesos $\times$ el tamaño del tipo (`float` $=4$ bytes).

**Ejemplo** — `c[i] = a[i] + b[i]`: $1$ FLOP y $12$ bytes, luego $AI \approx 0.08$.
<!-- Un $AI$ bajo: el *kernel* calcula poco y mueve muchos datos. -->

---

## **El Modelo Roofline**

<svg viewBox="0 0 620 250" style="width:52%; display:block; margin:0 auto;" xmlns="http://www.w3.org/2000/svg">
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
  <text x="30" y="115" fill="#35495e" font-size="16" text-anchor="middle" transform="rotate(-90 30 115)">Rendimiento (FLOP/s)</text>
  <text x="120" y="155" fill="#2a7ae2" font-size="15" transform="rotate(-31 120 135)">ancho de banda</text>
  <text x="360" y="62" fill="#2a7ae2" font-size="15">peak de cómputo</text>
  <text x="145" y="190" fill="#6b7785" font-size="13">memory bound</text>
  <text x="390" y="190" fill="#6b7785" font-size="13">compute bound</text>
</svg>

- Rendimiento $\leq \min(\text{peak de cómputo},\ AI \times \text{ancho de banda})$, en FLOP/s.
- El **ancho de banda** se mide en bytes/s (típicamente GB/s).
<!-- , de modo que $\frac{\text{FLOP}}{\text{byte}} \times \frac{\text{byte}}{\text{s}} = \frac{\text{FLOP}}{\text{s}}$. -->
- El **punto de inflexión** (punto naranja, *ridge point*) separa las regiones *memory bound* y *compute bound*.
- En la **T4** de Colab: *peak* de cómputo $\approx 8.1$ TFLOP/s y ancho de banda $\approx 320$ GB/s, luego el punto de inflexión está en $8100/320 \approx 25$ FLOP/byte.

---

## **Información del GPU en el sistema**

- Luego, la optimización involucra ambas cosas:
  - Nuestro programa (intensidad aritmética, uso de memoria).
  - Las características del GPU (número de *threads*, ancho de banda).

- Se pueden obtener detalles del GPU a través del API de CUDA: `cudaGetDeviceProperties` 

- Ejemplo: [simpleDeviceQuery.cu](../code/intro/simpleDeviceQuery.cu)

```cuda
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
printf("Device %d: \"%s\"\n", dev, deviceProp.name);
```

<!-- En el *shell* de Linux: `nvidia-smi` o `lspci | grep NVIDIA`. -->

Más información en la documentación sobre *device management*.

---

## **Ejercicio: ¿compute o memory bound?**

Usar [saxpy.cu](../code/intro/saxpy.cu) ($N = 2^{20}$).

1. Perfilar SAXPY con `ncu --set basic`. En **GPU Speed Of Light**: comparar **Compute (SM) Throughput [%]** con **Memory Throughput [%]**.
2. Calcular la **Intensidad Aritmética** de este kernel (`y[i] = a*x[i] + y[i]`).
3. ¿Es SAXPY *compute* o *memory bound*? Comparar con el modelo roofline.
4. Estimar el **ancho de banda efectivo** en base a los datos movidos y la duración del *kernel* (`Duration` en `ncu`).
<!-- RESPUESTAS. -->
<!-- 1. Medido en la T4: Memory Throughput $\approx 90$% contra Compute (SM) Throughput $\approx 5$-$15$%. El mayor de los dos es el cuello de botella, y acá gana la memoria por lejos. -->
<!-- 2. Por elemento: 2 FLOP (un `*` y un `+`) y 12 bytes: $8N$ leídos ($x$ e $y$) y $4N$ escritos ($y$). Luego $AI = 2/12 \approx 0.17$ FLOP/byte. El total con $N = 2^{20}$ es $12N = 12.58$ MB. -->
<!-- 3. Memory bound, y por mucho: $0.17$ está muy a la izquierda del punto de inflexión de la T4 ($\approx 25$ FLOP/byte), unas 150 veces menor. Coincide con lo observado en el punto 1. -->
<!-- 4. Ancho de banda efectivo = $12N$ bytes / Duration. Medido en la T4: $12582912$ B / $48.86$ µs = $257.5$ GB/s, o sea $80.5$% de los 320 GB/s. El profiler mide $255.79$ GB/s por su cuenta (ejercicio siguiente): la cuenta a mano acierta dentro de un $0.7$%. -->
<!-- OJO: ese $80.5$% NO tiene por qué coincidir con el $\approx 90$% de Memory Throughput del punto 1, y de hecho no coincide. Son dos cosas distintas y vale la pena detenerse acá. -->
<!-- Lo que calculamos a mano es tráfico de DRAM dividido por el peak teórico de 320 GB/s. Lo que reporta Speed Of Light es el MÁXIMO sobre toda la jerarquía de memoria (L1/TEX, L2 y DRAM) y contra el peak sostenido que mide la propia herramienta, no contra el número de la ficha técnica. -->
<!-- Y la medición del ejercicio siguiente lo zanja: `dram__bytes.sum.per_second` da $255.79$ GB/s, que es $79.9$% de 320. O sea la DRAM efectivamente va al $80$%, y el $90$% del Speed Of Light NO es la DRAM: es otra unidad del camino, casi con seguridad el pipe L1/TEX, que SAXPY castiga con 3 accesos de 32 bits por thread (dos lecturas y una escritura). -->
<!-- Se puede confirmar en clase con: `ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed ./saxpy.x`. Se espera `dram__throughput` $\approx 80$% y `l1tex` $\approx 90$%. -->
<!-- La moraleja para la clase es la que importa: un porcentaje de Speed Of Light no es un ancho de banda, y para comparar contra el roofline hay que usar los bytes absolutos. -->
<!-- Dos trampas al dividir. Primera, la unidad: `ncu` reporta `Duration` en µs o en ns según la magnitud. -->
<!-- Segunda, y es la que más se cobra alumnos: NO convertir los bytes a MB dividiendo por $2^{20}$. Eso da mebibytes, y los 320 GB/s de la T4 son decimales ($256$ bits $\times$ 10 Gbps $/ 8 = 320 \times 10^9$ B/s). Mezclar ambas convenciones da $245.6$ en vez de $257.5$, un $4.9$% ($= 2^{20}/10^6$) de error justo cuando se quiere comparar contra el techo. Lo correcto es dividir los bytes crudos por la duración y luego por $10^9$. -->

---

## **Ejercicio: ¿cuántos bytes se movieron realmente?**

Ahora ver **números absolutos** de memoria y compute. Usar:

```sh
!ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__bytes.sum.per_second ./saxpy.x
```

5. ¿Coinciden las lecturas y escrituras medidas con los $8N$ y $4N$ bytes del punto 2?
6. Comparar `dram__bytes.sum.per_second` con los $320$ GB/s de la T4, y con su estimación a mano del punto 4.

<!-- Si `ncu` rechaza alguna métrica: `ncu --query-metrics | grep dram__bytes`. -->
<!-- RESPUESTAS. -->
<!-- 5. Sí, coinciden muy de cerca: $\approx 8.39$ MB leídos ($8N$) y $\approx 4.19$ MB escritos ($4N$). -->
<!-- El "por qué" es lo interesante, y son dos razones. Primero, los accesos son perfectamente contiguos y coalescentes: cada warp pide sectores completos de 32 bytes y no se desperdicia ningún byte transferido. Segundo, `ncu` usa por defecto `--cache-control all`, es decir vacía las cachés del GPU antes de cada repetición del kernel, de modo que todo el tráfico llega efectivamente hasta la DRAM. -->
<!-- Corolario que vale la pena decir en voz alta: fuera del profiler el mismo kernel NO tiene por qué mover esos bytes. La L2 de la T4 es de 4 MB, justo el tamaño de un arreglo, así que parte de $y$ puede seguir en caché desde el `cudaMemcpy` previo y nunca releerse desde la DRAM. El modelo de 12 bytes por elemento es una cota superior del tráfico, no una predicción exacta. -->
<!-- 6. Medido en la T4: `dram__bytes.sum.per_second` $= 255.79$ GB/s, o sea $79.9$% de los 320. Coincide con la estimación a mano del punto 4 ($257.5$ GB/s) dentro de un $0.7$%, que es la validación que buscábamos: el modelo de $12N$ bytes describe bien el tráfico real. Si a un alumno le difiere mucho más, casi siempre es la unidad de `Duration` o la confusión MB/MiB del punto 4. -->
<!-- Este es además el número que zanja la discusión del punto 4. `dram__bytes.sum.per_second` es un ancho de banda de DRAM en GB/s absolutos, comparable directamente con los 320 GB/s, y da $79.9$%. El Memory Throughput [%] de Speed Of Light da $\approx 90$% porque es otra cosa: el máximo sobre toda la jerarquía de memoria, no la utilización de la DRAM. Los dos números son correctos y miden cosas distintas; para el roofline sirve este, el absoluto. -->


---

## **Ejercicio: la multiplicación matriz-vector**

Descargar: [matvec_roofline.cu](../code/intro/ejercicios/matvec_roofline.cu)

Producto $y = A x$, con $A$ de $N \times N$ y un *thread* por cada **fila** de $A$, es decir por cada elemento de $y$.

@include[cuda]{static/code/intro/ejercicios/matvec_roofline.cu:15-24}

```sh
!nvcc -arch=sm_75 matvec_roofline.cu -o matvec_roofline.x && !./matvec_roofline.x
!ncu --set basic ./matvec_roofline.x
```

---

## **Ejercicio: la multiplicación matriz-vector**

1. Para **un elemento de $y$**: ¿cuántos FLOP hace el *kernel* y cuántos bytes lee y escribe? Calcular la $AI$.
2. Ahora el **mínimo inevitable** para todo el producto: $A$ se lee una vez, $x$ una vez y $y$ se escribe una vez. ¿Qué $AI$ da? ¿Depende de $N$?
3. Repetir con $N = 2048$ y $N = 4096$ (cambiar el `#define N` y recompilar). ¿Cambia la $AI$ al crecer $N$? ¿Y la duración que reporta `ncu`?
<!-- 4. En la multiplicación de matrices la $AI$ mínima era $N/6$ y crecía sin límite; acá se queda fija. ¿Por qué? ¿Cuánto se puede ganar optimizando este *kernel*? -->
<!-- RESPUESTAS. -->
<!-- 1. Por elemento de y: 2N FLOP (N multiplicaciones y N sumas). El thread lee una fila entera de A (4N bytes) y TODO el vector x (4N bytes), y escribe 4: son 8N + 4 bytes. Luego AI = 2N/(8N+4) -> 0.25 FLOP/byte, otra vez independiente de N. -->
<!-- Cómo contar las FLOP, si hay dudas: FLOP = (cuántos elementos tiene la salida) x (costo de un elemento). El ladrillo es siempre un producto punto de largo N, o sea N multiplicaciones y N sumas = 2N FLOP. Acá la salida es un vector, N elementos, luego 2N^2 FLOP en total. En la multiplicación de matrices la salida tiene N^2 elementos y por eso da 2N^3: el ladrillo es idéntico, lo único que cambia es cuántos se hacen. -->
<!-- Detalle de las sumas: sumar N productos son estrictamente N-1 sumas, así que serían 2N-1 FLOP. Se usa 2N porque el código parte de suma = 0 y hace N sumas, porque la diferencia es 1 en 2N (0.05% a N = 1024), y sobre todo por consistencia con el techo: los 8.1 TFLOP/s de la T4 salen de 2560 cores x 2 x 1.59 GHz, y ese x2 está porque NVIDIA cuenta cada FMA (a*b+c, una sola instrucción) como 2 FLOP. -->
<!-- 2. A se lee una vez (4N^2), x una vez (4N) y se escribe y (4N): 4N^2 + 8N bytes. Con 2N^2 FLOP eso da AI = 2N^2/(4N^2+8N) -> 0.5 FLOP/byte. -->
<!-- Las dos AI de este ejercicio son límites, no valores exactos, y lo que se desprecia siempre es lo mismo: el vector, que es O(N), frente a la matriz, que es O(N^2). Conviene decirlo así en vez de "tomamos el límite". Valores exactos: 2N/(8N+4) para la ingenua y N/(2N+4) para la mínima. -->
<!-- Si alguien hace la división exacta en vez del límite, a N = 1024 le va a dar 0.499 y no 0.500. El culpable es el término 8N, y es buena ocasión para mostrar cómo converge: 0.444 a N=16, 0.485 a N=64, 0.496 a N=256, 0.499 a N=1024. Sólo importa para N chico, que es justamente el que nadie perfila. -->
<!-- (En la multiplicación de matrices, en cambio, la AI mínima N/6 es exacta.) -->
<!-- Y acá está lo importante del ejercicio: ese 0.5 es una CONSTANTE. Aunque se programe perfecto, esta operación jamás cruza el punto de inflexión (25). Comparar con la multiplicación de matrices, cuyo mínimo era N/6 y crecía sin techo. -->
<!-- 3. La AI no se mueve: ni la ingenua (0.25) ni la mínima (0.5) dependen de N, y por lo tanto tampoco se mueven los GFLOP/s ni los GB/s. Lo único que cambia es la duración, que crece como N^2: del orden de 16 microsegundos a N = 1024, 66 a N = 2048 y 262 a N = 4096, o sea un factor 16 entre la primera y la última. -->
<!-- Que haya que recompilar para cambiar N es a propósito: obliga a mirar el código. Y el resultado es el que importa: agrandar el problema NO mejora la posición en el roofline, sólo hace que tarde más. -->
<!-- (Esas duraciones salen de dividir el tráfico mínimo por los ~256 GB/s medidos en el ejercicio 1; son estimaciones, no mediciones. La duración real la da `ncu`.) -->
<!-- 4. Es la razón entre trabajo y datos, y es una propiedad del ALGORITMO, no de quien lo programa. Matriz-vector hace O(N^2) operaciones sobre O(N^2) datos: el cuociente es fijo. Matriz-matriz hace O(N^3) operaciones sobre O(N^2) datos, y por eso su AI crece con N. Esa es la única razón por la que una puede llegar a ser compute bound y la otra no. -->
<!-- Cuánto se gana optimizando: nada que valga la pena. La brecha entre el kernel ingenuo (0.25) y el mínimo (0.5) es de apenas 2 veces; en la multiplicación de matrices era de 2N/3, o sea 683 veces a N = 1024. -->
<!-- El número que conviene dejar escrito en la pizarra: por el roofline, rendimiento <= AI x ancho de banda = 0.5 x 320 = 160 GFLOP/s. Ese es el techo ABSOLUTO de y = Ax en la T4, un 2% del peak de cómputo, y no hay forma de programarlo mejor para superarlo. (Es la misma cuenta de la parte (d) de la pregunta de roofline del quiz.) -->
<!-- Sobre la columna GB/s, por si sale baja: threads consecutivos leen A[fila*n + k] con fila consecutivo, o sea direcciones separadas por 4N bytes. Ese patrón NO es coalescente y puede desperdiciar ancho de banda. Es un tema del capítulo 2 y no cambia ninguna de las conclusiones anteriores: las AI son aritmética pura y el techo de 160 GFLOP/s sigue en pie. Si acaso, es un segundo gancho hacia el capítulo 2: allá el problema de la matriz-matriz era el reuso, acá es el patrón de acceso. -->

<!-- --- -->
<!---->
<!-- ## **Ejercicio: la multiplicación de matrices** -->
<!---->
<!-- Descargar: [matmul_roofline.cu](../code/intro/ejercicios/matmul_roofline.cu) -->
<!---->
<!-- Multiplicación $C = A B$ de matrices $N \times N$, un *thread* por cada elemento de $C$: cada uno recorre una fila de $A$ y una columna de $B$. -->
<!---->
<!-- @include[cuda]{static/code/intro/ejercicios/matmul_roofline.cu:16-26} -->
<!---->
<!-- ```sh -->
<!-- !nvcc -arch=sm_75 matmul_roofline.cu -o matmul_roofline.x && !./matmul_roofline.x -->
<!-- ``` -->
<!---->
<!-- --- -->

<!-- ## **Ejercicio: la multiplicación de matrices** -->
<!---->
<!-- 1. Para **un elemento de $C$**: ¿cuántos FLOP hace el *kernel*, y cuántos bytes lee y escribe de la memoria? Calcular la $AI$. ¿Depende de $N$? -->
<!-- 2. Ahora no contar lo que hace el *kernel*, sino **el total obligatorio** para todo el producto: leer $A$ y $B$ una vez y escribir $C$ una vez. ¿Cuántos bytes son? Con los mismos $2N^3$ FLOP, ¿qué $AI$ da para $N = 1024$, y de qué lado del punto de inflexión queda? -->
<!-- 3. Correr y perfilar (`ncu --metrics dram__bytes.sum`). ¿Dónde cae el tráfico medido respecto de los dos conteos anteriores? ¿Y los GFLOP/s medidos respecto del *peak*? -->
<!-- <!-- 4. El mismo algoritmo da $AI = 0.25$ y $AI = 170$. ¿Qué hace la diferencia? ¿Donde se ubicaría en el diagrama roofline? --> -->
<!---->
<!-- <!-- RESPUESTAS. --> -->
<!-- <!-- 1. Atajo para verlo sin álgebra, y es el que conviene mostrar en la pizarra: mirar UNA iteración del lazo en k. Hace 1 multiplicación y 1 suma (2 FLOP) y lee dos floats (8 bytes). AI = 2/8 = 0.25. Como todas las iteraciones son iguales, el largo del lazo no cambia la razón: por eso la N se cancela. --> -->
<!-- <!-- Queda justo al lado de SAXPY (0.167). --> -->
<!-- <!-- Cómo contar las FLOP, si hay dudas: FLOP = (cuántos elementos tiene la salida) x (costo de un elemento). El ladrillo es siempre un producto punto de largo N, o sea N multiplicaciones y N sumas = 2N FLOP. Acá la salida es una matriz, N^2 elementos, luego 2N^3 FLOP en total. --> -->
<!-- <!-- Verificación cruzada que conviene mostrar: la columna j de C es A por la columna j de B, así que una multiplicación de matrices son N multiplicaciones matriz-vector: N x 2N^2 = 2N^3. Cierra. --> -->
<!-- <!-- Detalle de las sumas: sumar N productos son estrictamente N-1 sumas, así que el producto punto son 2N-1 FLOP y no 2N. Se usa 2N porque el código parte de suma = 0 y hace N sumas, porque la diferencia es 1 en 2N (0.05% a N = 1024), y sobre todo por consistencia con el techo: los 8.1 TFLOP/s de la T4 salen de 2560 cores x 2 x 1.59 GHz, y ese x2 está porque NVIDIA cuenta cada FMA (a*b+c, una sola instrucción) como 2 FLOP. Contando 2N, el numerador y el techo quedan en la misma vara. --> -->
<!-- <!-- 2. Cada matriz se lee o escribe una sola vez: 3 N^2 elementos = 12 N^2 bytes = 12.58 MB para N = 1024. Con 2N^3 = 2147 MFLOP eso da AI = N/6 = 170.7 FLOP/byte, casi 7 veces pasado el punto de inflexión: del lado compute bound. --> -->
<!-- <!-- Cuál resultado es exacto y cuál es un límite: N/6 es EXACTO, sin aproximación, porque el tráfico mínimo es exactamente 12N^2 y las FLOP exactamente 2N^3. El 0.25 de la pregunta 1, en cambio, es asintótico: el valor exacto es 2N/(8N+4), que a N = 1024 da 0.2499. Lo que se desprecia es el único byte escrito frente a los 8N leídos. --> -->
<!-- <!-- Lo que separa a las preguntas 1 y 2 NO es contar por elemento o contar la matriz entera: AI es una razón, así que da lo mismo (el conteo ingenuo para la matriz completa es 2N^3/(8N^3+4N^2), que también tiende a 0.25). Lo que cambia es CUÁLES bytes se cuentan: la 1 cuenta los que el kernel pide, releyendo cada fila y cada columna una y otra vez; la 2 cuenta los inevitables, tocando cada matriz una sola vez. --> -->
<!-- <!-- Precisión, por si la preguntan: no es que cada elemento se lea Y se escriba una vez. A se lee una vez, B se lee una vez y C se escribe una vez, o sea 3N^2 accesos. (Una versión estilo BLAS, C = alpha*A*B + beta*C, además lee C: serían 4N^2.) --> -->
<!-- <!-- El contraste con SAXPY es el punto: SAXPY mueve 12N bytes y su mínimo inevitable también es 12N. No tiene brecha, no hay nada que reusar, ya está en su techo. La multiplicación tiene una brecha de 2N/3. --> -->
<!-- <!-- El cuociente entre los dos tráficos es el reuso que estamos botando: (8N^3)/(12N^2) = 2N/3, o sea 683 veces a N = 1024. El kernel ingenuo mueve 683 veces más datos que el mínimo. --> -->
<!-- <!-- Vale la pena detenerse en la coincidencia: esos 12.58 MB son exactamente los mismos bytes que mueve SAXPY. Sobre los mismos bytes obligatorios, SAXPY hace 2.1 MFLOP y la multiplicación hace 2147 MFLOP: 1024 veces más aritmética. Eso es estar al otro lado del roofline. --> -->
<!-- <!-- 3. ATENCIÓN, acá se rompe la receta de los ejercicios anteriores: NO se puede predecir la duración dividiendo el tráfico ingenuo por el ancho de banda. Esos 8.59 GB a 256 GB/s darían 33.6 ms, y el kernel corre en unos pocos ms. --> -->
<!-- <!-- La razón es que las cachés ya están recuperando parte del reuso por su cuenta: la fila de A se reparte entre los threads del warp y buena parte de B queda en L2 entre un bloque y el siguiente. Por eso el tráfico medido cae ENTRE los dos modelos, mucho más cerca del mínimo que del conteo ingenuo. Ese "entre medio" es la respuesta buscada; el conteo ingenuo es una cota superior, no una predicción. --> -->
<!-- <!-- Esperable: del orden de 1 a 4 ms, o sea unos 500 a 2000 GFLOP/s, entre el 7% y el 27% del peak. Comparar con el menos del 1% de cualquier kernel elemento a elemento del ejercicio 1: aún mal escrita, la multiplicación de matrices usa el GPU un orden de magnitud mejor. --> -->
<!-- <!-- 4. La diferencia es el REUSO. Cada elemento de A hace falta para N elementos distintos de C, y el kernel ingenuo lo vuelve a leer desde la memoria global las N veces. El dato ya estuvo en el chip y lo botamos. --> -->
<!-- <!-- SAXPY está condenado a la izquierda del roofline: cada dato se usa una sola vez y no hay nada que reusar. La multiplicación de matrices no: tiene reuso de sobra y sólo hay que aprovecharlo. Guardar cada dato traído y usarlo varias veces antes de soltarlo es exactamente el tema del capítulo 2. --> -->
<!-- <!-- No prometer de más: esa AI ideal de 170 supone poder guardar TODO en el chip, y no se puede (12.58 MB de datos contra 4 MB de L2). Con tiles de T x T en memoria compartida lo alcanzable es AI = T/4: con T = 32 son 8 FLOP/byte, 32 veces mejor que 0.25 pero todavía a la izquierda del punto de inflexión (25). Cruzarlo pediría T = 100, o sea 78 KB de shared por bloque, y hay 64 KB. --> -->
<!-- <!-- O sea: el capítulo 2 cierra la mayor parte de la brecha de 683, no la cierra entera. Las librerías tipo GEMM llegan más lejos agregando reuso también en los registros (cada thread calcula un parche de varios elementos de C, no uno solo). --> -->


---

## Fin capítulo 1

Próximo capítulo: el uso de la memoria del GPU
