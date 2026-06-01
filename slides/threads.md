---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

# **Programación en GPUs**
## Control de los threads y optimización

---

## **Organización de los threads**

- **Grid**: compuesto de bloques de *threads*.
- **Block**: compuesto de *warps* de *threads* (máximo de $1024$ *threads* en total).
- **Warp**: compuesto de $32$ *threads*.

---

## **Sincronización de los threads**

- **Grid**: no hay sincronización de bloques.
- **Block**: se puede sincronizar con `__syncthreads()`.
- **Warp**: sincronización implícita.

---

## **Códigos**

Los códigos de esta clase están disponibles para descargar:

- [cuda_thread_block.cu](../code/threads/cuda_thread_block.cu), [matriz_mult.cu](../code/threads/matriz_mult.cu)
- Reducción: [reduccion_global.cu](../code/threads/reduccion_global.cu) … [reduccion_global8.cu](../code/threads/reduccion_global8.cu), [reduccion_compartida.cu](../code/threads/reduccion_compartida.cu)
- [grid_stride.cu](../code/threads/grid_stride.cu)
- [warp_shuffle_down.cu](../code/threads/warp_shuffle_down.cu), [warp_shuffle_up.cu](../code/threads/warp_shuffle_up.cu), [warp_shuffle_xor.cu](../code/threads/warp_shuffle_xor.cu)
- [gpu_suma_error.py](../code/threads/gpu_suma_error.py), [gpu_producto_punto_error.py](../code/threads/gpu_producto_punto_error.py)

---

## **Modelo de CUDA**

**SIMT**: *single instruction, multiple threads*.

Modelo híbrido entre **SIMD** (*single instruction, multiple data*) y **SMT** (*simultaneous multithreading*).

En programación paralela:
- SIMD = vectorización (AVX)
- SMT = OpenMP

---

## **SIMT vs. SIMD vs. SMT**

- **SIMD**: procesamiento de elementos de vectores cortos en paralelo.
- **SMT**: instrucciones de varios *threads* corren en paralelo.
- **SIMT**:
  - Instrucción simple, múltiples registros.
  - Instrucción simple, múltiples direcciones de memoria.
  - Instrucción simple, múltiples flujos de ejecución.

---

## **SIMT — múltiples registros**

Instrucción simple, múltiples registros:
- Escribimos la instrucción de un solo *thread* (*scalar spelling*).
- Cada *thread* tiene su propio registro.
- Hay repetición de variables entre registros (más redundante, pero más flexible).

---

## **SIMT — múltiples direcciones de memoria**

Instrucción simple, múltiples direcciones de memoria:
- Los *threads* pueden acceder a cualquier parte de la memoria.
- Para SIMD (vectorización) no es así, pero sí para SMT.
- De todas maneras, ¡es mejor mantener el acceso contiguo!

---

## **SIMT — múltiples flujos de ejecución**

Instrucción simple, múltiples flujos de ejecución:
- Es posible poner `if`/`else` en las instrucciones para los *threads* (*branching*).
- Implica divergencia en la ejecución de los *threads*.
- Generalmente no es una buena idea... se llama *warp divergence*.

---

## **SIMT vs. SIMD vs. SMT**

En términos de flexibilidad: **SIMD < SIMT < SMT**.

Costos de SIMT:
- Si la ocupación (*occupancy*) de los *threads* es baja, el rendimiento es bajo.
- El *warp divergence* afecta el rendimiento.

Antes de CUDA 9 solo se podía sincronizar *threads* dentro de un bloque con `__syncthreads()`. CUDA 9 añadió:
- *Warp primitives* (más control de los *threads* en un *warp*).
- *Cooperative groups*: más posibilidades de sincronización, hasta el nivel del *grid* entero (con restricciones).

---

## **Diseño del GPU**

*Throughput* alto, *latency* alto:
- Los CPUs optimizan *latency* (tiempo de demora).
- Los GPUs optimizan *throughput* (cantidad de datos procesados).

Cada *core* de un GPU es mucho más lento que un *core* de un CPU... pero hay miles de *cores*, así que un GPU puede usar miles de *threads* e intercambiar entre ellos (*context switching*) mucho más rápido que un CPU.

---

## **Threads, warps, blocks**

Ejemplo 1: [cuda_thread_block.cu](../code/threads/cuda_thread_block.cu).

- Operaciones asincrónicas (bloques, *warps*) imprimen a la pantalla en cualquier orden.
- Operaciones sincrónicas (*threads* en un *warp*) imprimen en orden.
- Los *threads* en un *warp* operan en *lock-step*: todos hacen lo mismo en el mismo momento.

---

# Occupancy

---

## **Occupancy**

$$\text{Occupancy} = \frac{\text{warps activos}}{\text{máx. warps activos}}$$

- Los recursos para los *threads* se asignan **por bloque**.
- Usar muchos recursos por *thread* puede limitar el *occupancy*.
- El *occupancy* puede estar limitado por:
  - Uso de registros.
  - Uso de memoria compartida.
  - Tamaño de los bloques.

---

## **Occupancy: registros**

Se puede obtener información sobre el uso de registros del compilador con `--resource-usage`.

Consideremos la arquitectura **Fermi**: $32$K registros por SM, $48$ *warps* activos por SM ($1536$ *threads*).

**Ejemplo A**: *kernel* que usa $21$ registros por *thread*.
- *Threads* activos $= 32\text{K}/21 = 1560$.
- $1560 > 1536$, *occupancy* $= 1$.

---

## **Occupancy: registros**

**Ejemplo B**: *kernel* que usa $64$ registros por *thread*.
- *Threads* activos $= 32\text{K}/64 = 512$.
- $512/1536 = 0.33$.

El uso de los registros se puede controlar con `--maxrregcount`.

---

## **Occupancy: memoria compartida**

Fermi tiene $16$ KB o $48$ KB de memoria compartida.

**Ejemplo A**: *kernel* que usa $48$ KB.
- Usa $32$ B de memoria compartida por *thread*.
- $48\text{KB}/32\text{B} = 1536$ *threads*, *occupancy* $= 1$.

---

## **Occupancy: memoria compartida**

**Ejemplo B**: *kernel* que usa $16$ KB.
- Usa $32$ B de memoria compartida por *thread*.
- $16\text{KB}/32\text{B} = 512$ *threads*, *occupancy* $= 0.33$.

---

## **Occupancy: tamaño del bloque**

- Cada SM (en Fermi) puede tener $8$ bloques activos.
- Usar bloques pequeños limitará el número total de *threads*.

---

## **Occupancy: tamaño del bloque**

| Tamaño bloque | Threads activos | Occupancy |
|:---:|:---:|:---:|
| 32  | 256  | 0.166 |
| 64  | 512  | 0.333 |
| 128 | 1024 | 0.666 |
| 192 | 1536 | 1 |
| 256 | 2048 (1536) | 1 |

---

## **¿Cuándo optimizar el occupancy?**

- Si el *kernel* está limitado por el *bandwidth* (*bandwidth bound*).
- Si el *bandwidth* logrado es mucho menor que el *peak*.

---

## **Occupancy: ejemplo**

Ejemplo 2: [matriz_mult.cu](../code/threads/matriz_mult.cu) (multiplicación de matrices).

- Compilamos con `--resource-usage`.
- Se puede medir el *occupancy* con el *profiler*:
  - `nvprof`: `achieved_occupancy`
  - `ncu`: `sm__warps_active.avg.pct_of_peak_sustained_active`

---

## **Occupancy: control**

Para controlar el *occupancy*:
- `__launch_bounds__(max_threads_por_bloque, <min_bloques_por_sm>)` en la definición del *kernel* (el segundo argumento es opcional).
- `--maxrregcount` en el compilador, para limitar el número de registros ocupados.

Si el compilador no puede satisfacer las restricciones, usará memoria fuera de los registros (*register spill*).

---

# Reducción: una aplicación de la optimización

---

## **Reducción paralela**

- Vimos el concepto de reducción en el curso de programación paralela.
- Significa obtener un solo valor de un conjunto de datos, en forma paralela.
- Un ejemplo sería la suma total de todos los elementos en un *array*.

---

## **Reducción paralela**

![w:560px](images/threads/parallel_reduction.jpeg)

<p class="credit">Fuente: www.eximia.co</p>

---

## **Reducción paralela**

![w:560px](images/threads/par_red.png)

<p class="credit">Fuente: sodocumentation.net</p>

---

## **Reducción paralela: memoria global**

Ejemplo 3: [reduccion_global.cu](../code/threads/reduccion_global.cu).

- *Kernel* invocado dentro de un ciclo.
- El *stride* cambia en cada iteración por un factor de $2$.

---

## **Reducción paralela: memoria global**

![w:560px](images/threads/reduction_global.png)

`stride = 8` no puede pasar porque el ciclo va hasta `stride < N`.

---

## **Reducción paralela: memoria global**

```cuda
__global__ void reduccion_memoria_global(float *data, int stride, int N)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_x + stride < N) {
    data_out[idx_x] += data_in[idx_x + stride];
  }
}
...
int main(){
  ...
  for (int stride = 1; stride < N; stride *= 2) {
    global_reduction_kernel<<<n_bloques, n_threads>>>(d_array, stride, N);
  }
  ...
}
```

---

## **Reducción paralela: memoria global**

Hay un problema con el *kernel*... ¿cuál es?

... todos los *threads* calculan algo, ¡pero los resultados no son necesarios!

---

## **Reducción paralela: menos threads trabajando**

Ejemplo 4: [reduccion_global2.cu](../code/threads/reduccion_global2.cu).

![w:560px](images/threads/reduction_global_strided.png)

---

## **Divergencia de warps**

Este método también sufre de un problema... En CUDA (gracias a SIMT) se puede tener algo como:

```cuda
if (threadIdx.x % 2 == 0) {
  // rama 1
} else {
  // rama 2
}
```

Los *threads* de índice par ejecutan la **rama 1** y los de índice impar la **rama 2**.

---

## **Divergencia de warps**

- Todos los *threads* dentro de un *warp* ejecutan la misma instrucción en el mismo momento.
- Entonces, ¿cómo podemos tener divergencia de los *threads* dentro de un *warp*?
- En el ejemplo (y en el código de la reducción) los *threads* de índice par ejecutan sus instrucciones, mientras los otros **esperan**.
- La divergencia de *warp* (*warp divergence*) implica menos eficiencia de un *kernel*.

---

## **Divergencia de warps**

![w:560px](images/threads/warp_divergence.png)

---

## **Divergencia de warps**

En los *profilers* se puede medir el nivel de *warp divergence*:
- `nvprof`: `branch_efficiency`
- `ncu`: `smsp__sass_average_branch_targets_threads_uniform.pct`

---

## **Reducción paralela: sin warp divergence**

Ejemplo 5: [reduccion_global3.cu](../code/threads/reduccion_global3.cu).

![w:560px](images/threads/reduction_global_noWD.png)

Puede ser problemático usar el índice global de los *threads* para un *array* muy grande...

---

## **Reducción paralela: usando bloques**

Ejemplo 6: [reduccion_global4.cu](../code/threads/reduccion_global4.cu).

![w:520px](images/threads/block_reduction.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Reducción paralela: acceso contiguo**

Ejemplo 7: [reduccion_global5.cu](../code/threads/reduccion_global5.cu).

![w:520px](images/threads/interleaved.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Optimización: loop unrolling**

- Ahora aplicaremos una técnica de optimización que también aplica a la programación secuencial: *loop unrolling*.
- Primero, veremos un ejemplo en programación secuencial.

---

## **Optimización: loop unrolling**

```c
for (int i = 0; i < N; i++){
  a[i] = b[i] + c[i];
}
```

```c
for (int i = 0; i < N; i+=2){
  a[i]   = b[i]   + c[i];
  a[i+1] = b[i+1] + c[i+1];
}
```

---

## **Optimización: loop unrolling**

- El segundo ciclo tiene menos iteraciones.
- Por lo tanto hay menos *overhead* de verificar si el ciclo puede terminar.
- También hay oportunidades de optimizar el acceso a la memoria (uso de *cache*, etc.).
- Para *loops* así, los compiladores modernos típicamente aplican *loop unrolling* automáticamente.

---

## **Reducción paralela: loop unrolling**

¿Cómo aplicamos *loop unrolling* en el *kernel* de reducción paralela?

Aplicamos una suma entre bloques **antes** de comenzar con el ciclo de la reducción dentro de los bloques.

---

## **Reducción paralela: loop unrolling**

Ejemplo 8: [reduccion_global6.cu](../code/threads/reduccion_global6.cu).

![w:560px](images/threads/reduction_global_unrolled.png)

---

## **Optimización: loop unrolling**

```c
for (int i = 0; i < N; i+=2){
  a[i]   = b[i]   + c[i];
  a[i+1] = b[i+1] + c[i+1];
}
```

Esto corresponde a *loop unrolling* con un factor de $2$.

---

## **Optimización: loop unrolling**

```c
for (int i = 0; i < N; i+=4){
  a[i]   = b[i]   + c[i];
  a[i+1] = b[i+1] + c[i+1];
  a[i+2] = b[i+2] + c[i+2];
  a[i+3] = b[i+3] + c[i+3];
}
```

Esto corresponde a *loop unrolling* con un factor de $4$.

---

## **Reducción paralela: loop unrolling**

```cuda
if (idx + blockDim.x < N) data[idx] += data[idx + blockDim.x];
```

Con esa línea sumamos valores en $2$ bloques (*loop unrolling* $\times 2$).

---

## **Reducción paralela: loop unrolling**

```cuda
if (idx + 3 * blockDim.x < N)
{
  float a1 = data[idx];
  float a2 = data[idx + blockDim.x];
  float a3 = data[idx + 2 * blockDim.x];
  float a4 = data[idx + 3 * blockDim.x];
  data[idx] = a1 + a2 + a3 + a4;
}
```

Aquí usamos valores en $4$ bloques (*loop unrolling* $\times 4$).

---

## **Reducción paralela: loop unrolling**

```cuda
if (idx + 7 * blockDim.x < N)
{
  float a1 = data[idx];
  float a2 = data[idx + blockDim.x];
  float a3 = data[idx + 2 * blockDim.x];
  float a4 = data[idx + 3 * blockDim.x];
  float a5 = data[idx + 4 * blockDim.x];
  float a6 = data[idx + 5 * blockDim.x];
  float a7 = data[idx + 6 * blockDim.x];
  float a8 = data[idx + 7 * blockDim.x];
  data[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
}
```

Aquí usamos valores en $8$ bloques (*loop unrolling* $\times 8$).

---

## **Reducción paralela: loop unrolling**

Ejemplo 9: [reduccion_global7.cu](../code/threads/reduccion_global7.cu).

Este código incluye $3$ *kernels* con distintos factores de *unrolling*.

---

## **Reducción paralela: warp unrolling**

- Cuando llegamos a $< 32$ elementos en un bloque, el trabajo cabe en un *warp*.
- Como los *threads* en un *warp* están sincronizados implícitamente, no necesitamos `__syncthreads()`.
- Además, tendremos *threads* que no trabajan mientras el número de sumas disminuye: menos eficiente.
- Podemos hacer *unroll* dentro del último *warp* para optimizar más.

---

## **Reducción paralela: warp unrolling**

```cuda
if (threadIdx.x < 32)
{
  volatile float *vmem = idata;
  vmem[threadIdx.x] += vmem[threadIdx.x + 32];
  vmem[threadIdx.x] += vmem[threadIdx.x + 16];
  vmem[threadIdx.x] += vmem[threadIdx.x + 8];
  vmem[threadIdx.x] += vmem[threadIdx.x + 4];
  vmem[threadIdx.x] += vmem[threadIdx.x + 2];
  vmem[threadIdx.x] += vmem[threadIdx.x + 1];
}
```

Ejemplo 10: [reduccion_global8.cu](../code/threads/reduccion_global8.cu). Todos los *threads* del *warp* calculan, pero solo necesitamos el resultado del primero.

---

## **¿Qué es `volatile`?**

- `volatile` es un ejemplo de un *qualifier* (calificador). Los calificadores dan indicaciones al compilador.
- `volatile` asegura que el compilador no intente optimizar la variable, es decir, que no la mueva temporalmente a un *cache* o registro durante la operación del programa.
- Esto asegura que no haya *race conditions* mientras los *threads* actualizan los valores de `vmem`.

---

## **Reducción paralela: memoria compartida**

- Hasta ahora hemos usado memoria global en todos los *kernels*.
- Podemos optimizar el acceso a la memoria usando **memoria compartida**.

Ejemplo 11: [reduccion_compartida.cu](../code/threads/reduccion_compartida.cu).

---

## **Reducción paralela: errores**

- El resultado del GPU no es exactamente igual al del CPU.
- Esto se debe a que siempre hay errores de redondeo, y en el cálculo secuencial estos errores se acumulan.
- Por la paralelización se espera que los errores sean **menores** en el GPU.

Códigos de Python que muestran la idea: [gpu_suma_error.py](../code/threads/gpu_suma_error.py) y [gpu_producto_punto_error.py](../code/threads/gpu_producto_punto_error.py). Ambos usan **PyCUDA**, un módulo de Python que veremos más tarde.

---

# Grid-stride loops

---

## **Grid-stride loops**

Se pueden usar *grid-stride loops* (ciclos con saltos del tamaño de la grilla) para aplicar un *kernel* a una parte de los datos de forma secuencial.

Como ejemplo, la operación **SAXPY** (*single-precision A·X plus Y*): $Y = A \cdot X + Y$

```c
void saxpy(int N, float a, float* x, float* y){
  for (int i = 0; i < N; ++i)
    y[i] = a * x[i] + y[i];
}
```

---

## **Grid-stride loops**

Si hay $N$ elementos en los vectores $X$ y $Y$, típicamente en el GPU usamos $N$ *threads*:

```cuda
__global__ void saxpy(int N, float a, float* x, float* y){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    y[i] = a * x[i] + y[i];
}
```

---

## **Grid-stride loops**

Si hay $N = 2^{20} = 4096 \times 256$ elementos, podemos invocar el *kernel* con:

```cuda
saxpy<<<4096, 256>>>(1<<20, 2.0, x, y);
```

Aseguramos que haya suficientes *threads* para los elementos de los vectores (usamos múltiplos de $32$, el número de *threads* en un *warp*).

---

## **Grid-stride loops**

Podemos usar menos *threads* y un *grid-stride loop*:

```cuda
__global__ void saxpy(int N, float a, float *x, float *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N; i += blockDim.x * gridDim.x)
  {
    y[i] = a * x[i] + y[i];
  }
}
```

---

## **Grid-stride loops**

- Cada *thread* tiene su propio ciclo `for`.
- El *stride* del ciclo es `blockDim.x * gridDim.x`, igual al número total de *threads* en el *grid*.
- Por ejemplo, con $1280$ *threads*, el *thread* $0$ calcula los elementos $0$, $1280$, $2560$, etc.

Ejemplo 12: [grid_stride.cu](../code/threads/grid_stride.cu).

---

## **Grid-stride loops: ¿por qué?**

- **Escalabilidad**: el *kernel* aplica a cualquier número de elementos en los vectores.
- **Reutilización**: hay reutilización de *threads*, y eso mejora el *occupancy*.

---

# Warp primitives

---

## **Warp primitives**

- Las *warp primitives* (funciones primitivas de *warp*) son funciones básicas para trabajar directamente con los *threads* dentro de un *warp*.
- Un poco de jerga: un *thread* dentro de un *warp* se conoce como un *lane*.
- Hay muchas funciones disponibles: *CUDA Programming Guide*.
- Como ejemplo, veremos una implementación de reducción dentro de un *warp* con *primitives*.

---

## **Warp primitives**

```cuda
#define FULL_MASK 0xffffffff  // 32-bits, todos igual a 1
for (int stride = 16; stride > 0; stride >>= 1)
  val += __shfl_down_sync(FULL_MASK, val, stride);
```

![w:520px](images/threads/shfl_down.png)

<p class="credit">Fuente: NVIDIA Developer Blog</p>

---

## **Warp primitives**

Ejemplo 13: [warp_shuffle_down.cu](../code/threads/warp_shuffle_down.cu).

- Es un ejemplo bastante artificial (solo funciona para un vector del tamaño de un *warp*), pero muestra el uso de un *warp primitive*.
- El *mask* corresponde a los *threads* que deben estar convergentes en el momento de llegar a la función primitiva.

---

## **Warp primitives**

- Intercambio de datos **entre registros**.
- Más eficiente que el uso de memoria compartida, que requiere un *load*, un *store* y otro registro para la dirección en memoria.
- Otra opción para este nivel de control es el uso de *cooperative groups* (fuera del ámbito de este curso...).

---

## **Warp primitives**

- Ejemplo 14: [warp_shuffle_up.cu](../code/threads/warp_shuffle_up.cu).
- Ejemplo 15: [warp_shuffle_xor.cu](../code/threads/warp_shuffle_xor.cu).

---

# ¡Gracias!

## Próxima clase: invocación de los kernels
