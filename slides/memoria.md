---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

# **Programación en GPUs**
## El uso de la memoria del GPU

---

## **Memoria y optimización**

- La mayoría de los programas gastan mucho tiempo moviendo los datos desde la memoria del *device* a su unidad de procesamiento
  (GPU).

Optimizar los accesos a la memoria $\implies$ optimizar el rendimiento

---

## **Jerarquía de memoria**

- Tanto en el *host* como en el *device*, existen distintos tipos de memorias.
- Por regla general: las memorias más rápidas son más pequeñas y más cercanas al *core*; las más lentas, más grandes y más alejadas.
- ¿Cuánto más lentas? Latencia en la T4: registros $\sim 4$ ciclos · L1 / compartida $\sim 30$ · L2 $\sim 200$ · DRAM $\sim 300$–$600$: **dos órdenes de magnitud**.
- En ancho de banda: DRAM $\approx 300$ GB/s contra $\approx 16$ GB/s del PCIe hacia el *host* ($\sim 20\times$).

<p class="credit">Latencias: Jia et al., <em>Dissecting the NVidia Turing T4 GPU via Microbenchmarking</em> (2019)</p>

---

## **Jerarquía de memoria (host)**

![w:560px](images/memoria/figure_4_1.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

<!-- Los registros y *caches* no son programables. -->

---

## **Jerarquía de memoria (device)**

![w:460px](images/memoria/figure_4_2.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

<!-- Se puede programar cualquier espacio de memoria que no sea *cache*. -->

<!-- --- -->
<!---->
<!-- ## **Códigos** -->
<!---->
<!-- Los códigos de esta clase están disponibles para descargar: -->
<!---->
<!-- - [variableGlobal.cu](../code/memoria/variableGlobal.cu), [variableGlobalDin.cu](../code/memoria/variableGlobalDin.cu) -->
<!-- - [copiarFila.cu](../code/memoria/copiarFila.cu), [copiarColumna.cu](../code/memoria/copiarColumna.cu) -->
<!-- - [transpuesta.cu](../code/memoria/transpuesta.cu), [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu) -->
<!-- - [aos.cu](../code/memoria/aos.cu), [soa.cu](../code/memoria/soa.cu), [alineamiento_datos.c](../code/memoria/alineamiento_datos.c) -->
<!-- - [memoria_constante.cu](../code/memoria/memoria_constante.cu), [memoriaPinned.cu](../code/memoria/memoriaPinned.cu), [memoria_unificada.cu](../code/memoria/memoria_unificada.cu) -->

---

# Memoria global

---

## **Memoria global**

- La memoria principal del device.
  - *Latency*: alto.
  - *Bandwidth*: bajo.
- Se puede asignar memoria global de forma **dinámica** con `cudaMalloc`.
- Se puede asignar memoria global de forma **estática** en el *device* con `__device__`.

---

## **Memoria global: declaración estática**

Ejemplo: [variableGlobal.cu](../code/memoria/variableGlobal.cu).
Declaramos una variable global en la memoria global del *device*.

```cuda
#define N 10
__device__ int devVar[N];

int main(){
  ...
  int hostVar[N];
  ...
  cudaMemcpyToSymbol(devVar, &hostVar, N*sizeof(int));
  ...
}
```

---

## **Memoria global: declaración dinámica**

Ejemplo: [variableGlobalDin.cu](../code/memoria/variableGlobalDin.cu).
El mismo programa, pero con declaración dinámica (ya no tiene *global scope*).

```cuda
#define N 10
int main(){
  ...
  int* hostVar = (int *) malloc(N*sizeof(int));
  int* devVar;
  cudaMalloc((int**)&devVar, N*sizeof(int));
  ...
  cudaMemcpy(devVar, hostVar, N*sizeof(int), cudaMemcpyHostToDevice);
  ...
}
```

---

## **Memoria global**

- Dado que la latencia de la memoria global es alta, la idea central es **optimizar su uso** durante la ejecución de un programa para mejorar su rendimiento.
- En general queremos:
  - Minimizar los accesos.
  - Cuando necesitamos acceder y cargar datos, hacerlo en bloque.

<!-- - Antes de ver las técnicas de optimización, recordemos cómo los *threads* acceden a la memoria. -->

Para entender la interacción entre GPU y la memoria del *device*, es necesario hablar en más detalle de los *warps*.

---

## **Los Warps**

![w:520px](images/warps_thread_blocks.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Los Warps**


![w:820px](images/warps_logical_hardware_view.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Los Warps**

- En CUDA, los *threads* no ejecutan las instrucciones de un código de forma independiente.
- Las instrucciones se despachan a *warps*.
  - Cada ciclo de reloj del GPU, un *warp* ejecuta una misma instrucción, en general sobre distintos datos de la memoria.
- En un programa eficiente, los *threads* de un *warp* acceden a la memoria **en bloque**.

<!-- --- -->

<!--
Diapositiva "hook" en pausa (reactivar quitando este comentario):
_class: hook
## **Los Warps**
<p class="destacado">¿Cómo se transfieren los datos entre procesador y memoria?</p>
-->

---

## **Transacciones de memoria**

- Los accesos de memoria también son hechos en términos de *warps*.
- Todos los accesos a la memoria global (DRAM) pasan por cache L2 (algunos también por L1).
- Las transacciones de memoria (lectura/escritura) están cuantizadas.
  - DRAM: 128 bytes (*segmento de memoria*).
  - Cache L1: 128 bytes (*línea de cache*).
  - Cache L2: 32 bytes (*línea de cache*).


---

## **Acceso a la memoria**

![w:1020px](images/memoria/figure_4_6.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria global: acceso eficiente**

La mejor forma (**patrón**) de acceder a la memoria global es con acceso **alineado** y **contiguo**.

- Alineado: primera dirección de memoria es múltiplo de 32 o 128 bytes.
- Contiguo: todos los *threads* del *warp* acceden a un bloque contiguo.

![w:1000px](images/memoria/figure_4_7.png)
<!-- ![w:1020px](images/memoria/aligned_coalesced.png) -->

<p class="credit">Alineado y contiguo — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria global: acceso ineficiente**

![w:1000px](images/memoria/figure_4_12.png)

<p class="credit">No alineado ni contiguo — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria global: acceso ineficiente (extremo)**

![w:1000px](images/memoria/non_coalesced.png)

<p class="credit">No alineado ni contiguo — Fuente: <em>Professional CUDA C Programming</em></p>


El acceso alineado no es tan importante comparado con el **acceso contiguo**.

---

## **Memoria global: acceso eficiente a matrices**

![w:320px](images/memoria/figure_4_23.png)

- Supongamos que almacenamos la matriz en forma de un arreglo 1D.

![w:420px](images/memoria/figure_4_24.png)

- Notar que los valores de cada fila son contiguos.

---

## **Memoria global: acceso eficiente a matrices**

![w:820px](images/memoria/row_column.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria global: acceso eficiente**

Ejemplo: [copiarfila.cu](../code/memoria/copiarfila.cu) y [copiarcolumna.cu](../code/memoria/copiarcolumna.cu).

- Matrices de $2048\times 2048$ elementos.
- Bloques 2D: $16\times 16$ threads.

Obtener las siguientes métricas con `ncu` (usando el flag `--metrics A,B`):
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct`

<!-- - Eficiencia load/store para `copiarFila` de $100\%$. -->
<!-- - Para `copiarColumna` la eficiencia de load es $25\%$, y de store es $12.5\%$. -->

<!-- --- -->
<!---->
<!-- ## **Memoria global: acceso eficiente** -->
<!---->
<!-- - Todo acceso a la memoria global pasa por L2, en líneas de $32$ bytes. -->
<!-- - Los **loads** además pueden quedar en L1. -->
<!--   - Una línea de $128$ bytes que un *warp* trajo puede servir cargas posteriores. -->
<!-- - Los **stores** no aprovechan L1 (*write-through*). -->
<!--   - Se resuelven en L2, sin reutilización posible. -->
<!---->
<!-- **Conclusión importante:** el uso de la memoria global es mucho más eficiente con **acceso contiguo**. -->


---

## **Memoria global: acceder por filas vs columnas**

Un *warp* pide $32 \times 4 = 128$ bytes útiles de `float`.

- **Por fila (contiguo)**: caben en $4$ segmentos de $32$ bytes. 
  - Eficiencia $128 / 128 = 100\%$.
- **Por columnas**: cada *thread* cae en un segmento distinto. Hasta $32$ segmentos, $32 \times 32 = 1024$ bytes movidos por $128$ útiles: $12.5\%$.
  - Con bloques `16x16`, las dos filas del *warp* son vecinas y comparten segmento: $16$ segmentos, $25\%$.

<!-- El ancho de banda **efectivo** cae en ese mismo factor: la DRAM trabaja igual, pero la mayoría de los bytes que mueve no se usan. -->

**Conclusión importante:** el uso de la memoria global es mucho más eficiente con **acceso contiguo**.

---

## **Transpuesta de una matriz**

![w:820px](images/memoria/row_column.png)

<p class="credit">Cargar por fila, guardar por columna — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Transpuesta de una matriz**

![w:820px](images/memoria/column_row.png)

<p class="credit">Cargar por columna, guardar por fila — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Transpuesta de una matriz**

Ejemplo: [transpuesta.cu](../code/memoria/transpuesta.cu).

La versión que carga por columnas es más rápida... ¿por qué?

- Las cargas por columna pasan por L1: la línea que trae un *thread* la reutilizan sus vecinos.
- Los *stores* no aprovechan L1, así que conviene que el acceso **contiguo** sea el de **guardar**.

---

# **Ejercicio: ¿por qué gana cargar por columnas?**

Descargar: [transpuesta.cu](../code/memoria/transpuesta.cu)

1. Ejecutar y comparar el *bandwidth* efectivo de `transpuestaCargarFilas` y `transpuestaCargarColumnas`.
2. ¿Cuál de los dos es más rápido?
3. Explicar el resultado: ¿qué operación alcanza a aprovechar el *cache* y cuál no?

---

# AoS vs. SoA

---

## **Estructuras de datos**

Un `struct` agrupa campos bajo un solo nombre; se acceden con `.`:

```cuda
struct Particula { float x; float y; };
Particula p;  p.x = 1.0f;  p.y = 2.0f;
```

Para $N$ partículas hay dos formas de organizar los mismos datos:

```cuda
Particula particulas[N];                        // AoS: particulas[i].x
struct Particulas { float x[N]; float y[N]; };  // SoA: particulas.x[i]
```

**AoS** (arreglo de estructuras) intercala `x y x y ...`; **SoA** (estructura de arreglos) separa `x x ... y y ...`. Para el *warp*, eso decide si el acceso es contiguo.

---

## **Opciones para estructuras de datos**

![w:420px](images/memoria/figure_4_22.png)

- **SoA**: cada *thread* lee **un campo** de muchos elementos → el *warp* accede a datos contiguos.
- **AoS**: cada *thread* usa **todos los campos** de su elemento; funciona bien si el *struct* está alineado ($8$ o $16$ bytes, como `float4`).

Ejemplo: [aos.cu](../code/memoria/aos.cu) y [soa.cu](../code/memoria/soa.cu).

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Alineamiento de estructuras**

La organización de los elementos en una estructura tiene consecuencias para el uso de la memoria: los mismos campos, en otro orden, ocupan otro tamaño.

Ejemplo: [alineamiento_datos.c](../code/memoria/alineamiento_datos.c).

- En CUDA los tipos vectoriales (`float2`, `float4`) ya vienen alineados a $8$ y $16$ bytes.
- Por eso el capítulo de aplicaciones usa `float4` para las posiciones en el código de n-cuerpos, y `float3` sólo para variables locales.

---

# Memoria compartida

---

## **Memoria compartida**

![w:460px](images/memoria/figure_4_2.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria compartida**

- Variables declaradas en el *kernel* con `__shared__` se guardan en memoria compartida.
- Esta memoria está *on-chip*: *bandwidth* alto, *latency* bajo.
- Cada SM tiene una cantidad limitada de memoria compartida, dividida entre los bloques de *threads*. Si usamos demasiada, el número de *warps* activos se reduce.
- Permite **comunicación entre los *threads*** (dentro de un bloque).

---

## **Memoria compartida — declaración estática**

```cuda
__shared__ float tile[ny][nx];
```

- Declarada dentro de un *kernel*: *scope* local; declarada fuera de cualquier *kernel*: *scope* global.
- Como la memoria compartida está asociada a un bloque de *threads*, típicamente `ny`, `nx` son iguales a las dimensiones de un bloque.

---

## **Memoria compartida — declaración dinámica**

```cuda
extern __shared__ int tile[];
```

- Tiene que ser declarada dentro de un *kernel*.
- El tamaño del *array* se define al invocar el *kernel*, con el tercer argumento de la configuración:

```cuda
kernel<<<grid, block, N * sizeof(int)>>>(...);
```

- Para declaración dinámica, solo se pueden declarar *arrays* unidimensionales.

---

## **Transpuesta: memoria compartida**

Volvemos al ejemplo de la transpuesta de una matriz, pero ahora usando memoria compartida.

![w:920px](images/memoria/figure_5_15.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Transpuesta: memoria compartida**

Ejemplo: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu). Hay cuatro *kernels*:

- `transpuestaGlobal`: la transpuesta con memoria global.
- `transpuestaComp`: memoria compartida **estática**.
- `transpuestaCompDin`: memoria compartida **dinámica**.
- `transpuestaCompPad`: memoria compartida estática con ***padding*** (volveremos a este).

---

## **Transpuesta: memoria compartida**

Consideramos un ejemplo: matriz de $4 \times 4$ elementos, con bloques de $2 \times 2$ (memoria compartida del mismo tamaño).

`blockDim.x`, `blockDim.y` son iguales a $2$; hay $2$ bloques en cada dimensión.

---

## **Transpuesta: memoria compartida**

![w:340px](images/memoria/transpose_fig1.png)

Índices globales de los *threads*:

```cuda
ix = blockDim.x * blockIdx.x + threadIdx.x;
iy = blockDim.y * blockIdx.y + threadIdx.y;
```

---

## **Transpuesta: memoria compartida**

![w:340px](images/memoria/transpose_fig2.png)

Índice lineal de los *threads*:

```cuda
ti = iy * N + ix;
```

---

## **Transpuesta: memoria compartida**

![w:340px](images/memoria/transpose_fig3.png)

Índices globales después de la "transpuesta de bloques":

```cuda
ixt = blockDim.y * blockIdx.y + threadIdx.x;
iyt = blockDim.x * blockIdx.x + threadIdx.y;
```

---

## **Transpuesta: memoria compartida**

![w:340px](images/memoria/transpose_fig4.png)

Índice lineal después de la "transpuesta de bloques":

```cuda
to = iyt * N + ixt;
```

---

## **Transpuesta: memoria compartida**

![w:320px](images/memoria/transpose_fig5.png)

Elementos guardados después de cargar de la memoria compartida:

```cuda
tile[threadIdx.y][threadIdx.x] = entrada[ti];
__syncthreads();
salida[to] = tile[threadIdx.x][threadIdx.y];
```

---

## **Acceso a la memoria compartida**

![w:1020px](images/memoria/figure_5_2.png)

<p class="credit">Acceso ideal — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Acceso a la memoria compartida**

![w:1020px](images/memoria/figure_5_3.png)

<p class="credit">Acceso desordenado, pero no problemático — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Acceso a la memoria compartida**

![w:1020px](images/memoria/figure_5_4.png)

<p class="credit">Potencialmente problemático... — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Organización de la memoria compartida (bancos)**

![w:1020px](images/memoria/figure_5_5.png)

<p class="credit">Bancos de ancho 4-bytes — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Organización de la memoria compartida (bancos)**

![w:1020px](images/memoria/figure_5_6.png)

<p class="credit">Bancos de ancho 8-bytes — Fuente: <em>Professional CUDA C Programming</em></p>

---

# Ejercicios

---

## **Ejercicio 3: ¿ayuda la memoria compartida?**

Descargar: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu)

El programa reporta el tiempo de cada *kernel*. Por ahora nos interesan tres:

- `transpuestaGlobal` · `transpuestaComp` · `transpuestaCompDin`

1. Ordenarlos de más lento a más rápido. ¿Cuánto se gana con memoria compartida?
2. ¿La declaración dinámica cuesta más que la estática?
3. ¿Qué pasaría si borráramos el `__syncthreads()`?

---

## **Ejercicio 3: los índices en papel**

Repetir el desarrollo de las diapositivas anteriores para una matriz de $8 \times 8$ con bloques de $4 \times 4$.

Para el *thread* `threadIdx = (1,2)` del bloque `blockIdx = (1,0)`, calcular:

```cuda
ix, iy      // índices globales
ti          // índice lineal de entrada
ixt, iyt    // índices tras la "transpuesta de bloques"
to          // índice lineal de salida
```

---

# Conflictos de bancos

---

## **Conflictos de bancos**

![w:1020px](images/memoria/figure_5_7.png)

<p class="credit">Todo bien acá — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Conflictos de bancos**

![w:1020px](images/memoria/figure_5_8.png)

<p class="credit">Todo bien acá también, gracias al ancho de 8-bytes — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Conflictos de bancos**

![w:1020px](images/memoria/figure_5_9.png)

<p class="credit">¡Conflicto! — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Conflictos de bancos**

![w:1020px](images/memoria/figure_5_10.png)

<p class="credit">¡Conflicto! — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Solución: *padding***

![w:920px](images/memoria/figure_5_11.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Transpuesta: conflictos de bancos**

```cuda
__shared__ float tile[BDIM][BDIM];
...
tile[threadIdx.y][threadIdx.x] = entrada[ti];
__syncthreads();
salida[to] = tile[threadIdx.x][threadIdx.y];
```

El acceso por columna corresponde a acceso al **mismo banco**.

---

## **Transpuesta: conflictos de bancos**

```cuda
__shared__ float tile[BDIM][BDIM+1];
...
tile[threadIdx.y][threadIdx.x] = entrada[ti];
__syncthreads();
salida[to] = tile[threadIdx.x][threadIdx.y];
```

Ahora los elementos de una columna van a **bancos distintos**.

---

# Memoria constante

---

## **Memoria constante**

- Reside en la memoria del *device*; cada SM tiene un *cache* asignado a la memoria constante.
- Se declara con `__constant__`. Debe tener *global scope*, fuera de cualquier *kernel*. Hay $64$ KB disponibles.
- Útil para constantes matemáticas aplicadas por todos los *threads*.
- Los *kernels* solo pueden **leer** de la memoria constante, así que hay que inicializarla desde el *host*:

```cuda
cudaError_t cudaMemcpyToSymbol(const void* simbolo, const void* src, size_t count);
```

---

## **Memoria constante**

Ejemplo: [memoria_constante.cu](../code/memoria/memoria_constante.cu).

---

# Memoria unificada

---

## **Transferencias de memoria**

![w:560px](images/memoria/figure_4_3.png)

<p class="credit">Ejemplo para Fermi C2050 GPU — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria *pinned***

- La memoria en el *host* es, por defecto, *paginable*.
- Está organizada en páginas que el sistema operativo puede mover a la memoria virtual (en el disco duro).
- Cuando el sistema requiere datos que están en el disco, ocurre un *page fault* y los datos se copian del disco al RAM. El GPU no controla el movimiento de las páginas.
- Transferir datos del *host* al *device* implica asignar memoria *page-locked* o *pinned* en el *host*: los datos se transfieren de *pageable* a *pinned* y después al *device*.

---

## **Memoria *pinned***

![w:760px](images/memoria/figure_4_4.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Asignación de memoria *pinned***

```cuda
cudaError_t cudaMallocHost(void **devPtr, size_t count);
cudaError_t cudaFreeHost(void *ptr);
```

El uso de demasiada memoria *pinned* puede afectar el rendimiento del sistema entero, ya que reduce la cantidad de memoria *pageable* disponible.

Ejemplo: [memoriaPinned.cu](../code/memoria/memoriaPinned.cu).

---

## **Memoria unificada**

- Desde CUDA 6.0, *Unified Memory* permite acceder a la memoria usando un **solo espacio de direcciones** para el GPU y el CPU. UM se encarga de la transferencia de datos automáticamente.
- Basada en *Unified Virtual Addressing* (CUDA 4.0), que unificó el espacio de direcciones en memoria.
- Declaración estática (a veces llamada *managed*): `__device__ __managed__ int y;`

---

## **Memoria unificada**
- Asignación dinámica:

```cuda
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=0);
```

El puntero `devPtr` es válido tanto en el *device* como en el *host*.

---

## **Memoria unificada**

Ejemplo: [memoria_unificada.cu](../code/memoria/memoria_unificada.cu).

---

# Ejercicios

---

## **Ejercicio 4: conflictos de bancos y *padding***

Descargar: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu)

En el ejercicio 3 ya comparamos los tres primeros *kernels*. Falta el cuarto: `transpuestaCompPad`.

1. `transpuestaComp` y `transpuestaCompPad` difieren en **un carácter**: `tile[BDIM][BDIM]` contra `tile[BDIM][BDIM+1]`. ¿Cuánto cambia el tiempo?
2. ¿Dónde queda `transpuestaCompPad` en el ranking del ejercicio 3?

---

## **Ejercicio 4: métricas**

En `nvprof`:
- `shared_load_transactions_per_request`
- `shared_store_transactions_per_request`

En `ncu`:
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`

3. Medir los conflictos de bancos en `transpuestaComp` y en `transpuestaCompPad`. ¿El número explica la diferencia de tiempo?

---

## **Ejercicio 5: ¿sirve la memoria constante?**

Descargar: [memoria_constante.cu](../code/memoria/memoria_constante.cu)

El programa ejecuta dos *kernels* que hacen exactamente el mismo cálculo:

- `kernel_device`: lee los $360$ ángulos desde la **memoria global**.
- `kernel_constante`: los lee desde la **memoria constante**.

1. El programa no imprime nada: hay que medirlo con el profiler.
2. Comparar la duración de ambos *kernels*.

```bash
nvcc -arch=sm_75 memoria_constante.cu -o memoria_constante.x
nvprof ./memoria_constante.x
```

---

## **Ejercicio 5: ¿por qué?**

3. Todos los *threads* leen **el mismo** ángulo en cada iteración del ciclo. ¿Por qué esto favorece a la memoria constante?
4. ¿Qué pasaría si cada *thread* leyera un ángulo **distinto**?

El programa invoca un tercer *kernel* (`kernel_inicial`) antes de medir. ¿Para qué sirve?

---

## **Ejercicio 6: *pinned* contra *pageable***

Descargar: [memoriaPinned.cu](../code/memoria/memoriaPinned.cu)

1. Medir el tiempo de las dos transferencias (*host* a *device* y de vuelta).
2. Reemplazar `cudaMallocHost`/`cudaFreeHost` por `malloc`/`free`: la memoria del *host* vuelve a ser *pageable*.
3. Medir de nuevo y comparar.

```bash
nvprof ./memoriaPinned.x
```

`nvprof` reporta las copias como `[CUDA memcpy HtoD]` y `[CUDA memcpy DtoH]`.

4. Si la memoria *pinned* es más rápida, ¿por qué no asignar **toda** la memoria del *host* así?

---

# Fin Capítulo 2

## Próxima capítulo: control de los threads
