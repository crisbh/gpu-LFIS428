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

## **Jerarquía de memoria (host)**

![w:560px](images/memoria/figure_4_1.png)

No se puede programar los registros y *caches*.

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Jerarquía de memoria (GPU)**

![w:560px](images/memoria/figure_4_2.png)

Se puede programar cualquier espacio de memoria que no sea *cache*.

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Códigos**

Los códigos de esta clase están disponibles para descargar:

- [variableGlobal.cu](../code/memoria/variableGlobal.cu), [variableGlobalDin.cu](../code/memoria/variableGlobalDin.cu)
- [copiarFila.cu](../code/memoria/copiarFila.cu), [copiarColumna.cu](../code/memoria/copiarColumna.cu)
- [transpuesta.cu](../code/memoria/transpuesta.cu), [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu)
- [aos.cu](../code/memoria/aos.cu), [soa.cu](../code/memoria/soa.cu), [alineamiento_datos.c](../code/memoria/alineamiento_datos.c)
- [memoria_constante.cu](../code/memoria/memoria_constante.cu), [memoriaPinned.cu](../code/memoria/memoriaPinned.cu), [memoria_unificada.cu](../code/memoria/memoria_unificada.cu)

---

# Memoria global

---

## **Memoria global**

- La memoria principal del GPU: *latency* alto, *bandwidth* bajo.
- Se puede asignar memoria global de forma **dinámica** con `cudaMalloc`.
- Se puede asignar memoria global de forma **estática** en el *device* con `__device__`.
- El uso eficiente de la memoria global es muy importante para optimizar un código de CUDA.

---

## **Memoria global: declaración estática**

Ejemplo 1: [variableGlobal.cu](../code/memoria/variableGlobal.cu) — declaramos una variable global en la memoria global del *device*.

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

Ejemplo 2: [variableGlobalDin.cu](../code/memoria/variableGlobalDin.cu) — el mismo programa, pero con declaración dinámica (ya no tiene *global scope*).

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

## **Memoria global: acceso eficiente**

La mejor forma de acceder a la memoria global es con acceso **alineado** y **contiguo**.

![w:520px](images/memoria/aligned_coalesced.png)

<p class="credit">Alineado y contiguo — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria global: acceso eficiente**

![w:520px](images/memoria/non_coalesced.png)

<p class="credit">No alineado ni contiguo — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria global: acceso eficiente**

El acceso alineado no es tan importante comparado con el acceso **contiguo**.

Ejemplo 3: [copiarFila.cu](../code/memoria/copiarFila.cu) y [copiarColumna.cu](../code/memoria/copiarColumna.cu).

![w:520px](images/memoria/row_column.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria global: acceso eficiente**

Métricas en `nvprof`:
- `gld_efficiency`, `gst_efficiency`

En `ncu`:
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct`

Eficiencia load/store para `copiarFila` de $100\%$. Para `copiarColumna` la eficiencia de load es $25\%$, y de store es $12.5\%$.

---

## **Memoria global: acceso eficiente**

- Las operaciones de **load** pasan por un *cache*.
- Pero las operaciones de **store** no pasan por *cache*, así que la eficiencia es menor para guardar valores.

**Conclusión importante:** el uso de la memoria global es mucho más eficiente con **acceso contiguo**.

---

## **Transpuesta de una matriz**

![w:520px](images/memoria/row_column.png)

<p class="credit">Cargar por fila, guardar por columna — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Transpuesta de una matriz**

![w:520px](images/memoria/column_row.png)

<p class="credit">Cargar por columna, guardar por fila — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Transpuesta de una matriz**

Ejemplo 4: [transpuesta.cu](../code/memoria/transpuesta.cu).

La versión que carga por columnas es más rápida... ¿por qué?

- Las cargas de datos pasan por el *cache*, mientras que la operación de guardar datos no utiliza ningún *cache*.
- Es mejor tener acceso contiguo para **guardar**, ya que no podemos aprovechar el *cache* en ese caso.

---

# AoS vs. SoA

---

## **Opciones para estructuras de datos**

![w:640px](images/memoria/figure_4_22.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Opciones para estructuras de datos**

Ejemplo 5: [aos.cu](../code/memoria/aos.cu) y [soa.cu](../code/memoria/soa.cu).

Normalmente se prefiere **SoA** (*structure of arrays*) en programación paralela: da acceso contiguo en el GPU.

---

## **Alineamiento de estructuras**

Paréntesis importante (relevante para la programación en general):

La organización de los elementos en una estructura tiene consecuencias para el uso de la memoria.

Ejemplo 5a: [alineamiento_datos.c](../code/memoria/alineamiento_datos.c).

---

# Memoria compartida

---

## **Memoria compartida**

![w:560px](images/memoria/figure_4_2.png)

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

![w:520px](images/memoria/figure_5_15.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Transpuesta: memoria compartida**

Ejemplo 6: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu). Hay tres *kernels*:

- El *kernel* para la transpuesta con memoria global.
- Un *kernel* que utiliza memoria compartida **estática**.
- Otro que utiliza memoria compartida **dinámica**.

---

## **Transpuesta: memoria compartida**

Consideramos un ejemplo: matriz de $4 \times 4$ elementos, con bloques de $2 \times 2$ (memoria compartida del mismo tamaño).

`blockDim.x`, `blockDim.y` son iguales a $2$; hay $2$ bloques en cada dimensión.

---

## **Transpuesta: memoria compartida**

![w:420px](images/memoria/transpose_fig1.png)

Índices globales de los *threads*:

```cuda
ix = blockDim.x * blockIdx.x + threadIdx.x;
iy = blockDim.y * blockIdx.y + threadIdx.y;
```

---

## **Transpuesta: memoria compartida**

![w:420px](images/memoria/transpose_fig2.png)

Índice lineal de los *threads*:

```cuda
ti = iy * N + ix;
```

---

## **Transpuesta: memoria compartida**

![w:420px](images/memoria/transpose_fig3.png)

Índices globales después de la "transpuesta de bloques":

```cuda
ixt = blockDim.y * blockIdx.y + threadIdx.x;
iyt = blockDim.x * blockIdx.x + threadIdx.y;
```

---

## **Transpuesta: memoria compartida**

![w:420px](images/memoria/transpose_fig4.png)

Índice lineal después de la "transpuesta de bloques":

```cuda
to = iyt * N + ixt;
```

---

## **Transpuesta: memoria compartida**

![w:420px](images/memoria/transpose_fig5.png)

Elementos guardados en la matriz de salida después de cargar de la memoria compartida:

```cuda
tile[threadIdx.y][threadIdx.x] = entrada[ti];
__syncthreads();
salida[to] = tile[threadIdx.x][threadIdx.y];
```

---

## **Acceso a la memoria compartida**

![w:520px](images/memoria/figure_5_2.png)

<p class="credit">Acceso ideal — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Acceso a la memoria compartida**

![w:520px](images/memoria/figure_5_3.png)

<p class="credit">Acceso desordenado, pero no problemático — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Acceso a la memoria compartida**

![w:520px](images/memoria/figure_5_4.png)

<p class="credit">Potencialmente problemático... — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Organización de la memoria compartida (bancos)**

![w:520px](images/memoria/figure_5_5.png)

<p class="credit">Bancos de ancho 4-bytes — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Organización de la memoria compartida (bancos)**

![w:520px](images/memoria/figure_5_6.png)

<p class="credit">Bancos de ancho 8-bytes — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Conflictos de bancos**

![w:520px](images/memoria/figure_5_7.png)

<p class="credit">Todo bien acá — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Conflictos de bancos**

![w:520px](images/memoria/figure_5_8.png)

<p class="credit">Todo bien acá también, gracias al ancho de 8-bytes — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Conflictos de bancos**

![w:520px](images/memoria/figure_5_9.png)

<p class="credit">¡Conflicto! — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Conflictos de bancos**

![w:520px](images/memoria/figure_5_10.png)

<p class="credit">¡Conflicto! — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Solución: *padding***

![w:520px](images/memoria/figure_5_11.png)

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

Ejemplo 7: [memoria_constante.cu](../code/memoria/memoria_constante.cu).

---

# Memoria unificada

---

## **Transferencias de memoria**

![w:560px](images/memoria/figure_4_3.png)

<p class="credit">Ejemplo para Fermi C2050 GPU — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria *pinned***

- La memoria en el *host* es, por defecto, *pageable*.
- Está organizada en páginas que el sistema operativo puede mover a la memoria virtual (en el disco duro).
- Cuando el sistema requiere datos que están en el disco, ocurre un *page fault* y los datos se copian del disco al RAM. El GPU no controla el movimiento de las páginas.
- Transferir datos del *host* al *device* implica asignar memoria *page-locked* o *pinned* en el *host*: los datos se transfieren de *pageable* a *pinned* y después al *device*.

---

## **Memoria *pinned***

![w:560px](images/memoria/figure_4_4.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Asignación de memoria *pinned***

```cuda
cudaError_t cudaMallocHost(void **devPtr, size_t count);
cudaError_t cudaFreeHost(void *ptr);
```

El uso de demasiada memoria *pinned* puede afectar el rendimiento del sistema entero, ya que reduce la cantidad de memoria *pageable* disponible.

Ejemplo 8: [memoriaPinned.cu](../code/memoria/memoriaPinned.cu).

---

## **Memoria unificada**

- Desde CUDA 6.0, *Unified Memory* permite acceder a la memoria usando un **solo espacio de direcciones** para el GPU y el CPU. UM se encarga de la transferencia de datos automáticamente.
- Basada en *Unified Virtual Addressing* (CUDA 4.0), que unificó el espacio de direcciones en memoria.
- Declaración estática (a veces llamada *managed*): `__device__ __managed__ int y;`
- Asignación dinámica:

```cuda
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=0);
```

El puntero `devPtr` es válido tanto en el *device* como en el *host*.

---

## **Memoria unificada**

Ejemplo 9: [memoria_unificada.cu](../code/memoria/memoria_unificada.cu).

---

# ¡Gracias!

## Próxima clase: control de los threads
