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
  - La diferencia puede ser entre uno y dos ordenes de magnitud en latencia.
<!-- - ¿Cuánto más lentas? Latencia en la T4: registros $\sim 4$ ciclos · L1 / compartida $\sim 30$ · L2 $\sim 200$ · DRAM $\sim 300$–$600$: **dos órdenes de magnitud**. -->
<!-- - En ancho de banda: DRAM $\approx 300$ GB/s contra $\approx 16$ GB/s del PCIe hacia el *host* ($\sim 20\times$). -->

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

## **Acceso eficiente a matrices**

![w:320px](images/memoria/figure_4_23.png)

- En memoria, una matriz $n_x\times n_y$ se almacena en forma 1D **por filas**.

![w:420px](images/memoria/figure_4_24.png)

- Supongamos que utilizamos un grid con bloques 2D.

---

## **El índice lineal**

![w:260px](images/memoria/transpose_fig2.png)

<p class="credit">Ejemplo: matriz 4×4 con bloques de 2×2 (líneas rojas)</p>

- Cada elemento tiene una **posición fija** en memoria.
- Un *thread* `(ix, iy)` puede calcular su **índice lineal** de dos formas:

```cuda
ti = iy * nx + ix; // por filas:    (fila iy, columna ix)
ti = ix * ny + iy; // por columnas: (fila ix, columna iy)
```

---

## **Acceso eficiente a matrices**

Esto define **qué elemento** le toca a cada *thread* al acceder a `matriz[ti]`.

- Por filas: `ti` $= 0, 1, 2, 3$ → posiciones **contiguas**.
- Por columnas: `ti` $= 0, 4, 8, 12$ → saltos de `ny` (una fila entera).

![w:620px](images/memoria/row_column.png)
<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Eficiencia del acceso**

- Recordar que los accesos de memoria no se solicitan individualmente para cada thread, si no que se realizan en términos de *warps*.
- A su vez, cada acceso de un *warp* se sirve en **sectores de $32$ bytes** (las líneas de L2).
- La **eficiencia del acceso** es la fracción de lo movido que realmente se usa:

$$\text{eficiencia de acceso} = \frac{\text{bytes útiles}}{32 \times \text{sectores tocados}}$$

---

## **Ejemplo: filas vs columnas**

Ejemplo: [copiarFila.cu](../code/memoria/copiarFila.cu) y [copiarColumna.cu](../code/memoria/copiarColumna.cu) (los dos necesitan [common.h](../code/memoria/common.h)).

- Matrices de $2048\times 2048$ elementos.
- Bloques 2D: $16\times 16$ threads.

Obtener las siguientes métricas con `ncu` (usando el flag `--metrics A,B`):
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct`

<!-- Resultados esperados: copiarFila 100% load y store; copiarColumna 25% load y store (bloques 16x16). El material original (2026-06) decía 12.5% para el store de copiarColumna, pero la cuenta da 25% para ambos, ya que usan el mismo índice: confirmar con ncu en la T4. -->

---

## **Ejercicio:**

1. ¿Cuál es el tamaño del grid necesario para cubrir la matriz?
2. En cada *warp*, ¿cuántos *threads* hay por fila y por columna del bloque?
3. ¿Cuánto *pesa* (en bytes) una fila de la matriz, si contiene valores tipo `float`?
4. ¿Cuántos bytes *útiles* requiere cada *warp* para operar?
5. ¿Cuántos sectores de memoria debe solicitar el *warp* para acceder a ellos, accediendo a la matriz por filas o por columnas?
6. Con la definición de eficiencia de acceso, ¿cual es el porcentaje esperado? ¿Coincide con lo medido?

<!-- 1. La matriz es de 2048x2048 y los bloques de 16x16: 2048/16 = 128 bloques por dimensión, o sea 128x128 = 16384 bloques en total. -->

<!-- 2. El warp son los primeros 32 threads del bloque: 16 threads en x (threadIdx.x = 0..15) por 2 filas en y (threadIdx.y = 0 y 1). -->

<!-- 3. Una fila son 2048 floats x 4 B = 8192 B = 8 KB. La matriz completa son 2048x8 KB = 16 MiB. -->

<!-- 4. Los 32 threads del warp piden un float cada uno: 32 x 4 B = 128 B útiles, igual al leer y al escribir. -->

<!-- 5. Por filas: cada una de las dos filas del warp es un tramo contiguo de 16x4 = 64 B, o sea 2 sectores de 32 B cada una; 4 sectores en total. Por columnas: los 16 valores de ix caen en 16 sectores distintos (iy e iy+1, separados por 4 B, sí comparten sector); 16 sectores en total. -->

<!-- 6. Por filas: 128/(4x32) = 128/128 = 100%. Por columnas: 128/(16x32) = 128/512 = 25%. Ambos valen para load y store, porque los dos kernels usan el mismo índice. Ojo: el material original (2026-06) daba 12.5% para el store de copiarColumna; esa cifra no se explica con este modelo y está sin confirmar en la T4. -->

---

## **Acceder por filas vs columnas**

Un *warp* pide $32 \times 4 = 128$ bytes útiles de `float`; el índice es el mismo al leer y al escribir, así que *load* y *store* rinden igual.

- **Por fila (contiguo)**: caben en $4$ sectores de $32$ bytes: $128 / 128 = 100\%$.
- **Por columnas**: cada *thread* cae en un sector distinto. Hasta $32$ sectores, $32 \times 32 = 1024$ bytes movidos por $128$ útiles: $12.5\%$.
  - Con bloques `16x16`, las dos filas del *warp* son vecinas y comparten sector: $16$ sectores, $128 / 512 = 25\%$.

<!-- El ancho de banda **efectivo** cae en ese mismo factor: la DRAM trabaja igual, pero la mayoría de los bytes que mueve no se usan. -->

**Conclusión importante:** el uso de la memoria global es mucho más eficiente con **acceso contiguo**.

---

## **Ejercicio: tamaño del bloque**

#### Ejercicio (propuesto)
Cambiar (`blockx`, `blocky`), manteniendo $256$ *threads* por bloque:

a) `32x8` · b) `16x16` · c) `8x32` · d) `4x64`

1. Antes de medir: ¿cuántos sectores de memoria pide un *warp* en cada caso?
2. ¿Puede `copiarColumna` llegar al $100\%$?
3. ¿El tiempo mejora en la misma proporción que la eficiencia de acceso?

<!-- Un warp son 32 threads consecutivos en el orden lineal (threadIdx.y * blockDim.x + threadIdx.x), así que cubre blockDim.x valores de ix y 32/blockDim.x valores de iy. Con bx = blockDim.x: copiarFila pide (32/bx) tramos de 4*bx bytes; copiarColumna pide bx tramos de 128/bx bytes. Eficiencia = 128 / (32 * sectores): -->

<!-- 32x8 → fila 4 sectores (100%), columna 32 sectores (12.5%) · 16x16 → 4 (100%) y 16 (25%) · 8x32 → 4 (100%) y 8 (50%) · 4x64 → 8 (50%) y 4 (100%). -->

<!-- O sea: el caso 32x8 es justamente el límite de "hasta 32 sectores → 12.5%" de la diapositiva anterior, y con 4x64 los dos patrones se INVIERTEN: copiarColumna llega a 100% porque los 8 iy consecutivos de cada ix llenan un sector completo, y copiarFila cae a 50% porque sus tramos de 4 floats (16 B) solo llenan medio sector. La respuesta a (2) es: a costa de arruinar el acceso por filas. -->

<!-- La pregunta (3) es la importante y hay que medirla, no deducirla: eficiencia y tiempo no son lo mismo. Con 4x64 el tráfico es mínimo pero los accesos siguen dispersos entre filas (8 KB de distancia), y bloques muy angostos en x pueden cambiar la ocupancia. No adelantar un resultado: que lo midan. -->

---

## **Transpuesta de una matriz**

- En algunas operaciones necesariamente utilizaremos ambos accesos: por filas y columnas.
- Ejemplo: al calcular la transpuesta de una matriz.
- Sin embargo, podemos elegir cuál utilizar para la matriz original (load) o al momento de guardar la transpuesta (store).

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

Ejemplo: [transpuesta.cu](../code/memoria/transpuesta.cu) (necesita [common.h](../code/memoria/common.h)).

Cada *kernels* tiene un acceso contiguo y uno disperso.

- `transpuestaCargarFilas`: carga contigua, **guarda** dispersa.
- `transpuestaCargarColumnas`: **carga** dispersa, guarda contigua.

**Regla:** si hay que tener un acceso disperso, conviene que sea el de **carga** — las cargas pasan por L1 y los *stores* no.

**Nota: en la T4 los tiempos son parecidos**: no es apreciable el beneficio en este caso.

<!-- NOTA — la cuenta, con 2048x2048 floats y bloques 16x16 (un warp son dos filas de 16 threads): el lado contiguo son dos tramos de 16*4 = 64 bytes, o sea 4 sectores; el lado disperso son 16 valores de ix separados por 8 KB, cada uno aportando un par iy, iy+1 de 8 bytes, o sea 16 sectores. Total 4 + 16 = 20 sectores por warp en LOS DOS kernels. Mismo tráfico, mismo tiempo. Contar sectores predice el empate antes de medir. -->

<!-- POR QUÉ EL LIBRO DICE QUE GANA CARGAR POR COLUMNAS — es un argumento de la época de Kepler, y ahí era cierto. En Kepler L1 traía las CARGAS en líneas de 128 bytes, mientras los stores esquivaban L1 y salían a L2 en segmentos de 32 bytes. Con esa asimetría de granularidad, poner el acceso disperso del lado de la carga movía bastante menos bytes: la línea de 128 bytes que traía un thread la reutilizaban los vecinos en iy de los otros warps del bloque. En una GPU de esa generación el ejemplo sí muestra la diferencia. -->

<!-- EN LA T4 la asimetría desapareció: Jia et al. (Dissecting the NVidia Turing T4 GPU via Microbenchmarking, 2019, tabla 3.1) miden la granularidad de carga de L1 en 32 bytes, igual que el sector de L2, contra 128 bytes en el K80. O sea que una carga dispersa trae un sector de 32 bytes, exactamente como un store disperso escribe un sector de 32 bytes. Lo que queda de la regla es que las cargas igual pueden ACERTAR en L1 y los stores no se alojan ahí, pero eso afecta latencia y tasa de aciertos, no la cantidad de bytes movidos, y en un kernel limitado por ancho de banda de este tamaño no se nota. -->

<!-- La regla no es falsa ni inútil: nunca puede empeorar las cosas y en hardware más viejo ayudaba mucho. Solo que en la T4 no es medible, y eso es justamente lo interesante para decir en clase. Es el tercer caso del capítulo donde una afirmación del libro atada a la arquitectura no se traslada a Turing (los otros: el flag -Xptxas -dlcm=cg y la equivalencia de tráfico entre AoS y SoA). La lección transferible es contar sectores, no memorizar reglas. -->

<!-- Medición del usuario en la T4 (2026-09): los dos kernels reportan tiempos parecidos, que es lo que predice la cuenta de sectores. -->


---

# Memoria compartida

---

## **Memoria compartida (shared memory)**

![w:460px](images/memoria/figure_4_2.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria compartida (shared memory)**

![w:660px](images/memoria/figure_5_1.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria compartida**

- La Memoria Compartida (**SMEM**) está ubicada *on-chip* (en el GPU):
  - *bandwidth* alto, *latency* bajo.

- Es compartida por todos los *threads* de cada bloque.
  - Permite **comunicación entre los *threads*** (dentro de un bloque).
  - Variables declaradas con `__shared__` se guardan en SMEM.

- Cada SM tiene una cantidad limitada de memoria compartida, dividida entre los bloques de *threads*. 
  - Si usamos demasiada, el número de *warps* activos se reduce.

---

## **Memoria compartida — declaración estática**

```cuda
__shared__ float tile[ny][nx];
```

- Declarada dentro de un *kernel*: *scope* local; declarada fuera de cualquier *kernel*: *scope* global.
- Como la memoria compartida está asociada a un bloque de *threads*, típicamente `nx` $=$ `blockDim.x` y `ny` $=$ `blockDim.y`.
- El **último** índice es el que avanza más rápido en memoria, por eso `nx` va al final. 
  - En otras palabras, el largo de una línea es igual al número de columnas, y vice versa.
  - Luego se indexa de la forma `tile[threadIdx.y][threadIdx.x]` y *threads* contiguos quedan contiguos.

<!-- NOTA — es la misma convención por filas de la matriz en memoria global: en tile[ny][nx] el primer índice es la fila (y) y el segundo la columna dentro de la fila (x), igual que en matriz[iy*nx + ix]. En C, tile[i][j] queda en el offset i*nx + j, así que el segundo índice es el que tiene vecinos contiguos. -->

<!-- Y no es cosmético: la memoria compartida tiene 32 bancos de 4 bytes, con banco = (índice de float) módulo 32. Con tile[32][32], escribir tile[ty][tx] cae en el offset ty*32 + tx, o sea banco = tx: los 32 threads del warp usan 32 bancos distintos y la ESCRITURA no tiene conflictos. Si se declarara al revés y se escribiera tile[tx][ty], threads consecutivos quedarían a 32 floats de distancia, todos en el banco ty: conflicto de 32 vías también al escribir. -->

<!-- Por eso la transpuesta deja el acceso con stride solo en la LECTURA (tile[threadIdx.x][threadIdx.y]), que es el único conflicto que después arregla transpuestaCompPad con tile[BDIM][BDIM+1]. Con el orden invertido habría conflicto en las dos puntas y el padding no alcanzaría. -->

<!-- En el código real (transpuesta_compartida.cu:54) el tile es tile[BDIM][BDIM], cuadrado, así que el orden no se nota: esta diapositiva es el único lugar donde aparece la forma general [ny][nx]. -->

---

## **Memoria compartida — declaración dinámica**

```cuda
extern __shared__ int tile[];
```

- Esto se usa cuando no conocemos el tamaño del arreglo al momento de la compilación.
- El tamaño del *array* se define al invocar el *kernel*, con el tercer argumento de la configuración (que hasta ahora habíamos omitido):

```cuda
kernel<<<grid, block, N * sizeof(int)>>>(...);
```

- Para declaración dinámica, solo se pueden declarar **arrays 1D**.

---

## **Transpuesta: memoria compartida**

- Volvamos al ejemplo de la transpuesta de una matriz.
- Usando memoria compartida, podemos optimizar los accesos de memoria.

![w:820px](images/memoria/figure_5_15.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

<!-- NOTA — por qué memoria compartida. En transpuestaGlobal el kernel hace salida[iy*N+ix] = entrada[ix*N+iy]: el store es contiguo, pero el load va por columnas y cada thread del warp cae en un sector distinto. Y no se puede arreglar dando vuelta el kernel: la transpuesta cambia el orden de los datos, así que en memoria global uno de los dos accesos es necesariamente disperso. La idea del tiling es sacar la transpuesta de la memoria global: traer un bloque (tile) con un acceso global contiguo, transponerlo DENTRO de la memoria compartida, y escribirlo de vuelta también con un acceso contiguo. La memoria compartida no tiene requisito de coalescencia — pero sí tiene bancos, que es el tema que viene después. -->

---

## **Transpuesta por bloques**

Consideremos un caso concreto: una matriz de $4 \times 4$ elementos, con bloques de $2 \times 2$ (memoria compartida del mismo tamaño).

`blockDim.x=2` 
`blockDim.y=2` 

Es decir, hay $2$ bloques en cada dirección, y cada tile tiene la forma:

```cuda
__shared__ float tile[2][2];
```

<!-- NOTA — el 4x4 con bloques de 2x2 es un ejemplo de juguete, elegido para poder dibujar los 16 índices en una diapositiva. El código real usa N = 4096 con BDIM = 32. Hay una diferencia que conviene tener presente: con BDIM = 32 un warp es exactamente UNA fila del bloque (threadIdx.y fijo, threadIdx.x = 0..31), y por eso el argumento de coalescencia de las diapositivas que siguen es exacto, no aproximado. En el dibujo de 2x2 un "warp" no existe como tal; el dibujo sirve solo para seguir los índices. -->

---

## **Transpuesta por bloques**

Transponer por bloques la matriz completa requiere **dos pasos**:

$$M = \begin{pmatrix} A & B \\ C & D \end{pmatrix} \longrightarrow M^{T} = \begin{pmatrix} A^{T} & C^{T} \\ B^{T} & D^{T} \end{pmatrix}$$

En este caso:
```
  0  1 |  2  3               0  1 |  8  9                    0  4 |  8 12
  4  5 |  6  7               4  5 | 12 13                    1  5 |  9 13
 ------+------              ------+------                   ------+------
  8  9 | 10 11               2  3 | 10 11                    2  6 | 10 14
 12 13 | 14 15               6  7 | 14 15                    3  7 | 11 15

   original               paso 1: mover bloques            paso 2 transponer
```


<!-- NOTA — esta es la idea que hace entendible todo el kernel. El PASO 1 mueve los bloques de lugar (B y C se intercambian) sin tocar lo que hay adentro, y lo hace la fórmula de ixt/iyt de la diapositiva siguiente. El PASO 2 transpone cada bloque por dentro, y lo hace el intercambio de índices en la memoria compartida (se escribe tile[ty][tx], se lee tile[tx][ty]). Ninguno de los dos por separado es una transpuesta. -->

<!-- Los tres paneles no son dibujos nuevos: son exactamente las tres figuras de esta misma secuencia. El primero es transpose_fig2.png (la de ti, diapositiva anterior), el del medio es transpose_fig4.png (la de to, dos diapositivas más adelante) y el tercero es transpose_fig5.png (la del final). Vale la pena decirlo para que el alumno vea que las figuras que vienen son estados de este mismo diagrama. -->

<!-- Analogía para decir en voz alta: cuatro fotos puestas en una grilla 2x2. Primero se REORDENAN las fotos sobre la mesa, intercambiando las dos de fuera de la diagonal. Después se DA VUELTA cada foto sobre su propia diagonal. Dos movimientos distintos, y hacen falta los dos. -->

<!-- El panel del medio no hay que dibujarlo aparte: es exactamente transpose_fig4.png, la figura de "to" que viene en dos diapositivas más. No es casualidad — transponer la grilla de bloques es una involución (hacerlo dos veces es la identidad), así que "a dónde va cada bloque" y "qué bloque llega acá" son el mismo mapa. Conviene decirlo con esa figura en pantalla: la misma imagen se lee como el índice lineal al que escribe cada thread, o como la matriz con los bloques ya movidos y los tiles todavía sin transponer. -->

<!-- Y por qué se separa así: el paso 1 es una permutación de TILES completos, y permutar tiles no rompe la contigüidad (dentro de un warp ixt sigue avanzando de a uno). Por eso los dos accesos globales quedan coalescidos y el único acceso con stride queda dentro de la memoria compartida, donde el costo son conflictos de bancos y no sectores desperdiciados. Ese es el canje sobre el que está construido el algoritmo. -->


---

## **Transpuesta por bloques**

![w:340px](images/memoria/transpose_fig1.png)

Índices globales de los *threads*:

```cuda
ix = blockDim.x * blockIdx.x + threadIdx.x;
iy = blockDim.y * blockIdx.y + threadIdx.y;
```

<!-- NOTA — cada casilla de la figura es el elemento que le toca a un thread, etiquetado (ix, iy) = (columna, fila). Las líneas rojas son los límites de los bloques de 2x2. Lo que hay que hacer notar: ix avanza a lo largo de la fila y lo maneja threadIdx.x, igual que en el ejemplo de copiarFila. Hasta aquí no hay nada nuevo: es el mismo mapeo thread → elemento de la clase anterior. -->

---

## **Transpuesta: memoria compartida**

![w:340px](images/memoria/transpose_fig2.png)

Índice lineal de los *threads*:

```cuda
ti = iy * N + ix;
```

<!-- NOTA — es la misma figura del índice lineal que ya usamos en memoria global, y la misma fórmula. ti es la posición EN MEMORIA del elemento que carga el thread (ix, iy). Lo importante para lo que viene: dentro de un warp (iy fijo, ix consecutivo) los ti son consecutivos, así que el load entrada[ti] es contiguo. Ese es el primero de los dos accesos globales del kernel, y ya está bien. -->

---

## **Transpuesta por bloques: paso 1**

![w:340px](images/memoria/transpose_fig3.png)

Índices globales después del **paso 1** (mover bloques):

```cuda
ixt = blockDim.y * blockIdx.y + threadIdx.x;
iyt = blockDim.x * blockIdx.x + threadIdx.y;
```

<!-- NOTA — este es el paso clave de todo el algoritmo, y conviene detenerse. Comparar con la diapositiva de ix/iy: se intercambian los índices de BLOQUE (blockIdx.x <-> blockIdx.y), pero NO los de thread — threadIdx.x sigue estando en la coordenada x. De ahí el paso 1: el bloque (bx, by) escribe en la posición de bloque (by, bx) de la salida, y dentro del bloque cada thread conserva su casilla. En la figura se ve en el bloque de arriba a la derecha (bx=1, by=0): sus etiquetas son (0,2), (1,2), (0,3), (1,3), o sea sus threads escribirán en el bloque de abajo a la izquierda. El paso 1 (mover los bloques) lo hace esta fórmula; el paso 2 (transponer dentro del bloque) lo hará la memoria compartida. -->

---

## **Transpuesta por bloques: paso 1**

![w:340px](images/memoria/transpose_fig4.png)

Índice lineal después del **paso 1**:

```cuda
to = iyt * N + ixt;
```

<!-- NOTA — lo que hay que mirar acá es qué pasa DENTRO de un warp. Con threadIdx.y fijo y threadIdx.x = 0..31: ixt = blockDim.y*blockIdx.y + threadIdx.x varía de a uno, mientras iyt queda constante. Entonces los to también son consecutivos, o sea el store salida[to] es contiguo. Junto con la diapositiva anterior (load contiguo), el resultado es que LOS DOS accesos a memoria global quedan coalescidos: ninguno de los dos hace la transpuesta. En la figura de 4x4 se ve en la primera fila: to = 0, 1, 8, 9 — dentro de cada bloque los valores son consecutivos, y el salto de 1 a 8 es el salto entre bloques, no entre threads de un warp. -->

---

## **Transpuesta por bloques: paso 2**

![w:320px](images/memoria/transpose_fig5.png)

Elementos guardados después de cargar de la memoria compartida.

```cuda
tile[threadIdx.y][threadIdx.x] = entrada[ti]; // escribe por filas en SMEM (paso 1)
__syncthreads();                              // sincronizamos bloque
salida[to] = tile[threadIdx.x][threadIdx.y];  // lee por columnas desde SMEM (paso 2)
```

<!-- Ojo con los comentarios del código: dicen "en SMEM" porque cada línea toca
DOS memorias a la vez, y lo que describen es solo el lado de la compartida. La
primera línea LEE de global (entrada[ti], por filas, contiguo) y ESCRIBE en
compartida por filas. La tercera LEE de compartida por columnas y ESCRIBE en
global (salida[ to], por filas, contiguo). O sea: el único acceso con stride en
todo el kernel es la lectura del tile. Si alguien lee "por columnas" como si
hablara de la memoria global, entiende justo lo contrario de lo que se demostró
en las dos diapositivas anteriores. -->

<!-- NOTA — estas tres líneas son el núcleo del kernel, y el paso 2 está en el
cambio de índices del tile: se ESCRIBE tile[threadIdx.y][threadIdx.x] (por
filas) y se LEE tile[threadIdx.x][threadIdx.y] (por columnas). Eso es lo que da
vuelta los datos, y ocurre en memoria compartida, no en la global. -->

<!-- Ejemplo concreto sobre la figura de 4x4: el thread (tx,ty) = (1,0) del
bloque (0,0) carga entrada[1] en tile[0][1], y después escribe salida[1] =
tile[1][0]; pero tile[1][0] lo cargó OTRO thread, el (0,1), desde entrada[4]. O
sea salida[1] = entrada[4], que es exactamente lo que muestra la figura: en la
posición 1 aparece un 4. Vale la pena hacer este seguimiento en vivo con un par
de casillas. -->

<!-- NOTA — y acá está la trampa que justifica la clase siguiente: leer el tile
por columnas es precisamente lo que provoca conflictos de bancos. Con
tile[32][32], el elemento tile[i][ty] queda en el índice i*32+ty, así que su
banco es (i*32+ty)%32 = ty: los 32 threads del warp piden el MISMO banco, un
conflicto de 32 vías. Con tile[32][33] (el padding de transpuestaCompPad) el
índice pasa a i*33+ty y el banco a (i+ty)%32, que es distinto para cada thread.
Por eso un solo carácter cambia tanto el tiempo. -->

---

## **Transpuesta por bloques: paso 2**

Notar el uso de `__syncthreads()` en el código anterior.

- Necesitamos garantizar que todos los *threads* en un bloque tendrán la información disponible en la memoria compartida antes de escribir `salida[to]`.
- Esto es el mismo concepto de *barrera* que se utiliza en `openMP` o `MPI`.
- Si no se usa, a veces el programa podría funcionar, pero no hay garantías de esto, por lo que decimos que su comportamiento es *indefinido*.

<!-- NOTA — la razón precisa: el thread (tx, ty) lee tile[tx][ty], una casilla que NO escribió él sino el thread (ty, tx). Sin la barrera hay una condición de carrera, porque nada garantiza que ese otro thread ya haya hecho su escritura. __syncthreads() es una barrera a nivel de BLOQUE, no del grid, y con eso alcanza porque el tile es privado del bloque. -->

<!-- Si se borra, el programa igual compila y a veces "funciona", pero el resultado es indefinido: desde Volta los threads de un warp no avanzan necesariamente en lock-step (independent thread scheduling), así que no hay que confiar en el viejo argumento de sincronía dentro del warp. Con BDIM = 32 el bloque tiene 32 warps y la corrupción es fácil de observar. Ese es el punto de la pregunta 3 del ejercicio. -->

<!-- Notar también que la barrera está DENTRO del if (ixt < N && iyt < N). Es correcta aquí porque N = 4096 es múltiplo de BDIM = 32 y la condición es uniforme en todo el bloque, pero si N no fuera múltiplo del bloque habría threads que no llegan a la barrera: __syncthreads() en código divergente es comportamiento indefinido. Buen detalle para mencionar si alguien pregunta. -->

<!-- Y un detalle del código, no del algoritmo: en main, transpuestaCompPad se lanza con un tercer argumento de memoria compartida dinámica aunque declara su tile de forma estática. No es un error (la memoria dinámica queda sin usar), pero reserva BDIM*(BDIM+1)*4 bytes de más por bloque y puede bajar la ocupancia. -->

---

## **Ejemplo Transpuesta con memoria compartida**

Ejemplo: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu) (necesita [common.h](../code/memoria/common.h)).

Hay cuatro *kernels*, y el programa **mide e imprime el tiempo de cada uno**:

- `transpuestaGlobal`: la transpuesta con memoria global.
- `transpuestaComp`: memoria compartida **estática**.
- `transpuestaCompDin`: memoria compartida **dinámica**.
- `transpuestaCompPad`: memoria compartida estática con ***padding*** (lo vemos en la clase siguiente, al llegar a conflictos de bancos).

<!-- NOTA — el programa usa N = 4096 y BDIM = 32, o sea bloques de 32x32 = 1024 threads. Los cuatro kernels son la misma transpuesta con distinta estrategia: transpuestaGlobal es la referencia sin memoria compartida; transpuestaComp y transpuestaCompDin son el MISMO algoritmo con declaración estática y dinámica, y deben dar tiempos casi iguales (sirve justamente para mostrar que la declaración dinámica no cuesta rendimiento, solo flexibilidad); transpuestaCompPad agrega el padding que se explica en "Conflictos de bancos". El ejercicio de esta clase compara los tres primeros; el cuarto queda para la clase siguiente. -->

<!-- MEDICIÓN DE REFERENCIA en la T4 (2026-09), N = 4096, bloques 32x32. Ancho de banda efectivo = 2*N*N*4 bytes / tiempo: transpuestaHost 0.43 s (0.3 GB/s) · transpuestaGlobal 1.547 ms (86.8 GB/s) · transpuestaComp 1.429 ms (93.9 GB/s) · transpuestaCompDin 1.438 ms (93.3 GB/s) · transpuestaCompPad 0.766 ms (175.2 GB/s). Los números pueden variar entre corridas, pero las RELACIONES se repiten. -->

<!-- Lo que hay que leer en esos números, y es más interesante de lo que parece: (1) estática y dinámica difieren en 0.6%, o sea son la misma cosa — declarar dinámico cuesta flexibilidad, no rendimiento. (2) La memoria compartida por sí sola gana apenas un 8% sobre la versión global (86.8 -> 93.9), aunque mueve MUCHO menos tráfico global. (3) El padding casi duplica: 1.87x sobre transpuestaComp y 2.02x sobre transpuestaGlobal, llegando a 175 GB/s, un 58% del peak de 300 GB/s de la T4. -->

<!-- POR QUÉ la memoria compartida sola casi no gana, que es la pregunta que va a salir: con bloques de 32x32 un warp es una fila (tx = 0..31, ty fijo). transpuestaGlobal pide 32 sectores en la carga dispersa más 4 en el store contiguo = 36 sectores por warp. transpuestaComp deja los DOS accesos globales contiguos: 4 + 4 = 8 sectores por warp, o sea 4.5 veces menos tráfico global. Y sin embargo solo gana 8%. El motivo es que el cuello de botella se mudó: leer tile[threadIdx.x][threadIdx.y] con tile[32][32] da banco = (tx*32+ty)%32 = ty para los 32 threads, un conflicto de 32 vías que serializa la lectura de memoria compartida en 32 transacciones. Con tile[32][33] el banco pasa a (tx+ty)%32, todos distintos, y vuelve a ser una sola transacción: ahí aparece el 1.87x. -->

<!-- Moraleja para decir en voz alta: la memoria compartida no es gratis ni automática. Mal usada, cambia un problema de coalescencia por uno de bancos y no gana nada. El ejemplo completo solo cierra con el padding de la clase siguiente. -->


---

# Organización de la memoria compartida

---

## **Acceso a la memoria compartida**

![w:1020px](images/memoria/figure_5_2.png)

<p class="credit">Acceso ideal — Fuente: <em>Professional CUDA C Programming</em></p>

- La Memoria Compartida se organiza en 32 *bancos*.
  - En principio, 1 para cada *thread* de un *warp*.
- El patrón de acceso a los bancos también influye en el rendimiento.
<!-- - **Regla de los bancos**: evitar que más de un *thread* acceda simultáneamente a un mismo elemento del banco. -->

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

<!-- --- -->
<!---->
<!-- ## **Organización de la memoria compartida (bancos)** -->
<!---->
<!-- ![w:1020px](images/memoria/figure_5_6.png) -->
<!---->
<!-- <p class="credit">Bancos de ancho 8-bytes — Fuente: <em>Professional CUDA C Programming</em></p> -->

---

# Ejercicios

---

## **Ejercicio 1: ¿ayuda la memoria compartida?**

Descargar: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu) y [common.h](../code/memoria/common.h)

El programa reporta el tiempo de cada *kernel*. Por ahora nos interesan tres:

- `transpuestaGlobal` · `transpuestaComp` · `transpuestaCompDin`

1. Ordenarlos de más lento a más rápido. ¿Cuánto se gana con memoria compartida?
2. ¿La declaración dinámica cuesta más que la estática?
3. ¿Qué pasaría si borráramos el `__syncthreads()`?

<!-- RESPUESTAS medidas en la T4 (2026-09), N = 4096: transpuestaGlobal 86.8 GB/s, transpuestaComp 93.9 GB/s, transpuestaCompDin 93.3 GB/s. -->

<!-- (1) La respuesta a "cuánto se gana" es INCÓMODA a propósito: apenas un 8%. Hay que dejar que los alumnos se lleven esa sorpresa, porque es la que motiva la clase siguiente. Si alguien pregunta por qué tan poco: los dos accesos globales quedaron contiguos (8 sectores por warp contra 36 del kernel global), pero el cuello de botella se mudó a la memoria compartida, donde leer el tile por columnas provoca un conflicto de bancos de 32 vías. El detalle está en la nota de la diapositiva del código. -->

<!-- (2) No: 93.9 contra 93.3 GB/s, un 0.6% de diferencia, que es ruido. La declaración dinámica cuesta flexibilidad (hay que pasar el tamaño al lanzar, y solo arrays 1D), no rendimiento. -->

<!-- (3) Condición de carrera: el thread (tx,ty) lee una casilla que escribió el thread (ty,tx). Sin la barrera el resultado es indefinido — puede "funcionar" y no hay que confiar en eso. Conviene que lo prueben: con bloques de 32x32 hay 32 warps por bloque y la corrupción se ve fácil. -->

<!-- Cuando lleguen al ejercicio 2 de la clase siguiente, transpuestaCompPad da 175.2 GB/s: 1.87x sobre transpuestaComp. Ahí recién se cobra la memoria compartida. -->

---

## **Ejercicio 1: los índices en papel**

Repetir el desarrollo de las diapositivas anteriores para una matriz de $8 \times 8$ con bloques de $4 \times 4$.

Para el *thread* `threadIdx = (1,2)` del bloque `blockIdx = (1,0)`, calcular:

```cuda
ix, iy      // índices globales
ti          // índice lineal de entrada
ixt, iyt    // índices tras el paso 1
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
- Transferir datos del *host* al *device* implica asignar memoria *page-locked* o *pinned* en el *host*: los datos se transfieren de *paginable* a *pinned* y después al *device*.

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

El uso de demasiada memoria *pinned* puede afectar el rendimiento del sistema entero, ya que reduce la cantidad de memoria *paginable* disponible.

Ejemplo: [memoriaPinned.cu](../code/memoria/memoriaPinned.cu).

---

## **Memoria unificada**

- Desde CUDA 6.0, *Unified Memory* (UM) permite acceder a la memoria usando un **solo espacio de direcciones** para el GPU y el CPU. 
  - UM se encarga de la transferencia de datos automáticamente.
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

## **Ejercicio 2: conflictos de bancos y *padding***

Descargar: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu) y [common.h](../code/memoria/common.h)

En el ejercicio 1 ya comparamos los tres primeros *kernels*. Falta el cuarto: `transpuestaCompPad`.

1. `transpuestaComp` y `transpuestaCompPad` difieren en **un carácter**: `tile[BDIM][BDIM]` contra `tile[BDIM][BDIM+1]`. ¿Cuánto cambia el tiempo?
2. ¿Dónde queda `transpuestaCompPad` en el ranking del ejercicio 1?

<!-- RESPUESTAS medidas en la T4 (2026-09), N = 4096: transpuestaCompPad 0.766 ms, 175.2 GB/s. (1) Un carácter da 1.87x sobre transpuestaComp (93.9 -> 175.2 GB/s) y 2.02x sobre transpuestaGlobal. (2) Pasa a ser el más rápido por lejos, y recién ahí la memoria compartida se paga: en el ejercicio 1 ganaba apenas 8%. -->

<!-- La cuenta de bancos: leer tile[threadIdx.x][threadIdx.y] con tile[32][32] pone el elemento en el índice tx*32+ty, o sea banco (tx*32+ty)%32 = ty, el MISMO para los 32 threads del warp: conflicto de 32 vías, 32 transacciones serializadas. Con tile[32][33] el índice pasa a tx*33+ty y el banco a (tx+ty)%32, distinto para cada thread: una sola transacción. La escritura tile[ty][tx] no tiene conflicto en ninguno de los dos casos. -->

<!-- Vale la pena cerrar con la perspectiva: 175 GB/s es un 58% del peak de 300 GB/s de la T4. Una transpuesta no llega al 100% porque los bloques recorren la DRAM de forma dispersa, pero pasar de 29% (global) a 58% cambiando un carácter es el mejor argumento del capítulo. -->

---

## **Ejercicio 2: métricas**

En `nvprof`:
- `shared_load_transactions_per_request`
- `shared_store_transactions_per_request`

En `ncu`:
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`

3. Medir los conflictos de bancos en `transpuestaComp` y en `transpuestaCompPad`. ¿El número explica la diferencia de tiempo?

---

## **Ejercicio 3: ¿sirve la memoria constante?**

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

## **Ejercicio 3: ¿por qué?**

3. Todos los *threads* leen **el mismo** ángulo en cada iteración del ciclo. ¿Por qué esto favorece a la memoria constante?
4. ¿Qué pasaría si cada *thread* leyera un ángulo **distinto**?

El programa invoca un tercer *kernel* (`kernel_inicial`) antes de medir. ¿Para qué sirve?

---

## **Ejercicio 4: *pinned* contra *paginable***

Descargar: [memoriaPinned.cu](../code/memoria/memoriaPinned.cu)

1. Medir el tiempo de las dos transferencias (*host* a *device* y de vuelta).
2. Reemplazar `cudaMallocHost`/`cudaFreeHost` por `malloc`/`free`: la memoria del *host* vuelve a ser *paginable*.
3. Medir de nuevo y comparar.

```bash
nvprof ./memoriaPinned.x
```

`nvprof` reporta las copias como `[CUDA memcpy HtoD]` y `[CUDA memcpy DtoH]`.

4. Si la memoria *pinned* es más rápida, ¿por qué no asignar **toda** la memoria del *host* así?

---

# Fin Capítulo 2

## Próxima capítulo: control de los threads
