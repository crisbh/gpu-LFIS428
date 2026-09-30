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

**Regla:** si hay que tener un acceso disperso, conviene que sea el de **carga**. Las cargas pasan por L1 y threads pueden reutilizar datos. Los *stores* no.

**Nota: en la T4 los tiempos son parecidos**: no es apreciable el beneficio en este caso.

<!-- NOTA — la cuenta, con 2048x2048 floats y bloques 16x16 (un warp son dos
filas de 16 threads): el lado contiguo son dos tramos de 16*4 = 64 bytes, o sea
4 sectores; el lado disperso son 16 valores de ix separados por 8 KB, cada uno
aportando un par iy, iy+1 de 8 bytes, o sea 16 sectores. Total 4 + 16 = 20
sectores por warp en LOS DOS kernels. Mismo tráfico, mismo tiempo. Contar
sectores predice el empate antes de medir. -->

<!-- POR QUÉ EL LIBRO DICE QUE GANA CARGAR POR COLUMNAS — es un argumento de la
época de Kepler, y ahí era cierto. En Kepler L1 traía las CARGAS en líneas de
128 bytes, mientras los stores esquivaban L1 y salían a L2 en segmentos de 32
bytes. Con esa asimetría de granularidad, poner el acceso disperso del lado de
la carga movía bastante menos bytes: la línea de 128 bytes que traía un thread
la reutilizaban los vecinos en iy de los otros warps del bloque. En una GPU de
esa generación el ejemplo sí muestra la diferencia. -->

<!-- EN LA T4 la asimetría desapareció: Jia et al. (Dissecting the NVidia Turing
T4 GPU via Microbenchmarking, 2019, tabla 3.1) miden la granularidad de carga de
L1 en 32 bytes, igual que el sector de L2, contra 128 bytes en el K80. O sea que
una carga dispersa trae un sector de 32 bytes, exactamente como un store
disperso escribe un sector de 32 bytes. Lo que queda de la regla es que las
cargas igual pueden ACERTAR en L1 y los stores no se alojan ahí, pero eso afecta
latencia y tasa de aciertos, no la cantidad de bytes movidos, y en un kernel
limitado por ancho de banda de este tamaño no se nota. -->

<!-- La regla no es falsa ni inútil: nunca puede empeorar las cosas y en
hardware más viejo ayudaba mucho. Solo que en la T4 no es medible, y eso es
justamente lo interesante para decir en clase. Es el tercer caso del capítulo
donde una afirmación del libro atada a la arquitectura no se traslada a Turing
(los otros: el flag -Xptxas -dlcm=cg y la equivalencia de tráfico entre AoS y
SoA). La lección transferible es contar sectores, no memorizar reglas. -->

<!-- Medición del usuario en la T4 (2026-09): los dos kernels reportan tiempos
parecidos, que es lo que predice la cuenta de sectores. -->


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
- El **último** índice (`nx`) es el que avanza más rápido en memoria.
  - En otras palabras, el largo de una línea es igual al número de columnas, y vice versa.
  - Luego se indexa de la forma `tile[threadIdx.y][threadIdx.x]` y *threads* contiguos quedan contiguos.

<!-- NOTA — es la misma convención por filas de la matriz en memoria global: en
tile[ny][nx] el primer índice es la fila (y) y el segundo la columna dentro de
la fila (x), igual que en matriz[iy*nx + ix]. En C, tile[i][j] queda en el
offset i*nx + j, así que el segundo índice es el que tiene vecinos contiguos.
-->

<!-- Y no es cosmético: la memoria compartida tiene 32 bancos de 4 bytes, con
banco = (índice de float) módulo 32. Con tile[32][32], escribir tile[ty][tx] cae
en el offset ty*32 + tx, o sea banco = tx: los 32 threads del warp usan 32
bancos distintos y la ESCRITURA no tiene conflictos. Si se declarara al revés y
se escribiera tile[tx][ty], threads consecutivos quedarían a 32 floats de
distancia, todos en el banco ty: conflicto de 32 vías también al escribir. -->

<!-- Por eso la transpuesta deja el acceso con stride solo en la LECTURA
(tile[threadIdx.x][threadIdx.y]), que es el único conflicto que después arregla
transpuestaCompPad con tile[BDIM][BDIM+1]. Con el orden invertido habría
conflicto en las dos puntas y el padding no alcanzaría. -->

<!-- En el código real (transpuesta_compartida.cu:54) el tile es
tile[BDIM][BDIM], cuadrado, así que el orden no se nota: esta diapositiva es el
único lugar donde aparece la forma general [ny][nx]. -->

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

<!-- NOTA — por qué memoria compartida. En transpuestaGlobal el kernel hace
salida[iy*N+ix] = entrada[ix*N+iy]: el store es contiguo, pero el load va por
columnas y cada thread del warp cae en un sector distinto. Y no se puede
arreglar dando vuelta el kernel: la transpuesta cambia el orden de los datos,
así que en memoria global uno de los dos accesos es necesariamente disperso. La
idea del tiling es sacar la transpuesta de la memoria global: traer un bloque
(tile) con un acceso global contiguo, transponerlo DENTRO de la memoria
compartida, y escribirlo de vuelta también con un acceso contiguo. La memoria
compartida no tiene requisito de coalescencia — pero sí tiene bancos, que es el
tema que viene después. -->

---

## **Transpuesta por bloques**

Este método se conoce como **tiling** (*tile*: baldosa, como las del piso).

Consideremos un caso concreto: una matriz de $4 \times 4$ elementos, con bloques de $2 \times 2$ (memoria compartida del mismo tamaño).

`blockDim.x=2` 
`blockDim.y=2` 

Es decir, hay $2$ bloques en cada dirección, y cada tile tiene la forma:

```cuda
__shared__ float tile[2][2];
```


<!-- NOTA — el 4x4 con bloques de 2x2 es un ejemplo de juguete, elegido para
poder dibujar los 16 índices en una diapositiva. El código real usa N = 4096 con
BDIM = 32. Hay una diferencia que conviene tener presente: con BDIM = 32 un warp
es exactamente UNA fila del bloque (threadIdx.y fijo, threadIdx.x = 0..31), y
por eso el argumento de coalescencia de las diapositivas que siguen es exacto,
no aproximado. En el dibujo de 2x2 un "warp" no existe como tal; el dibujo sirve
solo para seguir los índices. -->

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


<!-- NOTA — esta es la idea que hace entendible todo el kernel. El PASO 1 mueve
los bloques de lugar (B y C se intercambian) sin tocar lo que hay adentro, y lo
hace la fórmula de ixt/iyt de la diapositiva siguiente. El PASO 2 transpone cada
bloque por dentro, y lo hace el intercambio de índices en la memoria compartida
(se escribe tile[ty][tx], se lee tile[tx][ty]). Ninguno de los dos por separado
es una transpuesta. -->

<!-- Los tres paneles no son dibujos nuevos: son exactamente las tres figuras de
esta misma secuencia. El primero es transpose_fig2.png (la de ti, diapositiva
anterior), el del medio es transpose_fig4.png (la de to, dos diapositivas más
adelante) y el tercero es transpose_fig5.png (la del final). Vale la pena
decirlo para que el alumno vea que las figuras que vienen son estados de este
mismo diagrama. -->

<!-- Analogía para decir en voz alta: cuatro fotos puestas en una grilla 2x2.
Primero se REORDENAN las fotos sobre la mesa, intercambiando las dos de fuera de
la diagonal. Después se DA VUELTA cada foto sobre su propia diagonal. Dos
movimientos distintos, y hacen falta los dos. -->

<!-- El panel del medio no hay que dibujarlo aparte: es exactamente
transpose_fig4.png, la figura de "to" que viene en dos diapositivas más. No es
casualidad — transponer la grilla de bloques es una involución (hacerlo dos
veces es la identidad), así que "a dónde va cada bloque" y "qué bloque llega
acá" son el mismo mapa. Conviene decirlo con esa figura en pantalla: la misma
imagen se lee como el índice lineal al que escribe cada thread, o como la matriz
con los bloques ya movidos y los tiles todavía sin transponer. -->

<!-- Y por qué se separa así: el paso 1 es una permutación de TILES completos, y
permutar tiles no rompe la contigüidad (dentro de un warp ixt sigue avanzando de
a uno). Por eso los dos accesos globales quedan coalescidos y el único acceso
con stride queda dentro de la memoria compartida, donde el costo son conflictos
de bancos y no sectores desperdiciados. Ese es el canje sobre el que está
construido el algoritmo. -->


---

## **Transpuesta por bloques**

![w:340px](images/memoria/transpose_fig1.png)

Índices globales de los *threads*:

```cuda
ix = blockDim.x * blockIdx.x + threadIdx.x;
iy = blockDim.y * blockIdx.y + threadIdx.y;
```

<!-- NOTA — cada casilla de la figura es el elemento que le toca a un thread,
etiquetado (ix, iy) = (columna, fila). Las líneas rojas son los límites de los
bloques de 2x2. Lo que hay que hacer notar: ix avanza a lo largo de la fila y lo
maneja threadIdx.x, igual que en el ejemplo de copiarFila. Hasta aquí no hay
nada nuevo: es el mismo mapeo thread → elemento de la clase anterior. -->

---

## **Transpuesta: memoria compartida**

![w:340px](images/memoria/transpose_fig2.png)

Índice lineal de los *threads* (contiguos, i.e. por filas):

```cuda
ti = iy * N + ix;
```

<!-- NOTA — es la misma figura del índice lineal que ya usamos en memoria
global, y la misma fórmula. ti es la posición EN MEMORIA del elemento que carga
el thread (ix, iy). Lo importante para lo que viene: dentro de un warp (iy fijo,
ix consecutivo) los ti son consecutivos, así que el load entrada[ti] es
contiguo. Ese es el primero de los dos accesos globales del kernel, y ya está
bien. -->

---

## **Transpuesta por bloques: paso 1**

![w:340px](images/memoria/transpose_fig3.png)

Índices globales después del **paso 1** (mover bloques):

```cuda
ixt = blockDim.y * blockIdx.y + threadIdx.x;
iyt = blockDim.x * blockIdx.x + threadIdx.y;
```

<!-- NOTA — este es el paso clave de todo el algoritmo, y conviene detenerse.
Comparar con la diapositiva de ix/iy: se intercambian los índices de BLOQUE
(blockIdx.x <-> blockIdx.y), pero NO los de thread — threadIdx.x sigue estando
en la coordenada x. De ahí el paso 1: el bloque (bx, by) escribe en la posición
de bloque (by, bx) de la salida, y dentro del bloque cada thread conserva su
casilla. En la figura se ve en el bloque de arriba a la derecha (bx=1, by=0):
sus etiquetas son (0,2), (1,2), (0,3), (1,3), o sea sus threads escribirán en el
bloque de abajo a la izquierda. El paso 1 (mover los bloques) lo hace esta
fórmula; el paso 2 (transponer dentro del bloque) lo hará la memoria compartida.
-->

---

## **Transpuesta por bloques: paso 1**

![w:340px](images/memoria/transpose_fig4.png)

Índice lineal después del **paso 1**:

```cuda
to = iyt * N + ixt;
```

<!-- NOTA — lo que hay que mirar acá es qué pasa DENTRO de un warp. Con
threadIdx.y fijo y threadIdx.x = 0..31: ixt = blockDim.y*blockIdx.y +
threadIdx.x varía de a uno, mientras iyt queda constante. Entonces los to
también son consecutivos, o sea el store salida[to] es contiguo. Junto con la
diapositiva anterior (load contiguo), el resultado es que LOS DOS accesos a
memoria global quedan coalescidos: ninguno de los dos hace la transpuesta. En la
figura de 4x4 se ve en la primera fila: to = 0, 1, 8, 9 — dentro de cada bloque
los valores son consecutivos, y el salto de 1 a 8 es el salto entre bloques, no
entre threads de un warp. -->

---

## **Transpuesta por bloques: paso 2**

![w:320px](images/memoria/transpose_fig5.png)

Elementos de `salida` después de cargar de la memoria compartida.

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

- Necesitamos garantizar que todos los **threads en un bloque** tendrán la información disponible en la memoria compartida antes de escribir `salida[to]`.
- Este es el mismo concepto de *barrera* que se utiliza en `openMP` o `MPI`.
- Si no se usa, a veces el programa podría funcionar, pero no hay garantías de esto, por lo que decimos que su comportamiento es *indefinido*.

<!-- NOTA — la razón precisa: el thread (tx, ty) lee tile[tx][ty], una casilla
que NO escribió él sino el thread (ty, tx). Sin la barrera hay una condición de
carrera, porque nada garantiza que ese otro thread ya haya hecho su escritura.
__syncthreads() es una barrera a nivel de BLOQUE, no del grid, y con eso alcanza
porque el tile es privado del bloque. -->

<!-- Si se borra, el programa igual compila y a veces "funciona", pero el
resultado es indefinido: desde Volta los threads de un warp no avanzan
necesariamente en lock-step (independent thread scheduling), así que no hay que
confiar en el viejo argumento de sincronía dentro del warp. Con BDIM = 32 el
bloque tiene 32 warps y la corrupción es fácil de observar. Ese es el punto de
la pregunta 3 del ejercicio. -->

<!-- Notar también que la barrera está DENTRO del if (ixt < N && iyt < N). Es
correcta aquí porque N = 4096 es múltiplo de BDIM = 32 y la condición es
uniforme en todo el bloque, pero si N no fuera múltiplo del bloque habría
threads que no llegan a la barrera: __syncthreads() en código divergente es
comportamiento indefinido. Buen detalle para mencionar si alguien pregunta. -->

<!-- Y un detalle del código, no del algoritmo: en main, transpuestaCompPad se
lanza con un tercer argumento de memoria compartida dinámica aunque declara su
tile de forma estática. No es un error (la memoria dinámica queda sin usar),
pero reserva BDIM*(BDIM+1)*4 bytes de más por bloque y puede bajar la ocupancia.
-->

---

## **Ejemplo Transpuesta con memoria compartida**

Ejemplo: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu) (necesita [common.h](../code/memoria/common.h)).

4 cuatro *kernels*. El programa **mide e imprime el tiempo de cada uno**:

1. `transpuestaGlobal`: la transpuesta con memoria global.
2. `transpuestaComp`: memoria compartida **estática**.
3. `transpuestaCompDin`: memoria compartida **dinámica**.
4. `transpuestaCompPad`: memoria compartida estática con ***padding*** (lo vemos en la clase siguiente, al llegar a conflictos de bancos).

Referencia `transpuestaHost`: la transpuesta usando solo CPU.

<!-- NOTA — el programa usa N = 4096 y BDIM = 32, o sea bloques de 32x32 = 1024 threads. Los cuatro kernels son la misma transpuesta con distinta estrategia: transpuestaGlobal es la referencia sin memoria compartida; transpuestaComp y transpuestaCompDin son el MISMO algoritmo con declaración estática y dinámica, y deben dar tiempos casi iguales (sirve justamente para mostrar que la declaración dinámica no cuesta rendimiento, solo flexibilidad); transpuestaCompPad agrega el padding que se explica en "Conflictos de bancos". El ejercicio de esta clase compara los tres primeros; el cuarto queda para la clase siguiente. -->

<!-- MEDICIÓN DE REFERENCIA en la T4 (2026-09), N = 4096, bloques 32x32. Ancho de banda efectivo = 2*N*N*4 bytes / tiempo: transpuestaHost 0.43 s (0.3 GB/s) · transpuestaGlobal 1.547 ms (86.8 GB/s) · transpuestaComp 1.429 ms (93.9 GB/s) · transpuestaCompDin 1.438 ms (93.3 GB/s) · transpuestaCompPad 0.766 ms (175.2 GB/s). Los números pueden variar entre corridas, pero las RELACIONES se repiten. -->

<!-- Lo que hay que leer en esos números, y es más interesante de lo que parece: (1) estática y dinámica difieren en 0.6%, o sea son la misma cosa — declarar dinámico cuesta flexibilidad, no rendimiento. (2) La memoria compartida por sí sola gana apenas un 8% sobre la versión global (86.8 -> 93.9), aunque mueve MUCHO menos tráfico global. (3) El padding casi duplica: 1.87x sobre transpuestaComp y 2.02x sobre transpuestaGlobal, llegando a 175 GB/s, un 58% del peak de 300 GB/s de la T4. -->

<!-- POR QUÉ la memoria compartida sola casi no gana, que es la pregunta que va a salir: con bloques de 32x32 un warp es una fila (tx = 0..31, ty fijo). transpuestaGlobal pide 32 sectores en la carga dispersa más 4 en el store contiguo = 36 sectores por warp. transpuestaComp deja los DOS accesos globales contiguos: 4 + 4 = 8 sectores por warp, o sea 4.5 veces menos tráfico global. Y sin embargo solo gana 8%. El motivo es que el cuello de botella se mudó: leer tile[threadIdx.x][threadIdx.y] con tile[32][32] da banco = (tx*32+ty)%32 = ty para los 32 threads, un conflicto de 32 vías que serializa la lectura de memoria compartida en 32 transacciones. Con tile[32][33] el banco pasa a (tx+ty)%32, todos distintos, y vuelve a ser una sola transacción: ahí aparece el 1.87x. -->

<!-- Moraleja para decir en voz alta: la memoria compartida no es gratis ni automática. Mal usada, cambia un problema de coalescencia por uno de bancos y no gana nada. El ejemplo completo solo cierra con el padding de la clase siguiente. -->


---

# Organización de la memoria compartida (SMEM)

---

## **Acceso a la SMEM**

![w:1020px](images/memoria/figure_5_2.png)

<p class="credit">Acceso ideal — Fuente: <em>Professional CUDA C Programming</em></p>

- La Memoria Compartida se organiza en 32 *bancos*.
  - En principio, 1 para cada *thread* de un *warp*.
- El patrón de acceso a los bancos también influye en el rendimiento.
<!-- - **Regla de los bancos**: evitar que más de un *thread* acceda simultáneamente a un mismo elemento del banco. -->

---

## **Acceso a la memoria SMEM**

![w:1020px](images/memoria/figure_5_3.png)

<p class="credit">Acceso desordenado, pero no problemático — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Acceso a la SMEM**

![w:1020px](images/memoria/figure_5_4.png)

<p class="credit">Potencialmente problemático... — Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Organización de la SMEM (bancos)**

![w:800px](images/memoria/figure_5_5.png)

<p class="credit">Bancos de ancho 4-bytes — Fuente: <em>Professional CUDA C Programming</em></p>

- Cada banco entrega **4 bytes** (1 *palabra*) por ciclo, y segmentos consecutivos van a bancos consecutivos.
- $\text{banco} = (\text{dirección} / 4) \bmod 32$: la palabra $32$ está en el banco $0$.

<!-- NOTA — 32 bancos x 4 bytes = 128 bytes por ciclo. Un warp que lee 32 floats consecutivos toca cada banco una vez: una sola transacción. Hay conflicto cuando dos threads del warp piden palabras DISTINTAS del mismo banco (p. ej. las palabras 0 y 32); si piden la MISMA palabra hay broadcast y no hay conflicto. Adelanto para la clase siguiente: leer una columna de tile[32][32] es un stride de 32 palabras, así que los 32 threads caen en el mismo banco. Este es el modo de la T4 y de toda GPU desde Maxwell. -->

---

<!-- ## **Organización de la memoria compartida (bancos)** -->
<!---->
<!-- ![w:640px](images/memoria/figure_5_6.png) -->
<!---->
<!-- <p class="credit">Bancos de ancho 8-bytes — Fuente: <em>Professional CUDA C Programming</em></p> -->
<!---->
<!-- - **8 bytes** por banco y ciclo: $\text{banco} = (\text{dirección} / 8) \bmod 32$. -->
<!-- - Solo en Kepler (CC 3.x); la T4 usa siempre bancos de 4 bytes. -->

<!-- NOTA — la figura del libro no calza con su propia regla. Según la fórmula (y la guía de CUDA), el banco 0 contiene las palabras de 4 bytes 0 y 1 (un double completo), el banco 1 las palabras 2 y 3, etc.; la figura dibuja 0 y 32 en el banco 0. Además tiene una errata: el banco 30 dice 28/62 y debería decir 30/62. Si alguien lo nota, darle la razón: vale la fórmula. La ventaja del modo era doble: el doble de ancho de banda para datos de 8 bytes, y dos threads que leen las dos mitades del mismo double no generan conflicto. En GPUs actuales cudaDeviceSetSharedMemConfig no tiene efecto (está deprecada). -->

<!-- --- -->

<!-- ## **Ejercicio 1: ¿ayuda la memoria compartida?** -->
<!---->
<!-- Descargar: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu) y [common.h](../code/memoria/common.h) -->
<!---->
<!-- El programa reporta el tiempo de cada *kernel*. Por ahora nos interesan tres: -->
<!---->
<!-- - `transpuestaGlobal` · `transpuestaComp` · `transpuestaCompDin` -->
<!---->
<!-- 1. Ordenarlos de más lento a más rápido. ¿Cuánto se gana con memoria compartida? -->
<!-- 2. ¿La declaración dinámica cuesta más que la estática? -->
<!-- 3. ¿Qué pasaría si borráramos el `__syncthreads()`? -->
<!---->
<!-- <!-- RESPUESTAS medidas en la T4 (2026-09), N = 4096: transpuestaGlobal 86.8 GB/s, transpuestaComp 93.9 GB/s, transpuestaCompDin 93.3 GB/s. --> -->
<!---->
<!-- <!-- (1) La respuesta a "cuánto se gana" es INCÓMODA a propósito: apenas un 8%. Hay que dejar que los alumnos se lleven esa sorpresa, porque es la que motiva la clase siguiente. Si alguien pregunta por qué tan poco: los dos accesos globales quedaron contiguos (8 sectores por warp contra 36 del kernel global), pero el cuello de botella se mudó a la memoria compartida, donde leer el tile por columnas provoca un conflicto de bancos de 32 vías. El detalle está en la nota de la diapositiva del código. --> -->
<!---->
<!-- <!-- (2) No: 93.9 contra 93.3 GB/s, un 0.6% de diferencia, que es ruido. La declaración dinámica cuesta flexibilidad (hay que pasar el tamaño al lanzar, y solo arrays 1D), no rendimiento. --> -->
<!---->
<!-- <!-- (3) Condición de carrera: el thread (tx,ty) lee una casilla que escribió el thread (ty,tx). Sin la barrera el resultado es indefinido — puede "funcionar" y no hay que confiar en eso. Conviene que lo prueben: con bloques de 32x32 hay 32 warps por bloque y la corrupción se ve fácil. --> -->
<!---->
<!-- <!-- Cuando lleguen al ejercicio 2 de la clase siguiente, transpuestaCompPad da 175.2 GB/s: 1.87x sobre transpuestaComp. Ahí recién se cobra la memoria compartida. --> -->
<!---->
<!-- --- -->

<!-- ## **Ejercicio : cálculo de los índices a mano** -->
<!---->
<!-- Extender el ejemplo de la matriz transpuesta para una matriz de $8 \times 8$ con bloques de $4 \times 4$. -->
<!---->
<!-- Descargas: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu) y [common.h](../code/memoria/common.h) -->
<!---->
<!-- Para el *thread* `threadIdx = (1,2)` del bloque `blockIdx = (1,0)`, calcular: -->
<!---->
<!-- ```cuda -->
<!-- ix, iy      // índices globales -->
<!-- ti          // índice lineal de entrada -->
<!-- ixt, iyt    // índices tras el paso 1 -->
<!-- to          // índice lineal de salida -->
<!-- ``` -->
<!---->

<!-- RESPUESTAS — con N = 8, blockDim = (4,4), blockIdx = (1,0), threadIdx = (1,2), y las fórmulas del kernel transpuestaComp: ix = 4*1 + 1 = 5 · iy = 4*0 + 2 = 2 · ti = iy*N + ix = 2*8 + 5 = 21 · ixt = blockDim.y*blockIdx.y + threadIdx.x = 4*0 + 1 = 1 · iyt = blockDim.x*blockIdx.x + threadIdx.y = 4*1 + 2 = 6 · to = iyt*N + ixt = 6*8 + 1 = 49. -->

<!-- Qué significan: el thread CARGA entrada[21], el elemento (fila 2, columna 5), y lo deja en tile[2][1] (tile[threadIdx.y][threadIdx.x]). Después ESCRIBE salida[49], la posición (fila 6, columna 1). El paso 1 ya movió el bloque: el bloque (1,0) de la entrada escribe en el bloque (0,1) de la salida, y 6 y 1 caen justamente ahí (filas 4-7, columnas 0-3). -->

<!-- La pregunta que conviene hacer en voz alta: ¿qué valor escribe en salida[49]? No el que cargó, sino tile[1][2] (tile[threadIdx.x][threadIdx.y]), que cargó el thread vecino threadIdx = (2,1). Ese vecino tiene ix = 4 + 2 = 6, iy = 1, o sea cargó entrada[1*8 + 6] = entrada[14], el elemento (fila 1, columna 6). Y en efecto la transpuesta pide salida(6,1) = entrada(1,6). Cierre del círculo: el elemento que cargó nuestro thread, (2,5), lo escribe ese mismo vecino en salida(5,2) = salida[42]. Cada thread carga para un vecino y escribe lo que cargó otro: ese intercambio dentro del bloque es el paso 2, y por eso hace falta __syncthreads() entre la carga y la escritura. -->

<!-- Si alguien pregunta por qué leer el tile por columnas y no simplemente reusar lo que el cache trae de más (como en la transpuesta global): en la versión global, el reuso de los bytes sobrantes de cada sector es accidental y depende de que sigan en el cache; aquí nada se trae de más (las dos cuentas globales son contiguas) y el reuso lo garantizan el programador y la barrera. La lectura por columnas se hace en memoria compartida porque ahí un acceso disperso no desperdicia sectores: cuesta una transacción, salvo conflictos de bancos, que es el tema de la clase siguiente. -->

<!-- --- -->

# Conflictos de bancos

---

## **Conflictos de bancos**

<table class="bancos">
<tr><th>Banco 0</th><th>Banco 1</th><th>Banco 2</th><th>Banco 3</th><th class="gap">…</th><th>Banco 30</th><th>Banco 31</th></tr>
<tr><td class="ok">0<small>t0</small></td><td>1</td><td>2</td><td class="ok">3<small>t3</small></td><td class="gap">…</td><td>30</td><td class="ok">31<small>t31</small></td></tr>
<tr><td>32</td><td class="ok">33<small>t1</small></td><td>34</td><td>35</td><td class="gap">…</td><td>62</td><td>63</td></tr>
<tr><td>64</td><td>65</td><td class="ok">66<small>t2</small></td><td>67</td><td class="gap">…</td><td class="ok">94<small>t30</small></td><td>95</td></tr>
</table>


Cada *thread* cae en un banco distinto, aunque sea en otra fila: **1 transacción**.

<!-- NOTA — banco = palabra % 32: 0 -> 0, 33 -> 1, 66 -> 2, 3 -> 3, 94 -> 30, 31 -> 31. Todos distintos, así que la fila no importa: el warp se sirve en una sola transacción. Solo se dibujan algunos threads; la regla vale para los 32. -->

---

## **Conflictos de bancos**

<table class="bancos">
<tr><th>Banco 0</th><th>Banco 1</th><th>Banco 2</th><th>Banco 3</th><th class="gap">…</th><th>Banco 30</th><th>Banco 31</th></tr>
<tr><td class="ok">0<small>t0</small></td><td class="ok">1<small>t1 t2</small></td><td>2</td><td>3</td><td class="gap">…</td><td class="ok">30<small>t30 t31</small></td><td>31</td></tr>
<tr><td>32</td><td>33</td><td>34</td><td>35</td><td class="gap">…</td><td>62</td><td>63</td></tr>
<tr><td>64</td><td>65</td><td>66</td><td>67</td><td class="gap">…</td><td>94</td><td>95</td></tr>
</table>

Mismo banco y **misma palabra**. 
  - *broadcast* (no existe conflicto).

<!-- NOTA — t1 y t2 leen la MISMA palabra (1), igual que t30 y t31 (palabra 30). El banco entrega la palabra una vez y la reparte (broadcast): no hay conflicto. La figura original del libro mostraba aquí dos threads leyendo las dos mitades de una palabra de 8 bytes, un caso que solo existía en el modo de 8 bytes de Kepler; en la T4 el caso equivalente es este. Conflicto = palabras DISTINTAS en el mismo banco, no simplemente el mismo banco. -->

---

## **Conflictos de bancos**

<table class="bancos">
<tr><th>Banco 0</th><th>Banco 1</th><th>Banco 2</th><th>Banco 3</th><th class="gap">…</th><th>Banco 30</th><th>Banco 31</th></tr>
<tr><td class="ok">0<small>t0</small></td><td class="conflicto">1<small>t1</small></td><td>2</td><td>3</td><td class="gap">…</td><td class="ok">30<small>t30</small></td><td class="ok">31<small>t31</small></td></tr>
<tr><td>32</td><td class="conflicto">33<small>t2</small></td><td>34</td><td>35</td><td class="gap">…</td><td>62</td><td>63</td></tr>
<tr><td>64</td><td>65</td><td>66</td><td>67</td><td class="gap">…</td><td>94</td><td>95</td></tr>
</table>

Dos palabras **distintas** en el banco 1.
  - Conflicto de **2 vías**: 2 transacciones serializadas.

<!-- NOTA — 1 % 32 = 1 y 33 % 32 = 1: t1 y t2 piden palabras distintas del banco 1, y el banco entrega una por ciclo. El acceso se divide en 2 transacciones; el resto de los threads no tiene problema, pero el warp completo espera a la más lenta. -->

---

## **Conflictos de bancos**

<table class="bancos">
<tr><th>Banco 0</th><th>Banco 1</th><th>Banco 2</th><th>Banco 3</th><th class="gap">…</th><th>Banco 30</th><th>Banco 31</th></tr>
<tr><td>0</td><td class="conflicto">1<small>t0</small></td><td>2</td><td>3</td><td class="gap">…</td><td class="ok">30<small>t30</small></td><td class="ok">31<small>t31</small></td></tr>
<tr><td>32</td><td class="conflicto">33<small>t1</small></td><td>34</td><td>35</td><td class="gap">…</td><td>62</td><td>63</td></tr>
<tr><td>64</td><td class="conflicto">65<small>t2</small></td><td>66</td><td>67</td><td class="gap">…</td><td>94</td><td>95</td></tr>
</table>

Tres palabras distintas en el banco 1.
  - Conflicto de **3 vías**: 3 transacciones serializadas.

En general, podemos tener conflictos de **$N$ vías**, donde $2\leq N\leq 32$, lo cual puede reducir el rendimiento hasta en $1/N$.

<!-- NOTA — 1, 33 y 65 caen todas en el banco 1 (n % 32 = 1): conflicto de 3 vías. El caso extremo es un stride de 32 palabras, que pone a los 32 threads en el mismo banco: conflicto de 32 vías. Es exactamente lo que pasa al leer una columna de tile[32][32] en la transpuesta, dos diapositivas más adelante. -->

---

## **Solución: *padding***

<div class="lado-a-lado">
<table class="bancos compacto">
<tr><th>Banco 0</th><th>Banco 1</th><th>Banco 2</th><th>Banco 3</th><th>Banco 4</th><th>padding</th></tr>
<tr><td class="conflicto">0</td><td>1</td><td>2</td><td>3</td><td>4</td><td class="pad"></td></tr>
<tr><td class="conflicto">0</td><td>1</td><td>2</td><td>3</td><td>4</td><td class="pad"></td></tr>
<tr><td class="conflicto">0</td><td>1</td><td>2</td><td>3</td><td>4</td><td class="pad"></td></tr>
<tr><td class="conflicto">0</td><td>1</td><td>2</td><td>3</td><td>4</td><td class="pad"></td></tr>
<tr><td class="conflicto">0</td><td>1</td><td>2</td><td>3</td><td>4</td><td class="pad"></td></tr>
</table>
<table class="bancos compacto">
<tr><th>Banco 0</th><th>Banco 1</th><th>Banco 2</th><th>Banco 3</th><th>Banco 4</th></tr>
<tr><td class="ok">0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr>
<tr><td class="pad"></td><td class="ok">0</td><td>1</td><td>2</td><td>3</td></tr>
<tr><td>4</td><td class="pad"></td><td class="ok">0</td><td>1</td><td>2</td></tr>
<tr><td>3</td><td>4</td><td class="pad"></td><td class="ok">0</td><td>1</td></tr>
<tr><td>2</td><td>3</td><td>4</td><td class="pad"></td><td class="ok">0</td></tr>
<tr><td>1</td><td>2</td><td>3</td><td>4</td><td class="pad"></td></tr>
</table>
</div>

Una columna extra por fila desplaza cada fila un banco: la columna $0$ queda en bancos **distintos**.

<!-- NOTA — modelo de juguete del libro: 5 bancos y una matriz de 5x5, cada casilla marcada con su índice de columna. Sin padding (izquierda) una fila ocupa exactamente los 5 bancos, así que toda la columna 0 cae en el banco 0: leer una columna es un conflicto de 5 vías. Con una casilla extra por fila (derecha) cada fila empieza un banco más allá, y la columna 0 queda repartida en los bancos 0, 1, 2, 3, 4: una sola transacción. La memoria de más son las casillas grises, que nunca se leen. En la transpuesta real son 32 bancos y tile[32][33]: el elemento tile[tx][ty] está en el índice tx*33+ty, banco (tx*33+ty) % 32 = (tx+ty) % 32, distinto para cada thread del warp. Es la diapositiva siguiente. -->

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

- Esto es justamente lo que hizo más rápido el kernel `transpuestaCompPad` en el ejemplo del código [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu)
---

# Memoria constante
---

## **Memoria constante**

![w:460px](images/memoria/figure_4_2.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

---

## **Memoria constante**

- Reside en la memoria del *device* (igual que la memoria global).
  - Cada SM tiene un *cache* asignado a la memoria constante.
- Se declara con `__constant__`. Debe tener *global scope*, fuera de cualquier *kernel*. Hay $64$ KB disponibles.
- Útil para constantes matemáticas aplicadas por todos los *threads*.
- Los *kernels* solo pueden **leer** de la memoria constante, así que hay que inicializarla desde el *host*:

```cuda
cudaError_t cudaMemcpyToSymbol(const void* simbolo, const void* src, size_t count);
```

<!-- NOTA — conexión con lo que acabamos de ver: el cache de memoria constante sigue la misma regla que los bancos de memoria compartida. Si todo el warp pide la MISMA dirección, una sola lectura sirve a los 32 threads (broadcast, como la "misma palabra" en un banco). Si pide k direcciones distintas, la lectura se divide en k lecturas seriadas, como un conflicto de k vías. Por eso sirve para constantes que todos los threads leen a la vez, y no para tablas indexadas por thread. El ejemplo que sigue mide los dos casos. -->

---

<!-- ## **Memoria constante: ejemplo** -->

<!-- Ejemplo: [memoria_constante.cu](../code/memoria/memoria_constante.cu) (necesita -->
<!-- [common.h](../code/memoria/common.h)). -->
<!---->
<!-- Cada *thread* suma los $360$ ángulos de una tabla. El programa mide cuatro -->
<!-- versiones: -->
<!---->
<!-- - Tabla en memoria **global** o **constante**. -->
<!-- - Acceso **uniforme** (todo el *warp* lee el mismo ángulo) o **disperso** (cada -->
<!--   *thread* lee uno distinto). -->
<!---->
<!-- ```bash nvcc -arch=sm_75 memoria_constante.cu -o memoria_constante.x -->
<!-- ./memoria_constante.x ``` -->

<!-- NOTA — la memoria constante tiene su propio cache, que entrega UNA dirección por lectura a todo el warp (broadcast). Si el warp pide k direcciones distintas, la lectura se divide en k lecturas seriadas. El acceso uniforme es su mejor caso y el disperso el peor: 32 direcciones, 32 lecturas. La memoria global, en cambio, junta 32 floats consecutivos en un solo acceso coalescido y cacheado. Resultado esperado: con acceso uniforme la constante empata o gana; con acceso disperso pierde por mucho. En la T4 las lecturas uniformes de memoria global también quedan en el L1, así que la diferencia en el caso uniforme puede ser chica: eso también es parte de la lección. Los cuatro kernels acumulan en un registro y escriben una sola vez, para que lo único que cambie sea de dónde se leen los ángulos. -->

<!-- --- -->

# Transferencias entre *host* y *device*

<!-- NOTA — esta sección tiene un orden deliberado: primero la motivación (el
PCIe es lento), después dos herramientas. La memoria pinned hace más rápidas las
transferencias EXPLÍCITAS (cudaMemcpy); la memoria unificada las hace
IMPLÍCITAS. La pinned va primero porque la unificada se apoya en las mismas
ideas: páginas, fallos de página y memoria que el sistema operativo no puede
mover. -->

---

## **Transferencias de memoria**

![w:560px](images/memoria/figure_4_3.png)

<p class="credit">Ejemplo para Fermi C2050 GPU — Fuente: <em>Professional CUDA C Programming</em></p>

El bus PCIe es mucho más lento que la memoria del GPU: hay que transferir lo mínimo.

<!-- NOTA — la figura es de una Fermi (8 GB/s de PCIe contra 144 GB/s de GDDR5).
En la T4: PCIe 3.0 x16, unos 16 GB/s teóricos y unos 12 GB/s en la práctica,
contra 300 GB/s de GDDR6. La brecha es de unas 20 veces, y se agranda con cada
generación. Consecuencia práctica: un kernel rápido no sirve de nada si cada
lanzamiento va precedido de una copia grande. Hay que dejar los datos en el GPU
todo lo posible. Lo que sigue son dos herramientas para cuando SÍ hay que
transferir: memoria pinned (copias explícitas más rápidas) y memoria unificada
(copias implícitas). -->

---

## **Páginas de memoria**

- Cuando ejecutamos un programa, el sistema operativo (SO) está en control de administrar la memoria.
- La memoria que ve un programa es **virtual**: el SO la divide en bloques de tamaño fijo llamadas **páginas** (en Linux, $4$ KB).
- Una tabla de páginas traduce cada página virtual a un lugar físico en la RAM. 
  - Ese lugar **puede cambiar** en cualquier momento.
- Si falta RAM, el SO manda páginas al disco (*swap*) y las trae de vuelta cuando se tocan. 
  - Esto se denomina *page fault* (no es un error).

<!-- NOTA — la página es la unidad con que el sistema operativo administra la
memoria: no mueve bytes sueltos, mueve páginas completas. Para tener una escala:
un buffer de 64 MB son 16384 páginas de 4 KB. Existen también páginas grandes
(huge pages, 2 MB) para reducir el tamaño de la tabla. Lo importante para lo que
sigue es que la dirección virtual que usa el programa NO dice dónde están
físicamente los datos, y que ese lugar puede cambiar sin que el programa se
entere. Esta misma idea vuelve con la memoria unificada, que migra páginas entre
el host y el GPU. -->

---

## **Memoria paginable y *pinned***

- Por defecto la memoria del *host* es **paginable** (`malloc`).
  - El SO puede mover sus páginas o mandarlas al disco.
- Lo **opuesto** es la memoria ***pinned*** o *page-locked* (`cudaMallocHost`)
  - Las páginas quedan fijas en la RAM física.
- El GPU copia directo desde la RAM del host, y para eso necesita páginas que no se muevan. Desde memoria paginable, el *driver* primero copia a un buffer *pinned* propio.

<!-- NOTA — la copia por el PCIe la hace un motor DMA del GPU, que lee la RAM
del host directamente, sin pasar por el CPU. Para eso necesita que las páginas
no se muevan mientras copia, y el sistema operativo podría mover (o mandar al
disco) una página paginable en cualquier momento. Por eso el driver no copia
desde la memoria paginable: primero la copia a un buffer pinned propio y recién
desde ahí la manda por el PCIe. Una transferencia desde memoria paginable son en
realidad DOS copias. -->

---

## **Memoria paginable y *pinned***

![w:760px](images/memoria/figure_4_4.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

Paginable: **dos** copias. *Pinned*: **una**.

<!-- NOTA — izquierda: los datos están en memoria paginable, el driver los copia
primero a un buffer pinned (la flecha horizontal, una copia CPU a CPU) y después
el DMA los manda a la DRAM del GPU. Derecha: si los datos ya están en memoria
pinned, la primera copia desaparece. La ganancia depende del sistema; el ejemplo
que sigue la mide. -->

---

## **Memoria paginable y *pinned***

| | Paginable (`malloc`) | *Pinned* (`cudaMallocHost`) |
|---|---|---|
| Páginas | el SO las mueve | fijas en la RAM |
| Tamaño | limitado por SSD| limitada por la RAM física |
| Copia al GPU | dos copias | **una** copia |
| Costo | lo que está en disco es **lento**| le quita RAM al resto del sistema |

<!-- NOTA — ninguna es "mejor": son un compromiso. La paginable es flexible (el
programa puede pedir más memoria de la que hay, y el SO reparte la RAM entre
todos los procesos), pero paga una copia extra al transferir y es muy lenta si
sus páginas están en el disco. La pinned transfiere más rápido, pero cada byte
pinned es RAM que el resto del sistema pierde. Fijar las páginas tiene un
costo al asignar, pero en la T4 no se nota: ver el ejemplo. Consecuencias: pinned
solo para los buffers que se transfieren seguido. Si se pide más memoria pinned
de la que hay libre, cudaMallocHost falla; malloc, en cambio, casi nunca falla
al asignar (Linux reserva direcciones sin comprometer RAM hasta que se tocan).
Ojo en Colab: las máquinas virtuales normalmente no tienen swap, así que la
ventaja de "superar la RAM" es teórica ahí. -->


---

## **Asignación de memoria *pinned***

```cuda
cudaError_t cudaMallocHost(void **devPtr, size_t count);
cudaError_t cudaFreeHost(void *ptr);
```

- Usar demasiada memoria *pinned* puede afectar el rendimiento del sistema entero, ya que reduce la cantidad de memoria *paginable* disponible para el SO.
- Recomendación: priorizar datos que se transfieren entre *host* y *device* frecuentemente.

<!-- NOTA — el costo de la memoria pinned: es RAM que el sistema operativo ya no
puede mandar al disco ni reorganizar. Si se asigna demasiada, el resto del
sistema se queda sin memoria y empieza a paginar todo lo demás. Por eso no se
asigna TODO como pinned: solo los buffers que se transfieren seguido. Adelanto:
en el capítulo de kernels (clase 15), cudaMemcpyAsync con streams EXIGE memoria
pinned para poder solapar copias y cómputo. -->

---

## **Memoria *pinned*: ejemplo**

Ejemplo: [memoriaPinned.cu](../code/memoria/memoriaPinned.cu) (necesita [common.h](../code/memoria/common.h)).

El programa asigna el mismo buffer de $64$ MB como memoria **paginable** y como ***pinned***, y mide para cada uno el costo de asignarlo e inicializarlo, y la velocidad de las copias *host* → *device* y *device* → *host*.

```bash
nvcc -arch=sm_75 memoriaPinned.cu -o memoriaPinned.x
./memoriaPinned.x
```

<!-- NOTA — medido en la T4 (2026-09), una corrida, 64 MB y 10 copias: paginable 65.0
ms para asignar+iniciar, 4.77 GB/s host->device, 4.60 GB/s device->host; pinned
56.7 ms, 12.36 GB/s y 13.14 GB/s. Las copias: pinned es 2.6x más rápida hacia el
GPU y 2.9x de vuelta, y llega a los ~12-13 GB/s reales del PCIe 3.0 x16 de la
T4. La paginable se queda en menos de 5 GB/s porque cada copia son dos: primero
al buffer pinned del driver y recién después por el PCIe. La sorpresa es la
asignación: la pinned NO sale más cara (56.7 contra 65.0 ms). malloc es
lazy: solo reserva direcciones, y las 16384 páginas se asignan una por una
al inicializar, cada una con su fallo de página. cudaMallocHost fija todas las
páginas de una vez al asignar, así que la inicialización ya no paga fallos. El
costo de fijar existe, pero cambia de lugar y a este tamaño se compensa. El
verdadero costo de la memoria pinned no es el tiempo sino la RAM que el resto
del sistema pierde. Una sola corrida: los tiempos absolutos varían en Colab; las
razones son lo que importa. -->

---

## **Memoria unificada**

- Desde CUDA 6.0, *Unified Memory* (UM) permite acceder a la memoria usando un **solo espacio de direcciones** para el GPU y el CPU. 
  - UM se encarga de la transferencia de datos automáticamente.
- Basada en *Unified Virtual Addressing* (CUDA 4.0), que unificó el espacio de direcciones en memoria.

Esto simplifica la copia de datos entre *host* y *device* al programar al evitarnos usar `cudaMemcpy` entre ambos.

<!-- NOTA — la idea: un solo puntero que vale en el host y en el device, sin cudaMemcpy. Es una comodidad para programar, no magia: los bytes igual cruzan el PCIe, solo que ahora los mueve el driver en vez de nosotros. La diferencia con UVA (CUDA 4.0): UVA solo unificó las DIRECCIONES (un puntero dice si apunta al host o al device), pero no movía datos. UM además migra los datos al lado que los usa. -->

---

## **Memoria unificada**

- Declaración estática (a veces llamada *managed*): 

```cuda
__device__ __managed__ int y;
```

- Asignación dinámica:

```cuda
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=0);
```

El puntero `devPtr` es válido tanto en el *device* como en el *host*.

<!-- NOTA — __managed__ es la versión estática (una variable global, como __device__), cudaMallocManaged la dinámica (como cudaMalloc). Ambas se liberan igual que la memoria del device: cudaFree, no free. Detalle importante: después de lanzar un kernel, el host no debe tocar la memoria unificada hasta hacer cudaDeviceSynchronize(), porque el kernel es asíncrono y puede estar usándola. -->

---

## **Memoria unificada: el movimiento de datos**

- Los datos migran **por páginas y a pedido**.
  - Cuando el GPU toca una página que está en el *host*, ocurre un *page fault* y la página cruza el PCIe.
- Las páginas se quedan en el GPU hasta que el CPU las vuelve a tocar.
- El **primer acceso**  incurre en el costo del tiempo de la migración. 
- Para mitigarlo, se pueden mover los datos antes de lanzar el *kernel*:

```cuda
cudaMemPrefetchAsync(ptr, bytes, dispositivo);
```

<!-- NOTA — cada fallo de página detiene a los warps que tocaron esa página
hasta que llega por el PCIe. El driver agrupa los fallos y migra bloques de
páginas, pero aun así el primer kernel queda dominado por la migración, no por
el cálculo. cudaMemPrefetchAsync mueve todo de una vez, cerca de la velocidad
del PCIe, antes de lanzar. Antes de Pascal (Kepler, Maxwell) el GPU no tenía
fallos de página: el driver copiaba TODA la memoria unificada al GPU en cada
lanzamiento. La firma de cudaMemPrefetchAsync cambió en CUDA 13 (recibe un
cudaMemLocation en vez del número de dispositivo); el ejemplo compila con las
dos. -->

---

## **Memoria unificada: ejemplo**

Ejemplo: [memoria_unificada.cu](../code/memoria/memoria_unificada.cu) (necesita [common.h](../code/memoria/common.h)).

- `suma` hace `y = x + y` sobre dos arreglos de $64$ MB en memoria unificada, **sin ningún `cudaMemcpy`**. 
- El programa mide el mismo *kernel* cuatro veces.

```bash
nvcc -arch=sm_75 memoria_unificada.cu -o memoria_unificada.x
./memoria_unificada.x
nvprof ./memoria_unificada.x
```

<!-- NOTA — las cuatro mediciones: (1) el kernel con los datos recién inicializados en el host, así que paga los fallos de página; (2) el mismo kernel con los datos ya en el GPU; (3) cudaMemPrefetchAsync de x e y, después de volver a inicializarlos en el host (eso trae las páginas de vuelta al CPU); (4) el kernel después del prefetch. Lo esperado: (1) mucho mayor que (2), y la diferencia es la migración; (3) cerca de la velocidad del PCIe; (4) parecido a (2). nvprof agrega al final una sección "Unified Memory profiling result" con los bytes Host To Device, Device To Host y los grupos de fallos de página del GPU. Los tiempos medidos en la T4 están en la nota del ejercicio 3. -->

---

# Ejercicios

---

## **Ejercicio 1: conflictos de bancos y *padding***

Descargar: [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu) y [common.h](../code/memoria/common.h)

El programa mide los cuatro *kernels* de la transpuesta. Ahora nos interesa el cuarto: `transpuestaCompPad`.

1. `transpuestaComp` y `transpuestaCompPad` difieren en **un carácter**: `tile[BDIM][BDIM]` contra `tile[BDIM][BDIM+1]`. ¿Cuánto cambia el tiempo?
2. ¿Dónde queda `transpuestaCompPad` frente a los otros tres?

<!-- RESPUESTAS medidas en la T4 (2026-09), N = 4096: transpuestaCompPad 0.766 ms, 175.2 GB/s. (1) Un carácter da 1.87x sobre transpuestaComp (93.9 -> 175.2 GB/s) y 2.02x sobre transpuestaGlobal. (2) Pasa a ser el más rápido por lejos, y recién ahí la memoria compartida se paga: sin padding ganaba apenas 8% (86.8 -> 93.9 GB/s). -->

<!-- La cuenta de bancos: leer tile[threadIdx.x][threadIdx.y] con tile[32][32] pone el elemento en el índice tx*32+ty, o sea banco (tx*32+ty)%32 = ty, el MISMO para los 32 threads del warp: conflicto de 32 vías, 32 transacciones serializadas. Con tile[32][33] el índice pasa a tx*33+ty y el banco a (tx+ty)%32, distinto para cada thread: una sola transacción. La escritura tile[ty][tx] no tiene conflicto en ninguno de los dos casos. -->

<!-- Vale la pena cerrar con la perspectiva: 175 GB/s es un 58% del peak de 300 GB/s de la T4. Una transpuesta no llega al 100% porque los bloques recorren la DRAM de forma dispersa, pero pasar de 29% (global) a 58% cambiando un carácter es el mejor argumento del capítulo. -->

---

## **Ejercicio 1: métricas**

En `nvprof`:
- `shared_load_transactions_per_request`
- `shared_store_transactions_per_request`

En `ncu`:
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`

3. Medir los conflictos de bancos en `transpuestaComp` y en `transpuestaCompPad`. ¿El número explica la diferencia de tiempo?

---

## **Ejercicio 2: ¿sirve la memoria constante?**

Descargar: [memoria_constante.cu](../code/memoria/memoria_constante.cu) y [common.h](../code/memoria/common.h)

El programa mide cuatro *kernels* que hacen exactamente el mismo cálculo: `globalUniforme`, `constanteUniforme`, `globalDisperso` y `constanteDisperso`.

1. Ejecutar el programa y ordenar los cuatro *kernels* de más rápido a más lento.
2. ¿En qué caso gana la memoria constante? ¿En cuál pierde?

---

## **Ejercicio 2: ¿por qué?**

3. Con acceso uniforme todos los *threads* leen **el mismo** ángulo. ¿Por qué esto favorece a la memoria constante?
4. Con acceso disperso, ¿por qué la memoria global casi no se ve afectada y la constante sí?
5. El programa lanza cada *kernel* una vez antes de medir. ¿Para qué?

<!-- RESPUESTAS medidas en la T4 (2026-09), N = 2^20, bloques de 256, promedio de 20 lanzamientos, dos corridas en Colab. globalUniforme 0.935 / 0.383 ms · constanteUniforme 0.644 / 0.262 ms · globalDisperso 1.583 / 0.650 ms · constanteDisperso 16.365 / 12.124 ms. Los tiempos absolutos cambian mucho entre corridas (hasta 2.4x); el orden y las razones de los casos rápidos no. (1) De más rápido a más lento, en ambas corridas: constanteUniforme, globalUniforme, globalDisperso, constanteDisperso. (2) Con acceso uniforme la constante gana 1.45x sobre la global (1.45 y 1.46 en las dos corridas). Con acceso disperso pierde por lejos: entre 10x y 19x más lenta que la global dispersa, y entre 25x y 46x más lenta que ella misma con acceso uniforme, del orden de los 32x que predice serializar las 32 direcciones del warp. La global también empeora al dispersar (1.7x en ambas corridas), en parte por el módulo en cada iteración, pero nada comparable. -->

<!-- (3) El cache de memoria constante hace broadcast: una sola lectura sirve a los 32 threads del warp. Además libera al L1 y al ancho de banda global para otros datos. -->

<!-- (4) En el acceso disperso el warp pide 32 direcciones distintas. La memoria global las atiende como un acceso coalescido: son 32 floats consecutivos (módulo 360), 128 bytes, que además quedan en cache. La memoria constante, en cambio, sirve una dirección por vez: 32 lecturas seriadas. Regla: memoria constante solo para datos que todo el warp lee a la vez. -->

<!-- (5) El primer lanzamiento de un programa CUDA paga costos únicos: inicialización del contexto, carga del código del kernel, caches fríos. Si se midiera, el primer kernel saldría injustamente lento. Por eso cada kernel se lanza una vez sin medir (calentamiento) y después se promedian 20 lanzamientos. -->

<!-- (5, continuación) Un solo lanzamiento de calentamiento NO alcanza para calentar los relojes. En reposo la T4 baja a un estado de bajo consumo con relojes lentos, y tarda de decenas a cientos de milisegundos de trabajo continuo en llegar a su frecuencia máxima (boost). Cada medición de este programa dura apenas 5-20 ms, así que las primeras pueden hacerse con la GPU todavía acelerando. Eso explica las dos corridas: la segunda fue 2.4x más rápida en los tres kernels rápidos (la GPU ya venía despierta de las corridas anteriores), mientras que constanteDisperso cambió solo 1.35x porque se mide al final y dura unos 340 ms, suficiente para que el reloj suba. Las razones entre kernels medidos seguidos se mantienen (1.45x y 1.46x) porque ambos corren a un reloj parecido. -->

<!-- El efecto contrario también existe: la T4 es una tarjeta de 70 W sin ventilador propio, y bajo carga sostenida puede calentarse o tocar su límite de potencia y bajar el reloj (throttling). Eso calza con la transpuesta con conflictos de bancos, donde una corrida posterior salió MÁS lenta. Otras fuentes menores de ruido: el cronómetro corre en la CPU, que en Colab es virtual y compartida, y en intervalos de pocos milisegundos cualquier pausa pesa. -->

<!-- Todo esto es una hipótesis sin verificar. Para comprobarla en Colab, registrar el estado de la GPU cada 100 ms mientras corre el programa: nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv -lms 100 > relojes.csv & ; después ./memoria_constante.x dos veces y kill %1. Si la hipótesis es correcta, clocks.sm parte bajo y sube durante la primera corrida. Moraleja para los alumnos: calentar de verdad antes de medir, repetir la medición, y comparar razones, no tiempos absolutos. -->


---

## **Ejercicio 3: memoria unificada**

Descargar: [memoria_unificada.cu](../code/memoria/memoria_unificada.cu) y [common.h](../code/memoria/common.h)

1. ¿Por qué el primer lanzamiento tarda tanto más que el segundo, si el *kernel* es el mismo?
2. ¿Cuántos GB/s logra `cudaMemPrefetchAsync`? Compararlo con el PCIe de la T4.
3. En `nvprof`, ¿cuántos bytes *Host To Device* y cuántos *GPU page faults* aparecen?
4. ¿Cuándo conviene la memoria unificada y cuándo `cudaMemcpy` explícito?

<!-- RESPUESTAS medidas en la T4 (2026-09), varias corridas, dos arreglos de 64 MB. Corrida típica: (1) primer acceso con fallos de página 62.257 ms · (2) datos ya en el GPU 0.802 ms · (3) cudaMemPrefetchAsync de x e y 12.154 ms, 11.0 GB/s · (4) después del prefetch 0.806 ms. Algunas corridas dan (1) = 34.366 ms, con (2) 0.805, (3) 13.097 ms / 10.2 GB/s y (4) 0.800. Error máximo 0 siempre. Lo que se mantiene: el kernel con los datos en el GPU tarda 0.80 ms en todas las corridas (192 MB en 0.80 ms, unos 250 GB/s, el 84% del peak de 300 GB/s), y el prefetch rinde 10-11 GB/s, cerca de los ~12 GB/s reales del PCIe 3.0 x16. Lo que varía es SOLO la migración por fallos de página: entre 34 y 62 ms, o sea entre 4.0 y 2.2 GB/s efectivos, y entre 43 y 78 veces el kernel residente. Aun en la mejor corrida, prefetch más kernel (13.9 ms) es 2.5 veces más rápido que dejar que el kernel migre los datos (34.4 ms); en la típica, casi 5 veces. -->

<!-- Por qué varía la migración por fallos y no el resto (hipótesis, sin verificar): los fallos de página del GPU los atiende el driver en el CPU, y en Colab el CPU es virtual y compartido; además el driver agrupa fallos y adelanta páginas vecinas con heurísticas que dependen de cómo llegan los fallos, y eso cambia de corrida en corrida. El prefetch, en cambio, es una sola copia grande por el DMA, y el kernel residente solo depende de la DRAM del GPU. Moraleja: la migración a pedido no solo es más lenta, también es IMPREDECIBLE. Con prefetch el tiempo es menor y además estable. -->

<!-- (1) El kernel es el mismo, pero en el primer lanzamiento los 128 MB están en el host: cada página que el GPU toca provoca un fallo de página y tiene que cruzar el PCIe mientras el kernel espera. En el segundo lanzamiento los datos ya están en el GPU y solo se mide el cálculo, que para una suma es tráfico de DRAM a unos 300 GB/s. -->

<!-- (2) El prefetch mueve 128 MB de una vez; debería acercarse a la velocidad real del PCIe 3.0 x16 de la T4, unos 12 GB/s (16 teóricos). Mucho más rápido que migrar por fallos de página, porque no hay que detener warps ni atender fallos uno por uno. -->

<!-- (3) Host To Device deberían ser unos 128 MB por cada migración completa de x e y (dos veces en el programa: la del primer lanzamiento y la del prefetch), y Device To Host unos 192 MB: 128 cuando la segunda inicialización trae x e y de vuelta al CPU (el driver no sabe que se van a sobrescribir), y 64 más cuando el host lee y para verificar. Los fallos de página aparecen como grupos, no uno por página. -->

<!-- (4) Memoria unificada: prototipos, estructuras de datos con punteros (listas, árboles) que serían un infierno de copiar a mano, y datos que no caben en el GPU (la unificada puede sobrepasar la memoria del device). cudaMemcpy explícito, o unificada con prefetch: cuando importa el rendimiento y se sabe de antemano qué datos necesita cada kernel. Lo que NUNCA conviene es dejar que un kernel crítico pague la migración por fallos de página. -->

---

# Fin Capítulo 2

## Próximo capítulo: control de los threads
