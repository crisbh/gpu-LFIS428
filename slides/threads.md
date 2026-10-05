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

<!-- ## **Códigos** -->
<!---->
<!-- Los códigos de esta clase están disponibles para descargar: -->
<!---->
<!-- - [cuda_thread_block.cu](../code/threads/cuda_thread_block.cu), [matriz_mult.cu](../code/threads/matriz_mult.cu) -->
<!-- - Reducción: [reduccion_global.cu](../code/threads/reduccion_global.cu) … [reduccion_global8.cu](../code/threads/reduccion_global8.cu), [reduccion_compartida.cu](../code/threads/reduccion_compartida.cu) -->
<!-- - Reducción completa y atómicas: [reduccion-shuffle.cu](../code/threads/reduccion-shuffle.cu), [histograma.cu](../code/threads/histograma.cu) -->
<!-- - [grid_stride.cu](../code/threads/grid_stride.cu) -->
<!-- - [warp_shuffle_down.cu](../code/threads/warp_shuffle_down.cu), [warp_shuffle_up.cu](../code/threads/warp_shuffle_up.cu), [warp_shuffle_xor.cu](../code/threads/warp_shuffle_xor.cu) -->
<!-- - [gpu_suma_error.py](../code/threads/gpu_suma_error.py), [gpu_producto_punto_error.py](../code/threads/gpu_producto_punto_error.py) -->
<!---->
<!-- <!-- NOTA — todos los programas de esta clase son autocontenidos (no necesitan -->
<!-- common.h) y se compilan con nvcc -arch=sm_75. Los de la reducción imprimen el -->
<!-- resultado del host y del GPU para comparar; reduccion-shuffle.cu e -->
<!-- histograma.cu además miden su tiempo con eventos de CUDA, que se ven en la -->
<!-- clase de streams. Los dos .py usan PyCUDA, que veremos en la clase de -->
<!-- librerías. --> -->
<!---->
<!-- --- -->

## **Organización y sincronización de los threads**

| Nivel | Se compone de | Sincronización |
|---|---|---|
| *Grid* | bloques | no hay entre bloques (solo al terminar el *kernel*) |
| Bloque | *warps* (máx. $1024$ *threads*) | `__syncthreads()` |
| *Warp* | $32$ *threads* | `__syncwarp()`: **no** es implícita desde Volta |

<!-- NOTA — antes de Volta (CC 7.0) los 32 threads de un warp avanzaban juntos
(lock-step), y mucho código, incluido el material original de este curso,
asumía una sincronización implícita dentro del warp. Desde Volta (la T4 es
Turing, CC 7.5) cada thread tiene su propio contador de programa (independent
thread scheduling): el warp puede separarse y volver a juntarse en cualquier
punto. El código que asume lock-step puede fallar; hay que usar __syncwarp() o
las primitivas *_sync. Ya lo dijimos en la introducción; en esta clase lo
usamos en la reducción. -->

---

## **Modelo SIMT**

**SIMT** (*single instruction, multiple threads*) está entre **SIMD** (vectorización, AVX) y **SMT** (*threads* independientes, OpenMP):

- Escribimos el código de **un** *thread*; c/u tiene sus propios registros.
- Cada *thread* puede acceder a cualquier dirección (pero el acceso contiguo sigue siendo mejor).
- Cada *thread* puede tomar su propio camino en un `if`/`else` (a costa de **divergencia**).

En flexibilidad: **SIMD < SIMT < SMT**.

<!-- NOTA — la idea a transmitir: en SIMD el programador escribe operaciones sobre
vectores; en SIMT escribe el código de UN thread escalar y el hardware lo
ejecuta para 32 a la vez (un warp). Por eso CUDA se siente como programar
threads, pero rinde como vectorización mientras los 32 threads hagan lo mismo.
La flexibilidad tiene precio: acceso disperso (capítulo de memoria) y ramas
divergentes (más adelante en esta clase) cuestan rendimiento. -->

---

## **SIMT: desventajas**

- Con pocos *warps* activos (**occupancy** baja) no hay suficiente concurrency para esconder la latencia.
- La **divergencia de *warps*** serializa las ramas de un `if`.
- CUDA 9 agregó control fino: *warp primitives* (`__shfl_down_sync`, `__syncwarp`, ...) y *cooperative groups*.

<!-- NOTA — estos dos costos son la agenda de la clase: occupancy viene ahora,
divergencia dentro de la reducción. Las warp primitives de CUDA 9 también
aparecen en la reducción, como la forma correcta de trabajar dentro de un warp
desde Volta. Cooperative groups queda fuera del curso. -->

---

## **Diseño del GPU**

*Throughput* alto, *latency* alto:
- Los **CPUs**  optimizan *latency* (tiempo de espera).
- Los **GPUs**  optimizan *throughput* (cantidad de datos procesados).

Cada *core* de un GPU es mucho más lento que un *core* de un CPU... 

Pero hay miles de *cores*, así que un GPU puede usar miles de *threads* e intercambiar entre ellos (*context switching*) mucho más rápido que un CPU.

<!-- NOTA — conviene conectar con el roofline del capítulo 1: un acceso a DRAM
tarda cientos de ciclos (300-600 en la T4, ver la jerarquía de memoria), y el
GPU no intenta reducir esa latencia como un CPU con caches grandes y ejecución
especulativa. La esconde: mientras un warp espera su dato, el SM ejecuta otro.
Para eso necesita muchos warps listos, y eso es exactamente lo que mide el
occupancy. Las dos diapositivas que siguen lo muestran como una línea de
tiempo. -->

---

## **Latencia: ejemplo con un solo warp**

```cuda
float a = x[i];     // L: cargar de memoria global (latencia alta)
y[i] = 2.0f * a;    // C: calcular y guardar (necesita a)
```

Asumamos que una carga tarda $4$ unidades de tiempo (ciclos del reloj) en llegar, y el  *streaming multiprocessor* (SM) emite **una** instrucción por ciclo.

<table class="timeline">
<tr><th></th><th class="t">t0</th><th class="t">t1</th><th class="t">t2</th><th class="t">t3</th><th class="t">t4</th></tr>
<tr><th>Warp 0</th><td class="s1">L</td><td class="espera"></td><td class="espera"></td><td class="espera"></td><td class="s1">C</td></tr>
<tr><th>SM emite</th><td class="s1">W0</td><td>—</td><td>—</td><td>—</td><td class="s1">W0</td></tr>
</table>

El SM está esperando (*idle*) de $3$ de $5$ ciclos: el *warp* pasa casi todo el tiempo esperando.

<!-- NOTA — el modelo es de juguete a propósito: en la T4 un acceso a DRAM tarda
unos 300-600 ciclos (diapositiva de la jerarquía en el capítulo de memoria),
no 4. Con un solo warp, el SM estaría ocioso casi todo el tiempo: emite la
carga, y después no tiene nada que hacer hasta que llega el dato, porque la
instrucción siguiente depende de él. Las casillas rayadas son ese tiempo de
espera. -->


---

## **Esconder la latencia: varios warps**

```cuda
float a = x[i];     // L: cargar de memoria global (latencia alta)
y[i] = 2.0f * a;    // C: calcular y guardar (necesita a)
```

Ahora tenemos 4 *warps* para ejecutar las mismas tareas.

<table class="timeline">
<tr><th></th><th class="t">t0</th><th class="t">t1</th><th class="t">t2</th><th class="t">t3</th><th class="t">t4</th><th class="t">t5</th><th class="t">t6</th><th class="t">t7</th></tr>
<tr><th>Warp 0</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><th>Warp 1</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><th>Warp 2</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><th>Warp 3</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><th>SM emite</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</table>

**¿Como reparte el trabajo el SM para mantener el sistema ocupado?**

<!-- NOTA — dejar que la clase proponga antes de pasar a la diapositiva siguiente;
se puede llenar la tabla en la pizarra con sus ideas. Las respuestas que
buscamos son dos. (1) Más warps: mientras uno espera su carga, el SM emite la
instrucción de otro (paralelismo entre threads, TLP); es lo que muestran las
dos diapositivas que siguen, y lleva al occupancy. (2) Más trabajo
independiente dentro de cada thread: si cada thread carga dos elementos antes
de usar el primero (por ejemplo x[i] y x[i + N/2]), las dos cargas están en
vuelo a la vez (paralelismo de instrucciones, ILP); es lo que hace el loop
unrolling en la reducción. Las dos se combinan. Respuestas que conviene
descartar: hacer más rápido un solo thread no sirve, porque lo que domina es
la espera de la memoria, no el cálculo; y reordenar las dos instrucciones no
se puede, porque C depende de L. -->

---

## **Esconder la latencia: varios warps**

<table class="timeline">
<tr><th></th><th class="t">t0</th><th class="t">t1</th><th class="t">t2</th><th class="t">t3</th><th class="t">t4</th><th class="t">t5</th><th class="t">t6</th><th class="t">t7</th></tr>
<tr><th>Warp 0</th><td class="s1">L</td><td class="espera"></td><td class="espera"></td><td class="espera"></td><td class="s1">C</td><td></td><td></td><td></td></tr>
<tr><th>Warp 1</th><td></td><td class="s2">L</td><td class="espera"></td><td class="espera"></td><td class="espera"></td><td class="s2">C</td><td></td><td></td></tr>
<tr><th>Warp 2</th><td></td><td></td><td class="s3">L</td><td class="espera"></td><td class="espera"></td><td class="espera"></td><td class="s3">C</td><td></td></tr>
<tr><th>Warp 3</th><td></td><td></td><td></td><td class="s4">L</td><td class="espera"></td><td class="espera"></td><td class="espera"></td><td class="s4">C</td></tr>
<tr><th>SM emite</th><td class="s1">W0</td><td class="s2">W1</td><td class="s3">W2</td><td class="s4">W3</td><td class="s1">W0</td><td class="s2">W1</td><td class="s3">W2</td><td class="s4">W3</td></tr>
</table>

- Mientras un *warp* espera a que lleguen sus datos, el SM emite una instrucción **independiente** a otro *warp*.
- $4$ *warps* terminan en $8$ ciclos, en vez de $4 \times 5 = 20$ (en serie).
  - El SM siempre tiene algo que hacer.
- La latencia **sigue presente**, pero queda **escondida**.

<!-- NOTA — la regla general: para esconder una latencia de L ciclos hay que tener
suficiente trabajo independiente en vuelo; es la ley de Little aplicada al
GPU, bytes en vuelo = latencia x ancho de banda. Estimación de orden de
magnitud para la T4: 300 GB/s x unos 250 ns (unos 400 ciclos a 1.59 GHz) dan
unos 75 KB en vuelo, repartidos entre 40 SMs, unos 1.9 KB por SM, o sea unos
15 warps por SM si cada thread carga un float (128 B por warp). Por eso un SM
mantiene hasta 32 warps residentes, y por eso la sección siguiente mide
cuántos hay: eso es el occupancy. La otra forma de tener trabajo independiente
es dentro de un mismo warp (ILP): varias cargas independientes antes de usar
cualquiera de ellas, que es lo que hace el loop unrolling en la reducción. Y
explica la diapositiva anterior: el GPU no reduce la latencia, la esconde. -->

---

## **Threads, warps, blocks**

Ejemplo: [cuda_thread_block.cu](../code/threads/cuda_thread_block.cu).

- Bloques y *warps* se ejecutan en cualquier orden: lo que imprimen no está ordenado.
- Dentro de un *warp* los *threads* **suelen**  imprimir en orden, pero desde Volta **no está garantizado**.
 - **No**  hay que programar un código asumiendo *lock-step*.

<!-- NOTA — el programa recibe el tamaño del grid y del bloque como argumentos
(./cuda_thread_block.x 4 128) e imprime thread, bloque, warp y lane para
algunos threads. Vale la pena ejecutarlo varias veces: el orden entre bloques y
entre warps cambia. Que dentro de un warp salga en orden es un detalle de
implementación de printf, no una garantía; desde Volta los threads de un warp
pueden avanzar por separado. -->

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

<!-- NOTA — warps activos son los que un SM tiene asignados al mismo tiempo, no los
que están ejecutando en un ciclo dado. El máximo es un límite del hardware (32
warps por SM en la T4). Los recursos se reservan por bloque completo: si a un
bloque le falta un registro, no entra ninguno de sus warps. Por eso los tres
límites que siguen (registros, memoria compartida y tamaño del bloque) se
calculan en bloques enteros. -->

---

## **Occupancy: límites de la T4**

| Recurso por SM | T4 (CC $7.5$) |
|---|---|
| *Threads* activos | $1024$ ($32$ *warps*) |
| Bloques activos | $16$ |
| Registros | $65\,536$ |
| Memoria compartida | hasta $64$ KB |

<!-- NOTA — el material original usaba la arquitectura Fermi del libro (32K
registros, 48 warps y 8 bloques por SM, 16 o 48 KB de memoria compartida).
Los números de esta tabla son los de la T4 que usan los alumnos, y se pueden
consultar con cudaGetDeviceProperties: maxThreadsPerMultiProcessor,
maxBlocksPerMultiProcessor, regsPerMultiprocessor y
sharedMemPerMultiprocessor. El occupancy real es el MÍNIMO de lo que permite
cada límite por separado. -->

---

## **Occupancy: registros**

Se puede obtener el uso de registros del compilador con `--resource-usage`.

**Ejemplo A**: $64$ registros por *thread*: $65\,536/64 = 1024$ *threads*, *occupancy* $= 1$.

**Ejemplo B**: $128$ registros por *thread*: $65\,536/128 = 512$ *threads*, *occupancy* $= 512/1024 = 0.5$.

El uso de registros se puede controlar con `--maxrregcount`.

<!-- NOTA — los registros se asignan por warp en unidades de 256, así que el
número real puede ser un poco menor que la división exacta; para el
razonamiento basta la cuenta simple. -->

---

## **Occupancy: memoria compartida**

Con bloques de $256$ *threads* y $64$ KB de memoria compartida por SM:

**Ejemplo A**: $16$ KB por bloque: caben $4$ bloques, $1024$ *threads*, *occupancy* $= 1$.

**Ejemplo B**: $32$ KB por bloque: caben $2$ bloques, $512$ *threads*, *occupancy* $= 0.5$.

<!-- NOTA — conexión con el capítulo de memoria: la memoria compartida es rápida
pero escasa, y cada bloque reserva la suya completa. Un tile más grande puede
mejorar el reuso pero bajar el occupancy. Ojo con el ejemplo de padding:
tile[32][33] usa 4224 bytes en vez de 4096, un detalle que acá no cambia nada
pero que en kernels con mucha memoria compartida sí puede costar un bloque por
SM. -->

---

## **Occupancy: tamaño del bloque**

Máximo $16$ bloques y $1024$ *threads* por SM:

| Tamaño bloque | Bloques por SM | *Threads* activos | *Occupancy* |
|:---:|:---:|:---:|:---:|
| 32  | 16 | 512  | 0.5 |
| 64  | 16 | 1024 | 1 |
| 128 | 8  | 1024 | 1 |
| 256 | 4  | 1024 | 1 |
| 1024 | 1 | 1024 | 1 |

**Ojo:** Bloques de menos de $64$ *threads* no alcanzan a llenar el SM.

<!-- NOTA — el caso de 32 threads es el que más sorprende: el bloque es un warp
completo, pero como solo caben 16 bloques por SM, quedan 16 warps de 32
posibles. Por eso la recomendación habitual es bloques de 128 a 256 threads:
llenan el SM sin quedar limitados por el número de bloques, y dejan margen
para que varios bloques convivan. El ejercicio de matriz_mult.cu, con bloques
de 16 threads, cae en el caso peor que el de esta tabla. -->

---

## **¿Cuándo optimizar el occupancy?**

- Si el *kernel* está limitado por el *bandwidth* (*bandwidth bound*).
- Si el *bandwidth* logrado es mucho menor que el *peak*.

Recordar el **modelo roofline** (Capítulo 1): un *kernel* con $AI$ bajo cae en la región *memory bound*, y su techo es $AI \times$ ancho de banda.

Más *warps* activos $\Rightarrow$ más accesos a memoria $\Rightarrow$ se oculta la latencia y el *kernel* se acerca a ese techo.

Si el *kernel* ya alcanza el techo, o si es *compute bound*, subir el *occupancy* **no ayuda**.

<!-- NOTA — la idea clave: occupancy no es una meta en sí. Sirve para esconder
latencia de memoria, así que importa en kernels memory bound que están lejos
del techo de ancho de banda. Un kernel compute bound, o uno que ya llega al
techo, no gana nada con más warps. Hay kernels muy rápidos con occupancy baja,
que esconden la latencia con paralelismo dentro de cada thread (varias cargas
independientes en vuelo), que es justamente lo que hace el loop unrolling que
viene en la reducción. -->

---

## **Occupancy: ejemplo**

Ejemplo: [matriz_mult.cu](../code/threads/matriz_mult.cu) (multiplicación de matrices).

- Compilamos con `--resource-usage`.
- Se puede medir el *occupancy* con el *profiler*:
  - `nvprof`: `achieved_occupancy`
  - `ncu`: `sm__warps_active.avg.pct_of_peak_sustained_active`

<!-- NOTA — --resource-usage imprime, por cada kernel, los registros por thread y
la memoria compartida y constante que usa. achieved_occupancy es el occupancy
medido (promedio de warps activos durante la ejecución), que puede ser menor
que el teórico, por ejemplo cuando el último grupo de bloques no llena todos
los SMs. El ejercicio al final de esta sección usa este mismo programa. -->

---

## **Occupancy: control**

Para controlar el *occupancy*:
- `__launch_bounds__(max_threads_por_bloque, <min_bloques_por_sm>)` en la definición del *kernel* (el segundo argumento es opcional).
- `--maxrregcount` en el compilador, para limitar el número de registros ocupados.

Si el compilador no puede satisfacer las restricciones, usará memoria fuera de los registros (*register spill*).
 - Esto no es deseable ya que reduce el rendimiento. La idea es evitarlo.

<!-- NOTA — __launch_bounds__ le dice al compilador para qué tamaño de bloque
optimizar: con esa garantía puede repartir los registros sabiendo cuántos
threads van a convivir. --maxrregcount es más bruto y aplica a todo el
archivo. En los dos casos, si el kernel necesita más registros que los
permitidos, el compilador manda variables a memoria local (register spill),
que vive en la memoria global y es lenta: puede salir más caro que el
occupancy que se ganó. -->

---

## **Ejercicio: occupancy de `matriz_mult.cu`**

Descargar: [matriz_mult.cu](../code/threads/matriz_mult.cu). Usa bloques de $4 \times 4 = 16$ *threads*.

1. Con los límites de la T4, ¿cuál es el *occupancy* teórico? (`--resource-usage` da los registros).
2. Medirlo con el *profiler*.
3. Cambiar a bloques de $16 \times 16$ y repetir.

```bash
nvcc -arch=sm_75 --resource-usage matriz_mult.cu -o matriz_mult.x
nvprof --metrics achieved_occupancy ./matriz_mult.x
```

<!-- RESPUESTAS — (1) Con 16 threads por bloque el límite que manda es el de
bloques: 16 bloques x 16 threads = 256 threads activos de 1024 posibles, o sea
occupancy teórico 0.25. Los registros no limitan: el kernel usa muy pocos (ver
la salida de --resource-usage). (2) achieved_occupancy debería salir cerca de
0.25 o menos. (3) Con 16x16 = 256 threads, caben 4 bloques: occupancy teórico
1. Ojo con el tiempo: la matriz es de 256x256, así que el kernel dura
microsegundos y la diferencia puede quedar tapada por el costo del lanzamiento;
para verla conviene subir N (múltiplo del bloque). TODO: medir en la T4. -->

---

# Reducción: una aplicación de la optimización

---

## **Reducción paralela**

- Recordar el concepto de reducción visto en el curso de Programación Paralela.
- Significa obtener un solo valor de un conjunto de datos, en forma paralela.

Un ejemplo:
Calcular la sumatoria de todos los elementos en un *array* utilizando $N$ *threads*.

<!-- NOTA — la reducción es un buen caso de estudio porque es simple de entender y
casi todo lo que cuesta es memoria: una suma por cada 4 bytes leídos, o sea la
intensidad aritmética más baja posible. Todas las optimizaciones que siguen
apuntan a lo mismo: leer cada elemento una vez, de forma contigua, y
desperdiciar la menor cantidad de threads. -->

---

## **Reducción paralela**

<svg class="arbol" viewBox="0 0 960 310" width="860" style="display:block;margin:0.4em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="flecha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
<line x1="60" y1="50" x2="105" y2="93" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="150" y1="50" x2="105" y2="93" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="240" y1="50" x2="285" y2="93" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="330" y1="50" x2="285" y2="93" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="420" y1="50" x2="465" y2="93" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="510" y1="50" x2="465" y2="93" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="600" y1="50" x2="645" y2="93" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="690" y1="50" x2="645" y2="93" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="105" y1="135" x2="195" y2="178" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="285" y1="135" x2="195" y2="178" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="465" y1="135" x2="555" y2="178" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="645" y1="135" x2="555" y2="178" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="195" y1="220" x2="375" y2="263" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<line x1="555" y1="220" x2="375" y2="263" stroke="#555" stroke-width="2" marker-end="url(#flecha)"/>
<rect x="28" y="10" width="64" height="40" rx="4" fill="#9e9e9e"/>
<text x="60" y="38" font-size="24" fill="#fff" text-anchor="middle">13</text>
<rect x="118" y="10" width="64" height="40" rx="4" fill="#9e9e9e"/>
<text x="150" y="38" font-size="24" fill="#fff" text-anchor="middle">27</text>
<rect x="208" y="10" width="64" height="40" rx="4" fill="#9e9e9e"/>
<text x="240" y="38" font-size="24" fill="#fff" text-anchor="middle">15</text>
<rect x="298" y="10" width="64" height="40" rx="4" fill="#9e9e9e"/>
<text x="330" y="38" font-size="24" fill="#fff" text-anchor="middle">14</text>
<rect x="388" y="10" width="64" height="40" rx="4" fill="#9e9e9e"/>
<text x="420" y="38" font-size="24" fill="#fff" text-anchor="middle">33</text>
<rect x="478" y="10" width="64" height="40" rx="4" fill="#9e9e9e"/>
<text x="510" y="38" font-size="24" fill="#fff" text-anchor="middle">2</text>
<rect x="568" y="10" width="64" height="40" rx="4" fill="#9e9e9e"/>
<text x="600" y="38" font-size="24" fill="#fff" text-anchor="middle">24</text>
<rect x="658" y="10" width="64" height="40" rx="4" fill="#9e9e9e"/>
<text x="690" y="38" font-size="24" fill="#fff" text-anchor="middle">6</text>
<rect x="73" y="95" width="64" height="40" rx="4" fill="#76b900"/>
<text x="105" y="123" font-size="24" fill="#fff" text-anchor="middle">40</text>
<rect x="253" y="95" width="64" height="40" rx="4" fill="#76b900"/>
<text x="285" y="123" font-size="24" fill="#fff" text-anchor="middle">29</text>
<rect x="433" y="95" width="64" height="40" rx="4" fill="#76b900"/>
<text x="465" y="123" font-size="24" fill="#fff" text-anchor="middle">35</text>
<rect x="613" y="95" width="64" height="40" rx="4" fill="#76b900"/>
<text x="645" y="123" font-size="24" fill="#fff" text-anchor="middle">30</text>
<rect x="163" y="180" width="64" height="40" rx="4" fill="#76b900"/>
<text x="195" y="208" font-size="24" fill="#fff" text-anchor="middle">69</text>
<rect x="523" y="180" width="64" height="40" rx="4" fill="#76b900"/>
<text x="555" y="208" font-size="24" fill="#fff" text-anchor="middle">65</text>
<rect x="343" y="265" width="64" height="40" rx="4" fill="#76b900"/>
<text x="375" y="293" font-size="24" fill="#fff" text-anchor="middle">134</text>
<text x="760" y="38" font-size="22" fill="#455a64" font-style="italic">datos</text>
<text x="760" y="123" font-size="22" fill="#455a64" font-style="italic">paso 1: 4 sumas</text>
<text x="760" y="208" font-size="22" fill="#455a64" font-style="italic">paso 2: 2 sumas</text>
<text x="760" y="293" font-size="22" fill="#455a64" font-style="italic">paso 3: 1 suma</text>
</svg>

Algoritmo de *árbol*:
 - Dividimos el cálculo en *ramas* que calculan sumas parciales.
 - En cada paso, juntamos las sumas parciales (sincronizamos).

<!-- NOTA — el árbol de la figura: en cada nivel la mitad de los elementos se suma
con la otra mitad, así que para N elementos hacen falta log2(N) niveles.
Secuencial son N-1 sumas en N-1 pasos; en paralelo son las mismas N-1 sumas,
pero en log2(N) pasos. -->

<!-- NOTA — versión redibujada (SVG) de la figura anterior, con los mismos
valores. Las sumas están verificadas: 13+27=40, 15+14=29, 33+2=35, 24+6=30,
40+29=69, 35+30=65 y 69+65=134. En cada paso se suman PARES VECINOS, así que
la cantidad de valores se reduce a la mitad: con N = 8 hacen falta log2(8) = 3
pasos (4, 2 y 1 sumas). Secuencial son las mismas N-1 = 7 sumas, pero en 7
pasos. El patrón de pares vecinos es el de las primeras versiones del código
(reduccion_global, global2); las versiones con acceso contiguo suman en cambio
la primera mitad con la segunda, con el mismo número de pasos. -->

---

## **Reducción paralela**

![w:560px](images/threads/par_red.png)

<p class="credit">Fuente: sodocumentation.net</p>

<!-- NOTA — en esta figura conviene marcar dos cosas que se repiten en todas las
versiones: cuántos threads trabajan en cada nivel (la mitad del anterior) y
qué distancia hay entre los dos elementos que se suman (el stride). Las
versiones que siguen difieren justamente en cómo se asignan threads a
elementos y en el orden de los strides. -->

---

## **Reducción paralela: memoria global**

Ejemplo: [reduccion_global.cu](../code/threads/reduccion_global.cu).

- *Kernel* invocado dentro de un ciclo.
- El *stride* cambia en cada iteración por un factor de $2$.

<!-- NOTA — la primera versión no usa memoria compartida ni bloques de forma
inteligente: el ciclo sobre el stride está en el host, y cada nivel del árbol
es un lanzamiento de kernel distinto. Con N = 2^24 son 24 lanzamientos. La
ventaja es que el fin de cada kernel sincroniza todo el grid, así que no hace
falta ninguna barrera dentro del kernel. -->

---

## **Reducción paralela: memoria global**

![w:720px](images/threads/reduction_global.png)

`stride = 8` no puede pasar porque el ciclo va hasta `stride < N`.

<!-- NOTA — la condición del ciclo es stride < N: con stride = N/2 se suma la
última mitad sobre la primera y el resultado queda en data[0]. Con stride = N
ya no queda nada que sumar. -->

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

<!-- NOTA — ojo con el código de la diapositiva: muestra data_out y data_in, pero
reduccion_global.cu trabaja sobre un solo arreglo, data, que se va
actualizando en el lugar (data[idx] += data[idx + stride]). Es seguro porque
cada lanzamiento lee y escribe posiciones distintas, y el siguiente
lanzamiento empieza recién cuando termina el anterior. -->

---

## **Reducción paralela: memoria global**

Hay un problema con el *kernel*... ¿cuál es?

... todos los *threads* calculan algo, ¡pero los resultados no son necesarios!

<!-- NOTA — la respuesta: en cada lanzamiento todos los threads hacen la suma, pero
en el nivel con stride s solo sirven los resultados de los índices múltiplos
de 2s; el resto escribe valores que nadie va a leer. En el último nivel
trabajan millones de threads para producir un único número útil. Además, todo
pasa por memoria global, y cada nivel lee y escribe el arreglo entero. -->

---

## **Reducción paralela: menos threads trabajando**

Ejemplo: [reduccion_global2.cu](../code/threads/reduccion_global2.cu).

![w:720px](images/threads/reduction_global_strided.png)

<!-- NOTA — reduccion_global2.cu agrega la condición idx % (2*stride) == 0: ahora
solo suman los threads cuyos resultados importan. Se hace menos trabajo
inútil, pero aparece un problema nuevo, que es el tema de las diapositivas
siguientes: dentro de un warp, unos threads cumplen la condición y otros no. -->

---

## **Divergencia de warps**

Este método también sufre de un problema: **divergencia de threads**.

Si en un programa de CUDA tenemos algo como:

```cuda
if (threadIdx.x % 2 == 0) {
  // rama 1
} else {
  // rama 2
}
```

Los *threads* de índice par ejecutan la **rama 1** y los de índice impar la **rama 2**.

<!-- NOTA — el if en sí no es el problema; el problema es que la condición tome
valores distintos DENTRO de un mismo warp. Si todos los threads de un warp van
por la misma rama, no hay costo extra. -->

---

## **Divergencia de warps**

- Recordar que en CUDA (SIMT), todos los *threads* dentro de un *warp* ejecutan la misma instrucción en cada instante.
- Entonces, al tener condiciones de *if* y *else* dentro de un *kernel*, se produce divergencia.
- En el ejemplo (y en el código de la reducción) los *threads* de índice par ejecutan sus instrucciones, mientras los otros **esperan**.
- La divergencia de *warp* (*warp divergence*) implica menos eficiencia de un *kernel*.

<!-- NOTA — matiz para no enseñar algo desactualizado: el modelo de 'todos ejecutan
la misma instrucción en el mismo momento' es el de antes de Volta. Desde Volta
cada thread tiene su propio contador de programa, pero el hardware igual
ejecuta un warp de a una instrucción para un grupo de threads: cuando las
ramas se separan, se ejecuta una rama con unos threads activos y después la
otra con los otros. El costo para el rendimiento es el mismo: las dos ramas se
pagan en serie. -->

---

## **Divergencia de warps**

<svg viewBox="0 0 992 320" width="1000" style="display:block;margin:0.3em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-div" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2e7d32"/></marker></defs>
<text x="406" y="16" font-size="17" text-anchor="middle" fill="#333">threads</text>
<text x="79" y="38" font-size="15" text-anchor="middle" fill="#333">0</text>
<text x="226" y="38" font-size="15" text-anchor="middle" fill="#333">7</text>
<text x="394" y="38" font-size="15" text-anchor="middle" fill="#333">15</text>
<text x="562" y="38" font-size="15" text-anchor="middle" fill="#333">23</text>
<text x="730" y="38" font-size="15" text-anchor="middle" fill="#333">31</text>
<line x1="56" y1="48" x2="56" y2="300" stroke="#2e7d32" stroke-width="2.5" marker-end="url(#fl-div)"/>
<text x="44" y="174" font-size="16" text-anchor="middle" fill="#333" transform="rotate(-90 44 174)">tiempo</text>
<rect x="70" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="91" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="112" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="133" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="154" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="175" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="196" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="217" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="238" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="259" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="280" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="301" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="322" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="343" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="364" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="385" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="406" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="427" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="448" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="469" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="490" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="511" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="532" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="553" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="574" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="595" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="616" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="637" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="658" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="679" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="700" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="721" y="48" width="18" height="18" fill="#5b8fd9"/>
<rect x="70" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="91" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="112" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="133" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="154" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="175" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="196" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="217" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="238" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="259" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="280" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="301" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="322" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="343" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="364" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="385" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="406" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="427" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="448" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="469" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="490" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="511" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="532" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="553" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="574" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="595" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="616" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="637" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="658" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="679" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="700" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="721" y="69" width="18" height="18" fill="#5b8fd9"/>
<rect x="70" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="91" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="112" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="133" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="154" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="175" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="196" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="217" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="238" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="259" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="280" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="301" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="322" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="343" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="364" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="385" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="406" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="427" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="448" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="469" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="490" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="511" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="532" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="553" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="574" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="595" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="616" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="637" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="658" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="679" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="700" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="721" y="90" width="18" height="18" fill="#5b8fd9"/>
<rect x="70" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="91" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="112" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="133" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="154" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="175" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="196" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="217" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="238" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="259" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="280" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="301" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="322" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="343" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="364" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="385" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="406" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="427" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="448" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="469" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="490" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="511" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="532" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="553" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="574" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="595" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="616" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="637" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="658" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="679" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="700" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="721" y="111" width="18" height="18" fill="#5b8fd9"/>
<rect x="70" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="91" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="112" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="133" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="154" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="175" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="196" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="217" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="238" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="259" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="280" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="301" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="322" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="343" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="364" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="385" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="406" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="427" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="448" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="469" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="490" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="511" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="532" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="553" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="574" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="595" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="616" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="637" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="658" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="679" y="132" width="18" height="18" fill="#ffc59e"/>
<rect x="700" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="721" y="132" width="18" height="18" fill="#7030a0"/>
<rect x="70" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="91" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="112" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="133" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="154" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="175" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="196" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="217" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="238" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="259" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="280" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="301" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="322" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="343" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="364" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="385" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="406" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="427" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="448" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="469" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="490" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="511" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="532" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="553" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="574" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="595" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="616" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="637" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="658" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="679" y="153" width="18" height="18" fill="#ffc59e"/>
<rect x="700" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="721" y="153" width="18" height="18" fill="#7030a0"/>
<rect x="70" y="174" width="18" height="18" fill="#92d050"/>
<rect x="91" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="112" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="133" y="174" width="18" height="18" fill="#92d050"/>
<rect x="154" y="174" width="18" height="18" fill="#92d050"/>
<rect x="175" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="196" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="217" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="238" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="259" y="174" width="18" height="18" fill="#92d050"/>
<rect x="280" y="174" width="18" height="18" fill="#92d050"/>
<rect x="301" y="174" width="18" height="18" fill="#92d050"/>
<rect x="322" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="343" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="364" y="174" width="18" height="18" fill="#92d050"/>
<rect x="385" y="174" width="18" height="18" fill="#92d050"/>
<rect x="406" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="427" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="448" y="174" width="18" height="18" fill="#92d050"/>
<rect x="469" y="174" width="18" height="18" fill="#92d050"/>
<rect x="490" y="174" width="18" height="18" fill="#92d050"/>
<rect x="511" y="174" width="18" height="18" fill="#92d050"/>
<rect x="532" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="553" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="574" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="595" y="174" width="18" height="18" fill="#92d050"/>
<rect x="616" y="174" width="18" height="18" fill="#92d050"/>
<rect x="637" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="658" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="679" y="174" width="18" height="18" fill="#7030a0"/>
<rect x="700" y="174" width="18" height="18" fill="#92d050"/>
<rect x="721" y="174" width="18" height="18" fill="#92d050"/>
<rect x="70" y="195" width="18" height="18" fill="#92d050"/>
<rect x="91" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="112" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="133" y="195" width="18" height="18" fill="#92d050"/>
<rect x="154" y="195" width="18" height="18" fill="#92d050"/>
<rect x="175" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="196" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="217" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="238" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="259" y="195" width="18" height="18" fill="#92d050"/>
<rect x="280" y="195" width="18" height="18" fill="#92d050"/>
<rect x="301" y="195" width="18" height="18" fill="#92d050"/>
<rect x="322" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="343" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="364" y="195" width="18" height="18" fill="#92d050"/>
<rect x="385" y="195" width="18" height="18" fill="#92d050"/>
<rect x="406" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="427" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="448" y="195" width="18" height="18" fill="#92d050"/>
<rect x="469" y="195" width="18" height="18" fill="#92d050"/>
<rect x="490" y="195" width="18" height="18" fill="#92d050"/>
<rect x="511" y="195" width="18" height="18" fill="#92d050"/>
<rect x="532" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="553" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="574" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="595" y="195" width="18" height="18" fill="#92d050"/>
<rect x="616" y="195" width="18" height="18" fill="#92d050"/>
<rect x="637" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="658" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="679" y="195" width="18" height="18" fill="#7030a0"/>
<rect x="700" y="195" width="18" height="18" fill="#92d050"/>
<rect x="721" y="195" width="18" height="18" fill="#92d050"/>
<rect x="70" y="216" width="18" height="18" fill="#92d050"/>
<rect x="91" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="112" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="133" y="216" width="18" height="18" fill="#92d050"/>
<rect x="154" y="216" width="18" height="18" fill="#92d050"/>
<rect x="175" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="196" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="217" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="238" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="259" y="216" width="18" height="18" fill="#92d050"/>
<rect x="280" y="216" width="18" height="18" fill="#92d050"/>
<rect x="301" y="216" width="18" height="18" fill="#92d050"/>
<rect x="322" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="343" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="364" y="216" width="18" height="18" fill="#92d050"/>
<rect x="385" y="216" width="18" height="18" fill="#92d050"/>
<rect x="406" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="427" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="448" y="216" width="18" height="18" fill="#92d050"/>
<rect x="469" y="216" width="18" height="18" fill="#92d050"/>
<rect x="490" y="216" width="18" height="18" fill="#92d050"/>
<rect x="511" y="216" width="18" height="18" fill="#92d050"/>
<rect x="532" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="553" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="574" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="595" y="216" width="18" height="18" fill="#92d050"/>
<rect x="616" y="216" width="18" height="18" fill="#92d050"/>
<rect x="637" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="658" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="679" y="216" width="18" height="18" fill="#7030a0"/>
<rect x="700" y="216" width="18" height="18" fill="#92d050"/>
<rect x="721" y="216" width="18" height="18" fill="#92d050"/>
<rect x="70" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="91" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="112" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="133" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="154" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="175" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="196" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="217" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="238" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="259" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="280" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="301" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="322" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="343" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="364" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="385" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="406" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="427" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="448" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="469" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="490" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="511" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="532" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="553" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="574" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="595" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="616" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="637" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="658" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="679" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="700" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="721" y="237" width="18" height="18" fill="#5b8fd9"/>
<rect x="70" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="91" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="112" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="133" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="154" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="175" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="196" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="217" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="238" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="259" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="280" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="301" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="322" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="343" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="364" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="385" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="406" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="427" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="448" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="469" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="490" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="511" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="532" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="553" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="574" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="595" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="616" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="637" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="658" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="679" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="700" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="721" y="258" width="18" height="18" fill="#5b8fd9"/>
<rect x="70" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="91" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="112" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="133" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="154" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="175" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="196" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="217" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="238" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="259" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="280" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="301" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="322" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="343" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="364" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="385" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="406" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="427" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="448" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="469" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="490" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="511" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="532" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="553" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="574" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="595" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="616" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="637" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="658" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="679" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="700" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="721" y="279" width="18" height="18" fill="#5b8fd9"/>
<rect x="772" y="108" width="18" height="18" fill="#5b8fd9"/>
<text x="800" y="123" font-size="17" fill="#333">código común</text>
<rect x="772" y="138" width="18" height="18" fill="#ffc59e"/>
<text x="800" y="153" font-size="17" fill="#333">rama if</text>
<rect x="772" y="168" width="18" height="18" fill="#92d050"/>
<text x="800" y="183" font-size="17" fill="#333">rama else</text>
<rect x="772" y="198" width="18" height="18" fill="#7030a0"/>
<text x="800" y="213" font-size="17" fill="#333">en espera</text>
</svg>

<!-- NOTA — la figura muestra las dos ramas ejecutándose una después de la otra,
con la mitad de los threads del warp inactivos en cada una. En
reduccion_global2 es peor que la mitad: en el nivel con stride s solo trabaja
1 de cada 2s threads, así que en los niveles altos casi todo el warp está
esperando. -->

<!-- NOTA — versión redibujada (SVG) de la figura anterior, con el mismo patrón
leído de la imagen original. Cada columna es uno de los 32 threads de un warp
y el tiempo avanza hacia abajo. Al llegar al if, 16 threads toman la rama if
(naranja) y los otros 16 quedan en espera (morado); después se ejecuta la rama
else para esos 16, y ahora esperan los 16 primeros. Las dos ramas se pagan una
después de la otra: el warp tarda lo que suman las dos, aunque cada thread
ejecuta solo una. La figura original rotula la segunda rama como 'then
clause', pero son los threads que NO tomaron el if, o sea la rama else; por
eso aquí dice rama else. Cuando las ramas terminan, el warp vuelve a ejecutar
todo junto (código común). -->

---

## **Divergencia de warps**

En los *profilers* se puede medir el nivel de *warp divergence*:
- `nvprof`: `branch_efficiency`
- `ncu`: `smsp__sass_average_branch_targets_threads_uniform.pct`

<!-- NOTA — branch_efficiency (nvprof) es el porcentaje de saltos en que todo el
warp tomó el mismo camino; 100% es sin divergencia. En ncu la métrica
equivalente mide el promedio de threads activos en las ramas. Se puede pedir
que comparen reduccion_global2 con reduccion_global3 con esta métrica. -->

---

## **Reducción paralela: sin warp divergence**

Ejemplo: [reduccion_global3.cu](../code/threads/reduccion_global3.cu).

![w:620px](images/threads/reduction_global_noWD.png)

Puede ser problemático usar el índice global de los *threads* para un *array* muy grande...

<!-- NOTA — reduccion_global3.cu cambia QUÉ thread suma qué par: el thread idx suma
la posición 2*stride*idx. Así los threads que trabajan son siempre los
primeros (índices bajos) y quedan juntos en los mismos warps: los warps están
o completos o vacíos, y no hay divergencia. El problema del final de la
diapositiva es real: 2*stride*idx crece rápido, y para arreglos muy grandes no
cabe en un int, por eso el código usa unsigned long. -->

---

## **Reducción paralela: usando bloques**

Ejemplo: [reduccion_global4.cu](../code/threads/reduccion_global4.cu).

<svg viewBox="0 0 1000 300" width="1100" style="display:block;margin:0.3em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-blq" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#e65100"/></marker></defs>
<rect x="4" y="4" width="992" height="146" fill="#edf3dc" stroke="#999"/>
<rect x="4" y="156" width="992" height="140" fill="#fbe5d6" stroke="#999"/>
<text x="16" y="26" font-size="17" fill="#333">Reducción en el <tspan font-style="italic">device</tspan> usando muchos bloques en paralelo</text>
<text x="16" y="284" font-size="17" fill="#333">Reducción en el <tspan font-style="italic">host</tspan> de forma secuencial</text>
<text x="580" y="236" font-size="17" fill="#333">Copia del <tspan font-style="italic">device</tspan> al <tspan font-style="italic">host</tspan></text>
<line x1="82" y1="118.5" x2="447" y2="224.5" stroke="#7a8c50" stroke-width="1.3" stroke-dasharray="7,3,2,3"/>
<line x1="201" y1="118.5" x2="461" y2="224.5" stroke="#7a8c50" stroke-width="1.3" stroke-dasharray="7,3,2,3"/>
<line x1="320" y1="118.5" x2="475" y2="224.5" stroke="#7a8c50" stroke-width="1.3" stroke-dasharray="7,3,2,3"/>
<line x1="439" y1="118.5" x2="489" y2="224.5" stroke="#7a8c50" stroke-width="1.3" stroke-dasharray="7,3,2,3"/>
<line x1="558" y1="118.5" x2="503" y2="224.5" stroke="#7a8c50" stroke-width="1.3" stroke-dasharray="7,3,2,3"/>
<line x1="677" y1="118.5" x2="517" y2="224.5" stroke="#7a8c50" stroke-width="1.3" stroke-dasharray="7,3,2,3"/>
<line x1="796" y1="118.5" x2="531" y2="224.5" stroke="#7a8c50" stroke-width="1.3" stroke-dasharray="7,3,2,3"/>
<line x1="915" y1="118.5" x2="545" y2="224.5" stroke="#7a8c50" stroke-width="1.3" stroke-dasharray="7,3,2,3"/>
<line x1="47" y1="48" x2="52" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="57" y1="48" x2="52" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="67" y1="48" x2="72" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="77" y1="48" x2="72" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="87" y1="48" x2="92" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="97" y1="48" x2="92" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="107" y1="48" x2="112" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="117" y1="48" x2="112" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="52" y1="70" x2="62" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="72" y1="70" x2="62" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="92" y1="70" x2="102" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="112" y1="70" x2="102" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="62" y1="92" x2="82" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="102" y1="92" x2="82" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="166" y1="48" x2="171" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="176" y1="48" x2="171" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="186" y1="48" x2="191" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="196" y1="48" x2="191" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="206" y1="48" x2="211" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="216" y1="48" x2="211" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="226" y1="48" x2="231" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="236" y1="48" x2="231" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="171" y1="70" x2="181" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="191" y1="70" x2="181" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="211" y1="70" x2="221" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="231" y1="70" x2="221" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="181" y1="92" x2="201" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="221" y1="92" x2="201" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="285" y1="48" x2="290" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="295" y1="48" x2="290" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="305" y1="48" x2="310" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="315" y1="48" x2="310" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="325" y1="48" x2="330" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="335" y1="48" x2="330" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="345" y1="48" x2="350" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="355" y1="48" x2="350" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="290" y1="70" x2="300" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="310" y1="70" x2="300" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="330" y1="70" x2="340" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="350" y1="70" x2="340" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="300" y1="92" x2="320" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="340" y1="92" x2="320" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="404" y1="48" x2="409" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="414" y1="48" x2="409" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="424" y1="48" x2="429" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="434" y1="48" x2="429" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="444" y1="48" x2="449" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="454" y1="48" x2="449" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="464" y1="48" x2="469" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="474" y1="48" x2="469" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="409" y1="70" x2="419" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="429" y1="70" x2="419" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="449" y1="70" x2="459" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="469" y1="70" x2="459" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="419" y1="92" x2="439" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="459" y1="92" x2="439" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="523" y1="48" x2="528" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="533" y1="48" x2="528" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="543" y1="48" x2="548" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="553" y1="48" x2="548" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="563" y1="48" x2="568" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="573" y1="48" x2="568" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="583" y1="48" x2="588" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="593" y1="48" x2="588" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="528" y1="70" x2="538" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="548" y1="70" x2="538" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="568" y1="70" x2="578" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="588" y1="70" x2="578" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="538" y1="92" x2="558" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="578" y1="92" x2="558" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="642" y1="48" x2="647" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="652" y1="48" x2="647" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="662" y1="48" x2="667" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="672" y1="48" x2="667" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="682" y1="48" x2="687" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="692" y1="48" x2="687" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="702" y1="48" x2="707" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="712" y1="48" x2="707" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="647" y1="70" x2="657" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="667" y1="70" x2="657" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="687" y1="70" x2="697" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="707" y1="70" x2="697" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="657" y1="92" x2="677" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="697" y1="92" x2="677" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="761" y1="48" x2="766" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="771" y1="48" x2="766" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="781" y1="48" x2="786" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="791" y1="48" x2="786" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="801" y1="48" x2="806" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="811" y1="48" x2="806" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="821" y1="48" x2="826" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="831" y1="48" x2="826" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="766" y1="70" x2="776" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="786" y1="70" x2="776" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="806" y1="70" x2="816" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="826" y1="70" x2="816" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="776" y1="92" x2="796" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="816" y1="92" x2="796" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="880" y1="48" x2="885" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="890" y1="48" x2="885" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="900" y1="48" x2="905" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="910" y1="48" x2="905" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="920" y1="48" x2="925" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="930" y1="48" x2="925" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="940" y1="48" x2="945" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="950" y1="48" x2="945" y2="70" stroke="#555" stroke-width="1.2"/>
<line x1="885" y1="70" x2="895" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="905" y1="70" x2="895" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="925" y1="70" x2="935" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="945" y1="70" x2="935" y2="92" stroke="#555" stroke-width="1.2"/>
<line x1="895" y1="92" x2="915" y2="114" stroke="#555" stroke-width="1.2"/>
<line x1="935" y1="92" x2="915" y2="114" stroke="#555" stroke-width="1.2"/>
<circle cx="47" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="57" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="67" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="77" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="87" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="97" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="107" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="117" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="52" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="72" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="92" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="112" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="62" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="102" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="82" cy="114" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="166" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="176" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="186" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="196" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="206" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="216" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="226" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="236" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="171" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="191" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="211" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="231" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="181" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="221" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="201" cy="114" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="285" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="295" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="305" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="315" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="325" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="335" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="345" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="355" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="290" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="310" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="330" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="350" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="300" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="340" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="320" cy="114" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="404" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="414" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="424" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="434" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="444" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="454" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="464" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="474" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="409" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="429" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="449" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="469" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="419" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="459" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="439" cy="114" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="523" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="533" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="543" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="553" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="563" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="573" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="583" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="593" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="528" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="548" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="568" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="588" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="538" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="578" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="558" cy="114" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="642" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="652" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="662" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="672" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="682" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="692" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="702" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="712" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="647" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="667" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="687" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="707" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="657" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="697" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="677" cy="114" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="761" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="771" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="781" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="791" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="801" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="811" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="821" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="831" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="766" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="786" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="806" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="826" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="776" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="816" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="796" cy="114" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="880" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="890" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="900" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="910" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="920" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="930" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="940" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="950" cy="48" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="885" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="905" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="925" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="945" cy="70" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="895" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="935" cy="92" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="915" cy="114" r="4.5" fill="#f4a582" stroke="#333" stroke-width="1"/>
<circle cx="447" cy="230" r="5.0" fill="#cfe2f3" stroke="#333" stroke-width="1"/>
<circle cx="461" cy="230" r="5.0" fill="#cfe2f3" stroke="#333" stroke-width="1"/>
<circle cx="475" cy="230" r="5.0" fill="#cfe2f3" stroke="#333" stroke-width="1"/>
<circle cx="489" cy="230" r="5.0" fill="#cfe2f3" stroke="#333" stroke-width="1"/>
<circle cx="503" cy="230" r="5.0" fill="#cfe2f3" stroke="#333" stroke-width="1"/>
<circle cx="517" cy="230" r="5.0" fill="#cfe2f3" stroke="#333" stroke-width="1"/>
<circle cx="531" cy="230" r="5.0" fill="#cfe2f3" stroke="#333" stroke-width="1"/>
<circle cx="545" cy="230" r="5.0" fill="#cfe2f3" stroke="#333" stroke-width="1"/>
<line x1="438" y1="230" x2="358" y2="230" stroke="#e65100" stroke-width="1.6" stroke-dasharray="7,3,2,3" marker-end="url(#fl-blq)"/>
<circle cx="348" cy="230" r="6" fill="#d32f2f"/>
<rect x="52" y="126" width="60" height="16" fill="#edf3dc"/><text x="82" y="138" font-size="12" text-anchor="middle" fill="#555">bloque 0</text>
<rect x="171" y="126" width="60" height="16" fill="#edf3dc"/><text x="201" y="138" font-size="12" text-anchor="middle" fill="#555">bloque 1</text>
<rect x="290" y="126" width="60" height="16" fill="#edf3dc"/><text x="320" y="138" font-size="12" text-anchor="middle" fill="#555">bloque 2</text>
<rect x="409" y="126" width="60" height="16" fill="#edf3dc"/><text x="439" y="138" font-size="12" text-anchor="middle" fill="#555">bloque 3</text>
<rect x="528" y="126" width="60" height="16" fill="#edf3dc"/><text x="558" y="138" font-size="12" text-anchor="middle" fill="#555">bloque 4</text>
<rect x="647" y="126" width="60" height="16" fill="#edf3dc"/><text x="677" y="138" font-size="12" text-anchor="middle" fill="#555">bloque 5</text>
<rect x="766" y="126" width="60" height="16" fill="#edf3dc"/><text x="796" y="138" font-size="12" text-anchor="middle" fill="#555">bloque 6</text>
<rect x="885" y="126" width="60" height="16" fill="#edf3dc"/><text x="915" y="138" font-size="12" text-anchor="middle" fill="#555">bloque 7</text>
</svg>

<p class="credit">Fuente: adaptado de <em>Professional CUDA C Programming</em></p>

<!-- NOTA — reduccion_global4.cu cambia de estrategia: cada bloque reduce SU parte
del arreglo dentro de un solo kernel, sincronizando con __syncthreads() entre
niveles, y deja un resultado parcial por bloque; la suma de los parciales se
hace en el host. Detalle corregido en el código: los parciales se guardan en
un arreglo aparte, porque escribirlos en data[blockIdx.x] pisaba datos que el
bloque 0 todavía estaba sumando. Un solo lanzamiento en vez de 24. -->

<!-- NOTA — versión redibujada (SVG) de la figura anterior. Arriba, en el device:
cada bloque reduce su parte del arreglo en paralelo con los demás, con el
árbol de siempre y __syncthreads() entre niveles, y deja UN resultado parcial.
Las líneas punteadas son esos parciales, uno por bloque, que se copian al
host. Abajo, en el host: un ciclo secuencial suma los parciales. En
reduccion_global4.cu, con N = 2^24 y bloques de 1024, son 16384 parciales: el
host suma 16384 números en vez de 16 millones, y el GPU hace todo el resto en
un solo lanzamiento. La figura dibuja 8 bloques de 8 elementos solo para que
se vea. Más adelante la reducción completa (reduccion-shuffle.cu) reemplaza la
suma en el host por un atomicAdd por bloque. -->

---

## **Reducción paralela: acceso contiguo**

Ejemplo: [reduccion_global5.cu](../code/threads/reduccion_global5.cu).

<svg viewBox="0 0 620 392" width="660" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-cont" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<line x1="192" y1="50" x2="192" y2="68" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="408" y1="50" x2="208.8" y2="82.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="192" y1="101" x2="192" y2="119" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="246" y1="50" x2="246" y2="68" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="462" y1="50" x2="262.8" y2="82.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="246" y1="101" x2="246" y2="119" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="300" y1="50" x2="300" y2="68" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="516" y1="50" x2="316.8" y2="82.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="300" y1="101" x2="300" y2="119" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="354" y1="50" x2="354" y2="68" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="570" y1="50" x2="370.8" y2="82.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="354" y1="101" x2="354" y2="119" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="192" y1="160" x2="192" y2="178" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="300" y1="160" x2="208.2" y2="189.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="192" y1="211" x2="192" y2="229" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="246" y1="160" x2="246" y2="178" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="354" y1="160" x2="262.2" y2="189.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="246" y1="211" x2="246" y2="229" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="192" y1="270" x2="192" y2="288" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="246" y1="270" x2="206.3" y2="295.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<line x1="192" y1="321" x2="192" y2="339" stroke="#444" stroke-width="1.4" marker-end="url(#fl-cont)"/>
<rect x="170" y="10" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="36" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="224" y="10" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="36" font-size="18" text-anchor="middle" fill="#222">1</text>
<rect x="278" y="10" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="36" font-size="18" text-anchor="middle" fill="#222">7</text>
<rect x="332" y="10" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="36" font-size="18" text-anchor="middle" fill="#222">0</text>
<rect x="386" y="10" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="36" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="440" y="10" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="462" y="36" font-size="18" text-anchor="middle" fill="#222">1</text>
<rect x="494" y="10" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="36" font-size="18" text-anchor="middle" fill="#222">6</text>
<rect x="548" y="10" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="570" y="36" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="170" y="120" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="146" font-size="18" text-anchor="middle" fill="#222">7</text>
<rect x="224" y="120" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="146" font-size="18" text-anchor="middle" fill="#222">2</text>
<rect x="278" y="120" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="146" font-size="18" text-anchor="middle" fill="#222">13</text>
<rect x="332" y="120" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="146" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="386" y="120" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="120" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="120" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="120" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="230" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="256" font-size="18" text-anchor="middle" fill="#222">20</text>
<rect x="224" y="230" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="256" font-size="18" text-anchor="middle" fill="#222">5</text>
<rect x="278" y="230" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="230" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="230" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="230" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="230" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="230" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="340" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="366" font-size="18" text-anchor="middle" fill="#222">25</text>
<rect x="224" y="340" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="340" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="340" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="340" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="340" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="340" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="340" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<circle cx="192" cy="85" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="91" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="246" cy="85" r="16" fill="#e8641b" stroke="#444"/>
<text x="246" y="91" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="300" cy="85" r="16" fill="#e8641b" stroke="#444"/>
<text x="300" y="91" font-size="16" text-anchor="middle" fill="#fff">2</text>
<circle cx="354" cy="85" r="16" fill="#e8641b" stroke="#444"/>
<text x="354" y="91" font-size="16" text-anchor="middle" fill="#fff">3</text>
<circle cx="192" cy="195" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="201" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="246" cy="195" r="16" fill="#e8641b" stroke="#444"/>
<text x="246" y="201" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="192" cy="305" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="311" font-size="16" text-anchor="middle" fill="#fff">0</text>
<text x="156" y="36" font-size="16" text-anchor="end" fill="#333">Memoria global</text>
<text x="156" y="91" font-size="16" text-anchor="end" fill="#333">ID del thread</text>
<text x="156" y="109" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 4</text>
<text x="156" y="219" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<text x="156" y="329" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
</svg>

<p class="credit">Fuente: adaptado de <em>Professional CUDA C Programming</em></p>

<!-- NOTA — reduccion_global5.cu invierte el orden de los strides: empieza con
stride = blockDim/2 y lo va dividiendo por 2, y suma el thread threadIdx.x con
threadIdx.x + stride. Así los threads activos son siempre los primeros (sin
divergencia) Y cada warp lee posiciones consecutivas (acceso coalescido), lo
que conecta con el capítulo de memoria. Es el patrón que se usa en todas las
versiones que siguen. -->

<!-- NOTA — versión redibujada (SVG) de la figura anterior, con los mismos valores;
las sumas están verificadas: con stride 4, 3+4=7, 1+1=2, 7+6=13 y 0+3=3; con
stride 2, 7+13=20 y 2+3=5; con stride 1, 20+5=25, que es la suma de los ocho
datos. Es exactamente lo que hace reduccion_global5.cu: el stride empieza en
blockDim/2 y se divide por 2, y el thread t suma la casilla t + stride sobre
la casilla t. Dos cosas que mirar: los threads que trabajan son siempre los
primeros (0 a stride-1), así que no hay divergencia dentro de los warps
mientras stride sea al menos 32; y las lecturas de un paso son dos tramos
contiguos (t y t + stride), así que el acceso es coalescido. Las casillas lila
quedan con datos que ya no se usan. -->

---

## **Optimización: loop unrolling**

- Ahora aplicaremos una técnica de optimización que también aplica a la programación secuencial: *loop unrolling*.
- Primero, veremos un ejemplo en programación secuencial.

<!-- NOTA — el loop unrolling es una técnica clásica de compiladores. En el GPU
tiene un beneficio extra que no está en la diapositiva: varias cargas
independientes en el mismo thread pueden estar en vuelo al mismo tiempo, y eso
esconde latencia aunque haya pocos warps. -->

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

<!-- NOTA — las dos versiones hacen exactamente el mismo trabajo útil; la segunda
hace la mitad de las comparaciones y saltos del ciclo. Ojo: la versión
desenrollada supone que N es par; en código real hay que tratar el resto. -->

---

## **Optimización: loop unrolling**

- El segundo ciclo tiene menos iteraciones.
- Por lo tanto hay menos *overhead* de verificar si el ciclo puede terminar.
- También hay oportunidades de optimizar el acceso a la memoria (uso de *cache*, etc.).
- Para *loops* así, los compiladores modernos típicamente aplican *loop unrolling* automáticamente.

<!-- NOTA — en un ciclo tan simple el compilador probablemente ya lo desenrolla
solo (nvcc lo hace con ciclos de conteo conocido, y se puede pedir con #pragma
unroll, como en memoria_constante.cu). En la reducción, en cambio, el
unrolling que viene no es del ciclo sino del trabajo entre bloques, y eso el
compilador no lo puede hacer por nosotros. -->

---

## **Reducción paralela: loop unrolling**

¿Cómo aplicamos *loop unrolling* en el *kernel* de reducción paralela?

Aplicamos una suma entre bloques **antes** de comenzar con el ciclo de la reducción dentro de los bloques.

<!-- NOTA — la idea: antes del árbol dentro del bloque, cada thread suma elementos
de varios bloques de datos. Así el mismo número de threads reduce el doble (o
4, u 8 veces) de datos, y se lanzan la mitad de bloques. Es el antecedente
directo del grid-stride loop, que hace lo mismo sin un factor fijo. -->

---

## **Reducción paralela: loop unrolling**

Ejemplo: [reduccion_global6.cu](../code/threads/reduccion_global6.cu).

![w:720px](images/threads/reduction_global_unrolled.png)

<!-- NOTA — reduccion_global6.cu: cada bloque procesa 2*blockDim elementos. Primero
cada thread suma su elemento con el que está un bloque más adelante, y después
se hace el árbol de siempre sobre blockDim elementos. Se lanza con
n_bloques/2. -->

---

## **Optimización: loop unrolling**

```c
for (int i = 0; i < N; i+=2){
  a[i]   = b[i]   + c[i];
  a[i+1] = b[i+1] + c[i+1];
}
```

Esto corresponde a *loop unrolling* con un factor de $2$.

<!-- NOTA — volver al ejemplo secuencial para conectar el nombre: el factor de
unrolling es cuántas iteraciones del ciclo original se hacen en una pasada. -->

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

<!-- NOTA — con factor 4 hay un cuarto de las comparaciones del ciclo original.
Factores más grandes rinden cada vez menos y gastan más registros, que es la
tensión con el occupancy que vimos antes. -->

---

## **Reducción paralela: loop unrolling**

```cuda
if (idx + blockDim.x < N) data[idx] += data[idx + blockDim.x];
```

Con esa línea sumamos valores en $2$ bloques (*loop unrolling* $\times 2$).

<!-- NOTA — esta línea es todo el unrolling x2 del kernel: un thread suma dos
elementos separados por un bloque. Notar que el acceso sigue siendo contiguo
dentro del warp: threads consecutivos leen idx e idx + blockDim.x, los dos
consecutivos. -->

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

<!-- NOTA — con x4 conviene cargar primero los cuatro valores en variables y
sumarlos después: las cuatro cargas son independientes y pueden estar en vuelo
al mismo tiempo. Esa es la ganancia real del unrolling en el GPU, más que
ahorrar comparaciones. -->

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

<!-- NOTA — x8 lleva la idea al extremo: ocho cargas independientes por thread y un
octavo de los bloques. A partir de cierto factor la ganancia se satura, porque
el kernel ya está cerca del ancho de banda de la memoria. -->

---

## **Reducción paralela: loop unrolling**

Ejemplo: [reduccion_global7.cu](../code/threads/reduccion_global7.cu).

Este código incluye $3$ *kernels* con distintos factores de *unrolling*.

<!-- NOTA — reduccion_global7.cu ejecuta los tres factores (x2, x4, x8) uno tras
otro y compara el resultado de cada uno con el del host. Para comparar tiempos
hay que usar el profiler: nvprof ./reduccion_global7.x muestra los tres
kernels por separado. Todavía sin medir en la T4. -->

---

## **Reducción paralela: memoria compartida**

- Hasta ahora hemos usado memoria global en todos los *kernels*.
- Podemos optimizar el acceso a la memoria usando **memoria compartida**.

Ejemplo: [reduccion_compartida.cu](../code/threads/reduccion_compartida.cu).

<!-- NOTA — reduccion_compartida.cu copia el bloque a memoria compartida y hace el
árbol ahí, así la memoria global se lee una sola vez por elemento. Es una
versión más simple que las anteriores (usa la indexación con divergencia de
reduccion_global2) para aislar la idea de memoria compartida. Se lanza en
varias pasadas hasta que queda un número; desde la corrección, cada pasada
escribe sus parciales en otro arreglo y los punteros se intercambian. -->

---

## **Reducción paralela: warp unrolling**

- Cuando llegamos a $< 32$ elementos en un bloque, el trabajo cabe en un *warp*.
- Dentro de un *warp* no hace falta `__syncthreads()` (que sincroniza todo el bloque), pero desde Volta **sí** `__syncwarp()`.
- Además, tendremos *threads* que no trabajan mientras el número de sumas disminuye: menos eficiente.
- Podemos hacer *unroll* dentro del último *warp* para optimizar más.

<!-- NOTA — cuando el árbol llega a 32 elementos, todo el trabajo restante lo hace
un solo warp. Seguir usando __syncthreads() frena a todo el bloque por un
warp, y el if (threadIdx.x < stride) deja cada vez más lanes ociosos.
Desenrollar los últimos pasos ahorra las barreras de bloque; pero desde Volta
hay que sincronizar el warp, que es lo que muestra la diapositiva siguiente. -->

---

## **Reducción paralela: warp unrolling**

```cuda
if (threadIdx.x < 32) {
  volatile float *vmem = idata;
  float v = vmem[threadIdx.x];
  for (int stride = 32; stride > 0; stride >>= 1) {
    v += vmem[threadIdx.x + stride];
    __syncwarp();            // todos leyeron
    vmem[threadIdx.x] = v;
    __syncwarp();            // todos escribieron
  }
}
```

Ejemplo: [reduccion_global8.cu](../code/threads/reduccion_global8.cu). Todos los *threads* del *warp* calculan, pero solo necesitamos el resultado del primero.

<!-- NOTA — la versión original (la del libro) hacía las seis sumas seguidas
sin __syncwarp(), confiando en que el warp avanzaba en lock-step. En GPUs
anteriores a Volta funcionaba; desde Volta es una race condition: el thread t
lee vmem[t + stride], que el thread t + stride escribe en el MISMO paso. Por
eso cada paso separa la lectura de la escritura con dos __syncwarp(). Queda
feo, y es justamente la motivación de las warp primitives que vienen: el mismo
paso con __shfl_down_sync no necesita memoria compartida, ni volatile, ni
__syncwarp(). -->

---

## **¿Qué es `volatile`?**

- `volatile` es un calificador: le dice al compilador que **no** guarde la variable en un registro. Cada lectura y escritura va a la memoria.
- Así un *thread* ve lo que escribieron los otros, en vez de una copia vieja en su registro.
- Pero `volatile` **no sincroniza**: el orden entre los *threads* lo garantiza `__syncwarp()`.

<!-- NOTA — volatile es necesario pero no suficiente. Sin volatile, el compilador
podría guardar vmem[threadIdx.x] en un registro y nunca ver lo que escribieron
otros threads. Con volatile cada acceso va a la memoria, pero eso no ordena a
los threads entre sí: sin __syncwarp(), un thread podría leer antes de que el
otro escriba. El material original decía que volatile evitaba las race
conditions; no es así. -->

---

# Warp primitives

---

## **Warp primitives**

- Las *warp primitives* (funciones primitivas de *warp*) son funciones básicas para trabajar directamente con los *threads* dentro de un *warp*.
- Un poco de jerga: un *thread* dentro de un *warp* se conoce como un *lane*.
- Hay muchas funciones disponibles: *CUDA Programming Guide*.
- Las usaremos para el último paso de la reducción: sin memoria compartida, sin `volatile` y sin `__syncwarp()`.

<!-- NOTA — lane es el índice del thread dentro de su warp, de 0 a 31 (threadIdx.x
% 32). Todas las primitivas modernas terminan en _sync y reciben una máscara:
son las versiones seguras desde Volta, porque sincronizan a los threads de la
máscara antes de intercambiar datos. -->

---

## **Warp primitives**

```cuda
#define FULL_MASK 0xffffffff  // 32-bits, todos igual a 1
for (int stride = 16; stride > 0; stride >>= 1)
  val += __shfl_down_sync(FULL_MASK, val, stride);
```

<svg viewBox="0 0 860 390" width="760" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-shfl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker></defs>
<text x="58" y="24" font-size="18" font-weight="bold" text-anchor="end" fill="#222">lane</text>
<text x="93" y="24" font-size="18" font-weight="bold" text-anchor="middle" fill="#222">0</text>
<text x="141" y="24" font-size="18" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<text x="189" y="24" font-size="18" font-weight="bold" text-anchor="middle" fill="#222">2</text>
<text x="237" y="24" font-size="18" font-weight="bold" text-anchor="middle" fill="#222">3</text>
<text x="285" y="24" font-size="18" font-weight="bold" text-anchor="middle" fill="#222">4</text>
<text x="333" y="24" font-size="18" font-weight="bold" text-anchor="middle" fill="#222">5</text>
<text x="381" y="24" font-size="18" font-weight="bold" text-anchor="middle" fill="#222">6</text>
<text x="429" y="24" font-size="18" font-weight="bold" text-anchor="middle" fill="#222">7</text>
<line x1="285" y1="80" x2="93" y2="137" stroke="#222" stroke-width="2" marker-end="url(#fl-shfl)"/>
<line x1="333" y1="80" x2="141" y2="137" stroke="#222" stroke-width="2" marker-end="url(#fl-shfl)"/>
<line x1="381" y1="80" x2="189" y2="137" stroke="#222" stroke-width="2" marker-end="url(#fl-shfl)"/>
<line x1="429" y1="80" x2="237" y2="137" stroke="#222" stroke-width="2" marker-end="url(#fl-shfl)"/>
<line x1="189" y1="180" x2="93" y2="237" stroke="#222" stroke-width="2" marker-end="url(#fl-shfl)"/>
<line x1="237" y1="180" x2="141" y2="237" stroke="#222" stroke-width="2" marker-end="url(#fl-shfl)"/>
<line x1="141" y1="280" x2="93" y2="337" stroke="#222" stroke-width="2" marker-end="url(#fl-shfl)"/>
<rect x="70" y="40" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="93" y="66" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<rect x="118" y="40" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="141" y="66" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<rect x="166" y="40" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="189" y="66" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<rect x="214" y="40" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="237" y="66" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<rect x="262" y="40" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="285" y="66" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<rect x="310" y="40" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="333" y="66" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<rect x="358" y="40" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="381" y="66" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<rect x="406" y="40" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="429" y="66" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">1</text>
<rect x="70" y="140" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="93" y="166" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">2</text>
<rect x="118" y="140" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="141" y="166" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">2</text>
<rect x="166" y="140" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="189" y="166" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">2</text>
<rect x="214" y="140" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="237" y="166" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">2</text>
<rect x="262" y="140" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="310" y="140" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="358" y="140" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="406" y="140" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="70" y="240" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="93" y="266" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">4</text>
<rect x="118" y="240" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="141" y="266" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">4</text>
<rect x="166" y="240" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="214" y="240" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="262" y="240" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="310" y="240" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="358" y="240" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="406" y="240" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="70" y="340" width="46" height="38" fill="#f9b917" stroke="#333"/>
<text x="93" y="366" font-size="20" font-weight="bold" text-anchor="middle" fill="#222">8</text>
<rect x="118" y="340" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="166" y="340" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="214" y="340" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="262" y="340" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="310" y="340" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="358" y="340" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<rect x="406" y="340" width="46" height="38" fill="#fde7b0" stroke="#333"/>
<text x="478" y="65" font-size="16" font-family="monospace" fill="#222">unsigned m = 0xffffffff;</text>
<text x="478" y="126" font-size="16" font-family="monospace" fill="#222">v += __shfl_down_sync(m, v, 4);</text>
<text x="478" y="226" font-size="16" font-family="monospace" fill="#222">v += __shfl_down_sync(m, v, 2);</text>
<text x="478" y="326" font-size="16" font-family="monospace" fill="#222">v += __shfl_down_sync(m, v, 1);</text>
</svg>

<p class="credit">Fuente: adaptado de NVIDIA Developer Blog</p>

<!-- NOTA — __shfl_down_sync(mascara, v, d): cada lane recibe el valor v del lane
que está d posiciones más arriba. Con d = 16, 8, 4, 2, 1, después de 5 pasos
el lane 0 tiene la suma de los 32. Los lanes que reciben de 'fuera' del warp
se quedan con su propio valor, pero eso no importa porque solo se usa el
resultado del lane 0. -->

<!-- NOTA — versión redibujada (SVG) de la figura anterior. Por espacio muestra 8
lanes con desplazamientos 4, 2 y 1; en un warp real son 32 lanes y los
desplazamientos 16, 8, 4, 2 y 1, como en el código. Cada flecha va del lane t
+ d al lane t: el lane t recibe el valor v del lane que está d posiciones más
arriba y lo suma al suyo. Todos los lanes empiezan con 1, así que se ve la
cuenta: 2 en 4 lanes, 4 en 2, y 8 en el lane 0. Las casillas claras siguen
teniendo un valor en el hardware, pero ya no importa: solo se usa el resultado
del lane 0. No hay memoria compartida ni __syncwarp(): el intercambio es
directo entre registros, y el _sync de la función sincroniza a los lanes de la
máscara m. -->

---

## **Reducción dentro de un warp con shuffle**

```cuda
__device__ float sumaWarp(float v) {
  for (int d = 16; d > 0; d >>= 1)
    v += __shfl_down_sync(0xffffffff, v, d);
  return v;  // el lane 0 tiene la suma de los 32
}
```

- Intercambio de datos **entre registros**: más eficiente que la memoria compartida (sin *load*, *store* ni dirección).
- La máscara dice qué *threads* participan; el `_sync` ya los sincroniza.
- Ejemplos: [warp_shuffle_down.cu](../code/threads/warp_shuffle_down.cu) (un vector de $32$), [warp_shuffle_up.cu](../code/threads/warp_shuffle_up.cu), [warp_shuffle_xor.cu](../code/threads/warp_shuffle_xor.cu).

<!-- NOTA — compárese con el warp unrolling de reduccion_global8: el mismo
resultado, sin memoria compartida, sin volatile y sin __syncwarp() explícito.
Los ejemplos de up y xor muestran las otras dos formas de mover valores; xor
es la base de la reducción "en mariposa", donde al final TODOS los lanes
tienen la suma (no solo el 0). Otra opción para este nivel de control son los
cooperative groups, fuera del alcance del curso. -->

---

# Operaciones atómicas

---

## **Operaciones atómicas**

- Cuando muchos *threads* escriben en la **misma** dirección de memoria hay una *race condition*: el resultado depende del orden de ejecución.
- Una operación **atómica** garantiza que la lectura-modificación-escritura ocurra sin interrupción (es *indivisible*).

```cuda
__global__ void suma_atomica(int *contador, const int *datos) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  atomicAdd(contador, datos[i]); // seguro: sin race condition
}
```

- Otras: `atomicSub`, `atomicMax`, `atomicMin`, `atomicExch`, `atomicCAS`, ...

<!-- NOTA — el ejemplo clásico: si dos threads hacen contador++ a la vez, los dos
leen el mismo valor viejo, los dos suman 1 y los dos escriben: se pierde una
cuenta. atomicAdd hace la lectura, la suma y la escritura como una sola
operación que nadie puede interrumpir. Funcionan en memoria global y en
memoria compartida; atomicAdd existe para int, unsigned, float y double (este
último desde CC 6.0). -->

---

## **Atómicas: costo y buenas prácticas**

- Serializan los accesos concurrentes → pueden ser **lentas** si hay mucha contención.
- Buena práctica: reducir primero dentro del bloque (memoria compartida o *warp primitives*) y usar **una** atómica por bloque.

```cuda
// cada bloque aporta su resultado parcial al total global
if (threadIdx.x == 0) atomicAdd(total, suma_del_bloque);
```

- Es una alternativa simple (con contención) a la **reducción** en paralelo que vimos antes.

<!-- NOTA — contención es cuántos threads atacan la MISMA dirección: si son muchos,
el hardware los atiende de a uno. Por eso un atomicAdd por thread sobre un
solo contador es lento, y uno por bloque es barato. Es exactamente el diseño
de la reducción completa y del histograma con memoria compartida. Detalle:
atomicAdd en float suma en el orden en que llegan los bloques, que cambia
entre ejecuciones; el resultado puede variar en los últimos dígitos. -->

---

## **Atómicas: histograma**

Ejemplo: [histograma.cu](../code/threads/histograma.cu). Cuenta cuántas veces aparece cada valor ($0$ a $255$) en $16$ M de bytes.

- `histGlobal`: `atomicAdd` directo en memoria global.
- `histCompartida`: histograma por bloque en memoria compartida, y al final una atómica global por contador.
- Se mide con datos **aleatorios** y con datos **constantes** (todos los *threads* compiten por el mismo contador).

```bash
nvcc -arch=sm_75 histograma.cu -o histograma.x
./histograma.x
```

<!-- NOTA — un histograma es el caso natural de atómicas: no se sabe de antemano
qué contador va a incrementar cada thread. Los datos constantes son el peor
caso a propósito: todos los threads al mismo contador, contención máxima. La
comparación interesante es cómo reacciona cada kernel a ese peor caso; es el
ejercicio 2. -->

---

## **Grid-stride loops (repaso)**

- Ya lo vimo en el Capítulo 1: `for (i = idx; i < N; i += blockDim.x * gridDim.x)`.
- Cada *thread* procesa varios elementos: el *kernel* sirve para cualquier $N$.
- En la reducción, cada *thread* primero suma varios elementos **en un registro**: es el *loop unrolling* entre bloques, sin un factor fijo.

Ejemplo (SAXPY): [grid_stride.cu](../code/threads/grid_stride.cu).

<!-- NOTA — el material original decía que el grid-stride "mejora el
occupancy". No es así: el occupancy depende de los recursos por bloque, no de
cuántos bloques se lanzan. Lo que sí hace es desacoplar el tamaño del grid del
tamaño de los datos (basta con llenar el GPU, p. ej. 32 bloques por SM),
reducir el costo de lanzar y retirar bloques, y amortizar el trabajo fijo de
cada thread (calcular índices, inicializar) sobre muchos elementos. -->

---

## **Reducción completa en un kernel**

```cuda
float suma = 0.f;                                  // 1. grid-stride
for (int i = idx; i < N; i += blockDim.x * gridDim.x)
  suma += data[i];
suma = sumaWarp(suma);                             // 2. shuffle en el warp
if (lane == 0) parcial[warp] = suma;               // 3. un valor por warp
__syncthreads();
if (warp == 0) {
  suma = sumaWarp(lane < BLOQUE / 32 ? parcial[lane] : 0.f);
  if (lane == 0) atomicAdd(total, suma);           // 4. una atómica por bloque
}
```

Ejemplo: [reduccion-shuffle.cu](../code/threads/reduccion-shuffle.cu). Un solo *kernel*, sin copiar resultados parciales al *host*.

<!-- NOTA — este kernel junta todo lo de la clase: grid-stride (cada thread
suma muchos elementos en un registro), warp primitives (sin memoria compartida
dentro del warp), memoria compartida solo para los 8 valores por bloque, y una
atómica por bloque para el total. Comparado con la serie
reduccion_global...global8: no lanza el kernel en un ciclo, no copia parciales
al host, no tiene divergencia en el ciclo principal y lee la memoria global de
forma contigua y una sola vez. Lo esperable es que se acerque al ancho de
banda de la T4 (300 GB/s), porque una reducción tiene intensidad aritmética
mínima (una suma por cada 4 bytes). -->

---

## **Reducción paralela: errores**

- El resultado del GPU no es exactamente igual al del CPU.
- Esto se debe a que siempre hay errores de redondeo, y en el cálculo secuencial estos errores se acumulan.
- Por la paralelización se espera que los errores sean **menores** en el GPU.

Códigos de Python que muestran la idea: [gpu_suma_error.py](../code/threads/gpu_suma_error.py) y [gpu_producto_punto_error.py](../code/threads/gpu_producto_punto_error.py). Ambos usan **PyCUDA**, un módulo de Python que veremos pronto.

<!-- NOTA — con números de 32 bits, sumar secuencialmente millones de valores
acumula error porque el acumulador crece y cada nuevo sumando pequeño pierde
dígitos. El árbol de la reducción suma números de tamaño parecido en cada
nivel, y por eso suele tener menos error que el ciclo secuencial del host. En
reduccion-shuffle.cu la referencia del host se acumula en double, justamente
para tener un valor confiable con que comparar. -->

---

# Ejercicios

---

## **Ejercicio 1: la reducción completa**

Descargar: [reduccion-shuffle.cu](../code/threads/reduccion-shuffle.cu) y [reduccion_global8.cu](../code/threads/reduccion_global8.cu)

1. Ejecutar `reduccion-shuffle.x`. ¿Qué ancho de banda logra? Compararlo con los $300$ GB/s de la T4.
2. Medir los dos programas con `nvprof` y comparar el tiempo de sus *kernels*.
3. ¿Por qué el resultado del GPU puede cambiar en los últimos dígitos entre una ejecución y otra?

```bash
nvcc -arch=sm_75 reduccion-shuffle.cu -o reduccion-shuffle.x
nvprof ./reduccion-shuffle.x
```

<!-- RESPUESTAS — TODO: medir en la T4. (1) Lo esperable es que se acerque al
peak de 300 GB/s: la reducción solo lee cada elemento una vez y hace una suma
por cada 4 bytes, así que es puramente memory bound (roofline del capítulo 1).
(2) reduccion_global8 lee y escribe la memoria global varias veces dentro de
cada bloque y copia los parciales al host; reduccion-shuffle lee cada
elemento una sola vez. (3) atomicAdd sobre float suma los aportes de los
bloques en el orden en que terminan, que cambia entre ejecuciones; la suma en
punto flotante no es asociativa, así que el redondeo cambia. Conecta con la
diapositiva de errores. -->

---

## **Ejercicio 2: histograma y contención**

Descargar: [histograma.cu](../code/threads/histograma.cu)

1. Ejecutar el programa. Con datos **aleatorios**, ¿qué *kernel* es más rápido?
2. Con datos **constantes**, ¿qué le pasa a `histGlobal`? ¿Por qué?
3. ¿Por qué la versión con memoria compartida sufre menos con datos constantes?

<!-- RESPUESTAS — TODO: medir en la T4. (1) histCompartida debería ganar: las
atómicas en memoria compartida son mucho más baratas que en global, y al final
cada bloque hace solo 256 atómicas globales. (2) Con datos constantes los 16 M
atomicAdd globales caen sobre el MISMO contador: el hardware los serializa y
el kernel se vuelve mucho más lento (contención máxima). (3) En la versión
compartida la contención queda dentro de cada bloque (solo compiten sus 256
threads, en la memoria rápida del SM), y entre bloques hay una sola atómica
global por contador. Es la misma idea de la reducción: primero reducir
localmente, después una atómica por bloque. -->

---

# Fin Capítulo 3

## Próximo capítulo: invocación de los kernels
