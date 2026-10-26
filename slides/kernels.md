---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

# **Programación en GPUs**
## Invocación de los kernels

Streams, eventos y sincronización

<!-- ## **Códigos** -->
<!---->
<!-- Los códigos de esta clase están disponibles para descargar: -->
<!---->
<!-- - Streams: [streams-concurrencia.cu](../code/kernels/streams-concurrencia.cu) · [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu) · [prioritized_cuda_stream.cu](../code/kernels/prioritized_cuda_stream.cu) -->
<!-- - Callbacks/eventos: [cuda_callback.cu](../code/kernels/cuda_callback.cu) · [cuda_event.cu](../code/kernels/cuda_event.cu) · [cuda_event_with_streams.cu](../code/kernels/cuda_event_with_streams.cu) -->
<!-- - Paralelismo dinámico: [dynamic_parallelism.cu](../code/kernels/dynamic_parallelism.cu) · [recursion.cu](../code/kernels/recursion.cu) -->
<!-- - OpenMP / MPI: [openmp.cu](../code/kernels/openmp.cu) · [openmp_default_stream.cu](../code/kernels/openmp_default_stream.cu) · [simpleMPI.cu](../code/kernels/simpleMPI.cu) -->
<!-- - Overhead: [cuda_kernel.cu](../code/kernels/cuda_kernel.cu) -->

<!-- NOTA — los programas de la clase (cuda_event.cu, streams-concurrencia.cu
y cuda_pipelining.cu) están escritos para este curso: en español, sin
dependencias externas, y se compilan con nvcc -arch=sm_75. Los del apéndice
vienen del material original; varios necesitan los headers de los CUDA
samples (ver Compilar los ejemplos del apéndice). -->
<!-- ## **Códigos** -->
<!---->
<!-- Los códigos de esta clase están disponibles para descargar: -->
<!---->
<!-- - Streams: [cuda_default_stream.cu](../code/kernels/cuda_default_stream.cu) · [cuda_multi_stream.cu](../code/kernels/cuda_multi_stream.cu) · [cuda_multi_stream_with_sync.cu](../code/kernels/cuda_multi_stream_with_sync.cu) · [cuda_multi_stream_with_default.cu](../code/kernels/cuda_multi_stream_with_default.cu) · [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu) · [prioritized_cuda_stream.cu](../code/kernels/prioritized_cuda_stream.cu) -->
<!-- - Callbacks/eventos: [cuda_callback.cu](../code/kernels/cuda_callback.cu) · [cuda_event.cu](../code/kernels/cuda_event.cu) · [cuda_event_with_streams.cu](../code/kernels/cuda_event_with_streams.cu) -->
<!-- - Paralelismo dinámico: [dynamic_parallelism.cu](../code/kernels/dynamic_parallelism.cu) · [recursion.cu](../code/kernels/recursion.cu) -->
<!-- - OpenMP / MPI: [openmp.cu](../code/kernels/openmp.cu) · [openmp_default_stream.cu](../code/kernels/openmp_default_stream.cu) · [simpleMPI.cu](../code/kernels/simpleMPI.cu) -->
<!-- - Overhead: [cuda_kernel.cu](../code/kernels/cuda_kernel.cu) -->

<!-- NOTA — solo cuda_event.cu y cuda_pipelining.cu se reescribieron para este
curso (en español, sin dependencias externas y con un kernel que hace trabajo
real). El resto viene del material original y necesita los headers de los CUDA
samples: ver la diapositiva siguiente. -->

---

## **Compilar los ejemplos**

Los programas que incluyen `helper_timer.h` (todos menos `cuda_event.cu` y `cuda_pipelining.cu`) necesitan los *headers* de los CUDA *samples*:

```bash
git clone --depth 1 https://github.com/NVIDIA/cuda-samples
nvcc -arch=sm_75 -I cuda-samples/Common programa.cu -o programa.x
```

<!-- NOTA — helper_timer.h es un archivo de los ejemplos oficiales de NVIDIA (cuda-
samples), no del toolkit, así que Colab no lo trae. Clonar el repositorio una
vez por sesión basta; --depth 1 evita bajar toda la historia.
dynamic_parallelism.cu y recursion.cu además necesitan -rdc=true -lcudadevrt,
y simpleMPI.cu necesita MPI, que Colab no trae instalado. -->


---

## **Compilar los ejemplos**

Los programas que incluyen `helper_timer.h` (todos menos `cuda_event.cu` y `cuda_pipelining.cu`) necesitan los *headers* de los CUDA *samples*:

```bash
git clone --depth 1 https://github.com/NVIDIA/cuda-samples
nvcc -arch=sm_75 -I cuda-samples/Common programa.cu -o programa.x
```

<!-- NOTA — helper_timer.h es un archivo de los ejemplos oficiales de NVIDIA (cuda-
samples), no del toolkit, así que Colab no lo trae. Clonar el repositorio una
vez por sesión basta; --depth 1 evita bajar toda la historia.
dynamic_parallelism.cu y recursion.cu además necesitan -rdc=true -lcudadevrt,
y simpleMPI.cu necesita MPI, que Colab no trae instalado. -->

---

# Streams

---

## **Default stream**

- Cuando el *host* lanza un kernel, este retoma el control del programa sin esperar a que termine de ejecutarse.
- Si el *host* lanza un segundo kernel, por *default* su ejecución tiene que esperar.
  - Esto ocurre incluso si **el segundo kernel no depende de los resultados del primero**.

```cuda
cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
kernelA<<<grid, bloque>>>(d_x);
kernelB<<<grid, bloque>>>(d_y);   // no usa nada de kernelA
cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost);
```

---

## **Streams**

- Para lograr paralelismo a nivel de kernels (y otras operaciones de CUDA) usamos los *streams*.
- Un *stream* es una 'fila' de comandos para el GPU: copias, *kernels*, ...
- Si no decimos nada, todo va al *default stream* (número 0).

<table class="timeline">
<tr><th><em>default stream</em></th><td colspan="4" style="border:none;background:transparent !important"></td></tr>
<tr><th>GPU</th><td class="s3">H→D</td><td class="s3">kernelA</td><td class="s3">kernelB</td><td class="s3">D→H</td></tr>
<tr><th>con <em>streams</em></th><td colspan="4" style="border:none;background:transparent !important"></td></tr>
<tr><th>stream 1</th><td class="s1">H→D</td><td class="s1">kernelA</td><td class="s1">D→H</td><td></td></tr>
<tr><th>stream 2</th><td class="s2">kernelB</td><td></td><td></td><td></td></tr>
</table>

- Lo que está en el mismo *stream* se ejecuta **en orden**; lo que está en *streams* distintos **puede** ejecutarse a la vez.

<!-- NOTA — es la misma idea del occupancy y de esconder la latencia, un nivel
más arriba: allá, más warps esconden la espera de la memoria; acá, más
operaciones independientes llenan los SMs y los motores de copia que quedan
ociosos: mientras se copia, los SMs no hacen nada, y un kernel con pocos
bloques deja SMs libres. kernelB no usa nada de kernelA ni de la copia, así que nada obliga a
que espere; en el default stream espera igual. Ojo: dos kernels solo corren a
la vez si hay recursos libres. Un kernel que llena el GPU no deja espacio para
otro, así que la ganancia más clara es solapar copias con cálculo (el
pipelining, más adelante). La T4 admite hasta 128 grids residentes a la vez
(compute capability 7.5). -->

---

## **Streams**

- El *stream* es el cuarto argumento al lanzar el *kernel*:

```cuda
kernel<<< grid_size, block_size, shared_memory, stream >>>();
```

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
foo_kernel<<< grid_size, block_size, 0, stream >>>();
cudaMemcpyAsync(d_x, h_x, bytes, cudaMemcpyHostToDevice, stream);
cudaStreamDestroy(stream);
```

- Las copias también van a un *stream*: `cudaMemcpyAsync` recibe el *stream* como último argumento.
- `cudaMemcpy` (sin `Async`) usa el *default stream* y bloquea al *host*.

<!-- NOTA — cudaStreamDestroy no corta el trabajo pendiente: el stream se destruye
cuando termina lo que tiene en la fila. cudaMemcpyAsync necesita memoria
pinned para ser de verdad asincrónica; lo vemos con el pipelining. -->

---

## **Ejecución asincrónica**

<table class="timeline">
<tr><th>Host</th><td class="s1">lanza 1</td><td class="s2">lanza 2</td><td>CPU</td><td>CPU</td><td class="espera"></td><td class="espera"></td><td class="espera"></td><td>sigue</td></tr>
<tr><th>GPU</th><td></td><td class="s1">kernel 1</td><td class="s1"></td><td class="s1"></td><td class="s2">kernel 2</td><td class="s2"></td><td class="s2"></td><td></td></tr>
</table>

- Lanzar un *kernel* devuelve el control al *host* **de inmediato**, en cualquier *stream* (también en el *default*).
- El *host* sigue con su trabajo mientras el GPU calcula.
- Solo espera cuando se lo pedimos (rayado): `cudaDeviceSynchronize()`, `cudaStreamSynchronize()`, `cudaEventSynchronize()`, o un `cudaMemcpy`.

<!-- NOTA — esta es la primera forma de paralelismo que dan los streams, y existe
aunque usemos uno solo: host y GPU trabajan a la vez. Lanzar un kernel tarda
unos microsegundos; el kernel puede tardar milisegundos. Por eso el
cronómetro del host, sin sincronizar, mide solo el lanzamiento: es el problema
que resuelven los eventos (diapositiva siguiente). streams-concurrencia.cu lo
mide: imprime cuánto tardó el host en lanzar los 5 kernels del caso 2 contra
cuánto tardó el GPU en terminarlos. -->

---

## **Eventos de CUDA**

Como lanzar no espera, para medir un *kernel* hace falta que **el GPU** anote la hora: un evento es una marca en la fila de un *stream*. `cudaEventRecord(e)` la pone en el *default*; `cudaEventRecord(e, s)`, en el *stream* `s`.

@include[cuda]{static/code/kernels/cuda_event.cu:66-72}

`cudaEventElapsedTime(&ms, inicio, fin)` da la diferencia en ms. 

---

## **Eventos de CUDA**

Ejemplo: [cuda_event.cu](../code/kernels/cuda_event.cu).

<!-- NOTA — la diapositiva anterior muestra el problema: el host recupera el
control apenas lanza, así que un cronómetro del host sin sincronizar mide solo
el lanzamiento, y sincronizando mide además la espera y cualquier pausa del
CPU virtual (por eso variaban tanto los tiempos del capítulo de memoria,
medidos con segundos()). Los eventos los registra el propio GPU en la fila del
stream, así que miden solo lo que pasó entre las dos marcas. Ya los usamos sin
explicarlos: los programas de la reducción imprimen un "tiempo GPU" medido
así, y cuda_event.cu compara los dos cronómetros. Siguen sujetos a la
variabilidad de los relojes del GPU (el calentamiento sigue haciendo falta).
Ojo con cudaEventSynchronize(fin): sin esa espera, el host leería el evento
antes de que el GPU llegue a él. -->

---

## **El *default stream***

<div class="lado-a-lado">
<table class="timeline">
<tr><th colspan="6" style="text-align:center">1. todo en el <em>default</em></th></tr>
<tr><th>default</th><td class="s3">0</td><td class="s3">1</td><td class="s3">2</td><td class="s3">3</td><td class="s3">4</td></tr>
</table>
<table class="timeline">
<tr><th colspan="6" style="text-align:center">2. un <em>stream</em> por <em>kernel</em></th></tr>
<tr><th>s0</th><td class="s1">0</td><td></td><td></td><td></td><td></td></tr>
<tr><th>s1</th><td class="s1">1</td><td></td><td></td><td></td><td></td></tr>
<tr><th>s2</th><td class="s1">2</td><td></td><td></td><td></td><td></td></tr>
<tr><th>s3</th><td class="s1">3</td><td></td><td></td><td></td><td></td></tr>
<tr><th>s4</th><td class="s1">4</td><td></td><td></td><td></td><td></td></tr>
</table>
<table class="timeline">
<tr><th colspan="6" style="text-align:center">4. el <em>kernel</em> 3 en el <em>default</em></th></tr>
<tr><th>s0</th><td class="s1">0</td><td></td><td></td><td></td><td></td></tr>
<tr><th>s1</th><td class="s1">1</td><td></td><td></td><td></td><td></td></tr>
<tr><th>s2</th><td class="s1">2</td><td></td><td></td><td></td><td></td></tr>
<tr><th>default</th><td></td><td class="s3">3</td><td></td><td></td><td></td></tr>
<tr><th>s4</th><td></td><td></td><td class="s1">4</td><td></td><td></td></tr>
</table>
</div>

- El *default stream* espera a todos los *streams* creados con `cudaStreamCreate`, y ellos lo esperan a él: es una **barrera**.
- Para evitarlo: `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)`, o simplemente no usar el *default*.

---

## **El *default stream***

- Ejemplo: [streams-concurrencia.cu](../code/kernels/streams-concurrencia.cu): cinco formas de lanzar los mismos $5$ *kernels* (ejercicio 1).

<!-- NOTA — cada casilla es un kernel que dura T (el kernel esperar() del
programa solo da vueltas durante unos 20 ms, así que no compite por
recursos). Caso 1: todo en una sola fila, 5T. Caso 2: cinco filas
independientes, T. Caso 4: los kernels 0-2 corren juntos; el 3, en el default
stream, espera a que terminen; el 4 espera al 3. Total 3T, aunque solo UN
kernel cambió de stream. Esto es el "legacy default stream", el
comportamiento por omisión. Hay dos salidas: streams non-blocking (caso 5 del
programa, vuelve a T), o compilar con nvcc --default-stream per-thread, que
le da a cada thread del host su propio default stream, normal. Los casos 3 y
5 no están en la figura: son para que los predigan en el ejercicio. -->

---

## **Sincronización**

| Función | ¿Bloquea? | Espera a |
|---|---|---|
| `cudaDeviceSynchronize()` | sí | todo el GPU |
| `cudaStreamSynchronize(s)` | sí | el *stream* `s` |
| `cudaEventSynchronize(e)` | sí | el evento `e` |
| `cudaEventQuery(e)` | no | (solo pregunta) |
| `cudaStreamWaitEvent(s,e)` | no | `s` espera a `e` |
| `cudaMemcpy(...)` | sí | la copia |

Regla práctica: sincronizar **lo menos posible** y **lo más local** posible.

<!-- NOTA — de la más gruesa a la más fina: todo el device, un stream, un
evento. "Bloquea" se refiere al host. cudaStreamSynchronize espera lo lanzado
antes en ese stream; cudaMemcpy espera la copia y, por ir en el default
stream, todo lo lanzado antes en los streams bloqueantes. cudaDeviceSynchronize() después de cada kernel elimina todo el
paralelismo entre host y GPU, y entre streams. cudaEventQuery devuelve
cudaSuccess si el evento ya se cumplió y cudaErrorNotReady si no: sirve para
que el host haga otra cosa mientras espera. cudaStreamWaitEvent es la única de
la tabla que no involucra al host: la vemos al final, con las dependencias
entre streams. El cudaMemcpy de la última fila es la sincronización
"escondida" que tenían todos nuestros programas hasta ahora. -->

---

## **Streams: pipelining**

Una aplicación de los *streams*:
- Las transferencias de datos en unos *streams* coinciden con cómputos en otros.
- Otra manera de "esconder" la latencia.

<!-- NOTA — la idea es la de una línea de producción: si el trabajo se divide en
pedazos, la copia del pedazo 2 puede hacerse mientras se calcula el pedazo 1.
Es la misma idea de esconder latencia que el occupancy, pero a nivel de
transferencias: el PCIe es lento (unos 12 GB/s en la T4) y así se esconde
detrás del cómputo. -->

---

## **Streams y motores del GPU**

<svg viewBox="0 0 1000 300" width="1000" style="display:block;margin:0 auto" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-motores" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<text x="250" y="26" font-size="20" font-weight="bold" text-anchor="middle" fill="#0288d1">Host: las filas de los streams</text>
<text x="815" y="26" font-size="20" font-weight="bold" text-anchor="middle" fill="#0288d1">GPU</text>
<text x="100" y="86" font-size="16" text-anchor="end" fill="#333">stream 1</text>
<rect x="110" y="55" width="90" height="50" rx="3" fill="#a5d6a7" stroke="#555"/>
<text x="155" y="86" font-size="15" text-anchor="middle" fill="#222">H→D</text>
<rect x="205" y="55" width="90" height="50" rx="3" fill="#a5d6a7" stroke="#555"/>
<text x="250" y="86" font-size="15" text-anchor="middle" fill="#222">kernel</text>
<rect x="300" y="55" width="90" height="50" rx="3" fill="#a5d6a7" stroke="#555"/>
<text x="345" y="86" font-size="15" text-anchor="middle" fill="#222">D→H</text>
<line x1="395" y1="80" x2="466" y2="80" stroke="#444" stroke-width="2" marker-end="url(#fl-motores)"/>
<text x="100" y="166" font-size="16" text-anchor="end" fill="#333">stream 2</text>
<rect x="110" y="135" width="90" height="50" rx="3" fill="#90caf9" stroke="#555"/>
<text x="155" y="166" font-size="15" text-anchor="middle" fill="#222">H→D</text>
<rect x="205" y="135" width="90" height="50" rx="3" fill="#90caf9" stroke="#555"/>
<text x="250" y="166" font-size="15" text-anchor="middle" fill="#222">kernel</text>
<rect x="300" y="135" width="90" height="50" rx="3" fill="#90caf9" stroke="#555"/>
<text x="345" y="166" font-size="15" text-anchor="middle" fill="#222">D→H</text>
<line x1="395" y1="160" x2="466" y2="160" stroke="#444" stroke-width="2" marker-end="url(#fl-motores)"/>
<text x="100" y="246" font-size="16" text-anchor="end" fill="#333">stream 3</text>
<rect x="110" y="215" width="90" height="50" rx="3" fill="#ffcc80" stroke="#555"/>
<text x="155" y="246" font-size="15" text-anchor="middle" fill="#222">H→D</text>
<rect x="205" y="215" width="90" height="50" rx="3" fill="#ffcc80" stroke="#555"/>
<text x="250" y="246" font-size="15" text-anchor="middle" fill="#222">kernel</text>
<rect x="300" y="215" width="90" height="50" rx="3" fill="#ffcc80" stroke="#555"/>
<text x="345" y="246" font-size="15" text-anchor="middle" fill="#222">D→H</text>
<line x1="395" y1="240" x2="466" y2="240" stroke="#444" stroke-width="2" marker-end="url(#fl-motores)"/>
<rect x="470" y="55" width="130" height="210" rx="3" fill="#ffe082" stroke="#555"/>
<text x="535" y="156" font-size="16" text-anchor="middle" fill="#222">planificador</text>
<text x="535" y="177" font-size="16" text-anchor="middle" fill="#222">del GPU</text>
<line x1="602" y1="80" x2="686" y2="80" stroke="#444" stroke-width="2" marker-end="url(#fl-motores)"/>
<line x1="602" y1="160" x2="686" y2="160" stroke="#444" stroke-width="2" marker-end="url(#fl-motores)"/>
<line x1="602" y1="240" x2="686" y2="240" stroke="#444" stroke-width="2" marker-end="url(#fl-motores)"/>
<rect x="690" y="55" width="250" height="50" rx="3" fill="#eef3df" stroke="#555"/>
<text x="815" y="86" font-size="16" text-anchor="middle" fill="#222">motor de copia H→D</text>
<rect x="690" y="135" width="250" height="50" rx="3" fill="#eef3df" stroke="#555"/>
<text x="815" y="166" font-size="16" text-anchor="middle" fill="#222">SMs: kernels</text>
<rect x="690" y="215" width="250" height="50" rx="3" fill="#eef3df" stroke="#555"/>
<text x="815" y="246" font-size="16" text-anchor="middle" fill="#222">motor de copia D→H</text>
</svg>

- Dentro de un *stream*, en orden. Entre *streams*, cada motor puede atender a uno distinto **al mismo tiempo**.
- `asyncEngineCount`: cuántos motores de copia tiene el GPU (`cuda_pipelining.x` lo imprime).

<!-- NOTA — el GPU tiene hardware separado para copiar y para calcular: los
motores de copia (DMA) mueven datos por el PCIe sin ocupar los SMs. Con un
solo stream, ese hardware se usa por turnos: mientras se copia, los SMs no
hacen nada, y viceversa. Con varios streams, el planificador puede darle el
motor de copia a un stream mientras otro ocupa los SMs. Si el GPU tiene un
solo motor de copia, las copias H->D y D->H no se solapan entre sí; con dos o
más, sí. En la T4 deviceQuery suele reportar 3 (sin verificar en nuestro
Colab: lo dice la primera línea de cuda_pipelining.x). -->

---

## **Streams: pipelining**

Copiar $\to$ calcular $\to$ copiar de vuelta, con los datos en $4$ pedazos:

<table class="timeline">
<tr><th>1 stream</th><td colspan="12" style="border:none;background:transparent !important"></td></tr>
<tr><th>Copia H→D</th><td class="s1">1</td><td class="s1">2</td><td class="s1">3</td><td class="s1">4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><th>Kernel</th><td></td><td></td><td></td><td></td><td class="s1">1</td><td class="s1">2</td><td class="s1">3</td><td class="s1">4</td><td></td><td></td><td></td><td></td></tr>
<tr><th>Copia D→H</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td class="s1">1</td><td class="s1">2</td><td class="s1">3</td><td class="s1">4</td></tr>
<tr><th>4 streams</th><td colspan="12" style="border:none;background:transparent !important"></td></tr>
<tr><th>Copia H→D</th><td class="s1">1</td><td class="s2">2</td><td class="s3">3</td><td class="s4">4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><th>Kernel</th><td></td><td class="s1">1</td><td class="s2">2</td><td class="s3">3</td><td class="s4">4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><th>Copia D→H</th><td></td><td></td><td class="s1">1</td><td class="s2">2</td><td class="s3">3</td><td class="s4">4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</table>

Con varios *streams*, mientras uno calcula otro ya está copiando: el total baja de $12$ a $6$ tiempos.

<!-- NOTA — cada casilla es el tiempo de procesar un pedazo en esa etapa. Con
un solo stream las tres etapas van una detrás de otra (12 casillas). Con un
stream por pedazo, las etapas de pedazos distintos se superponen: cada fila
de la tabla es uno de los motores de la diapositiva anterior. El modelo
supone que las tres etapas duran lo mismo; en la realidad manda la etapa más
larga, y el total tiende a max(copia H->D, kernel, copia D->H) más el llenado
y vaciado del pipeline. En cuda_pipelining.cu la copia H->D mueve a y b, el
doble de bytes que la D->H. Las dos copias solo se solapan entre sí si el GPU
tiene más de un motor de copia. -->

---

## **Streams: pipelining**

@include[cuda]{static/code/kernels/cuda_pipelining.cu:78-92}

<!-- NOTA — es el ciclo de procesar() en cuda_pipelining.cu: un pedazo por
stream, y para cada pedazo las tres etapas en su stream. Los
cudaEventRecord marcan cuándo termina cada etapa: con ellos el programa
imprime una línea de tiempo como la de la diapositiva anterior. Las operaciones se
lanzan en orden de stream (todo el stream 0, después todo el 1...), y el
host termina el ciclo casi de inmediato: todo es asincrónico. Los eventos
solo marcan el FIN de cada etapa; el comienzo el programa lo deduce (una
etapa empieza cuando terminó la anterior de su stream y la misma etapa del
stream anterior), porque un evento puesto antes de una copia se cumple apenas
el stream llega a él, aunque la copia todavía espere al motor. -->

---

## **Streams: pipelining**

**3 requisitos:**
- La memoria en el *host* debe ser ***pinned***: `cudaMallocHost()`.
- La transferencia de datos debe ser asincrónica: `cudaMemcpyAsync()`.
- Las operaciones de CUDA deben estar en *streams* diferentes (y nada en el *default stream*).

Ejemplo: [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu). Mide el mismo trabajo con $1$ *stream* y con varios (ejercicio 2).

<!-- NOTA — acá se cierra lo de la memoria pinned del capítulo de memoria: era
opcional para cudaMemcpy (solo más rápida), pero para solapar es obligatoria.
Desde memoria paginable, el driver tiene que pasar los datos por su buffer
pinned propio, y cudaMemcpyAsync se comporta como una copia sincrónica: el
pipeline se rompe. Es la pregunta 3 del ejercicio. Lo del default stream es
la barrera de hace unas diapositivas. -->

---

## **Dependencias entre streams**

```cuda
kernelA<<<grid, bloque, 0, A>>>(...);
cudaEventRecord(e, A);          // marca en el stream A
cudaStreamWaitEvent(B, e);      // B espera la marca; el host no
kernelB<<<grid, bloque, 0, B>>>(...);  // usa el resultado de kernelA
```

<svg viewBox="0 0 1000 215" width="1000" style="display:block;margin:0 auto" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-espera" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#c62828"/></marker><pattern id="rayas-espera" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#f4f4f4"/><rect width="4" height="8" fill="#ddd"/></pattern></defs>
<text x="115" y="52" font-size="16" text-anchor="end" fill="#333">stream A</text>
<rect x="130" y="25" width="330" height="44" rx="3" fill="#a5d6a7" stroke="#555"/>
<text x="295" y="52" font-size="15" text-anchor="middle" fill="#222">kernelA</text>
<line x1="462" y1="15" x2="462" y2="79" stroke="#c62828" stroke-width="3"/>
<text x="472" y="22" font-size="16" font-weight="bold" fill="#c62828">e</text>
<line x1="462" y1="80" x2="462" y2="100" stroke="#c62828" stroke-width="2" marker-end="url(#fl-espera)"/>
<text x="115" y="132" font-size="16" text-anchor="end" fill="#333">stream B</text>
<rect x="130" y="105" width="150" height="44" rx="3" fill="#90caf9" stroke="#555"/>
<text x="205" y="132" font-size="15" text-anchor="middle" fill="#222">otro trabajo</text>
<rect x="280" y="105" width="182" height="44" rx="3" fill="url(#rayas-espera)" stroke="#555"/>
<text x="371" y="132" font-size="15" text-anchor="middle" fill="#222">espera a e</text>
<rect x="462" y="105" width="300" height="44" rx="3" fill="#90caf9" stroke="#555"/>
<text x="612" y="132" font-size="15" text-anchor="middle" fill="#222">kernelB</text>
<text x="115" y="187" font-size="16" text-anchor="end" fill="#333">host</text>
<rect x="130" y="167" width="120" height="30" rx="3" fill="#ffe082" stroke="#555"/>
<text x="190" y="187" font-size="14" text-anchor="middle" fill="#222">lanza todo</text>
<text x="265" y="187" font-size="14" font-style="italic" fill="#555">y sigue con lo suyo: no se bloquea</text>
<text x="980" y="210" font-size="13" text-anchor="end" fill="#666">tiempo →</text>
</svg>

<!-- NOTA — cudaStreamWaitEvent expresa "este kernel necesita el resultado de
aquel otro" sin volver a un solo stream y sin bloquear al host: el stream B
espera a que el stream A llegue al evento, y el host sigue lanzando trabajo.
Lo que haya en B antes de la espera (otro trabajo) corre en paralelo con
kernelA. La alternativa ingenua, cudaStreamSynchronize(A) antes de lanzar
kernelB, también es correcta pero deja al host parado. -->

---

# Ejercicios

---

## **Ejercicio 1: ¿qué se ejecuta a la vez?**

Descargar: [streams-concurrencia.cu](../code/kernels/streams-concurrencia.cu): $5$ *kernels* de duración $T$, lanzados de cinco formas (ver el comentario al inicio del código).

1. **Antes de ejecutar:** predecir el tiempo de cada caso (en unidades de $T$).
2. Ejecutar y comparar. ¿Qué predicción falló, y por qué?
3. Caso 3: cada *kernel* tiene su *stream*. ¿Por qué no se solapan?
4. ¿Qué cambia entre los casos 4 y 5?
5. ¿Cuánto tarda el *host* en lanzar los $5$ *kernels* (última línea)?

<!-- RESPUESTAS — TODO: medir en la T4. Lo esperado: casos 1 a 5 = 5T, T, 5T,
3T, T. (1-2) El que suele fallar es el 4: se espera T o 2T y da 3T, porque
el default stream es una barrera en las dos direcciones (el kernel 3 espera a
0-2 y el 4 espera al 3). (3) cudaStreamSynchronize bloquea al HOST hasta que
termina ese stream, así que el kernel siguiente se lanza recién cuando el
anterior terminó: los streams distintos no sirven de nada si el host no deja
de esperar. (4) En el 5 los streams son non-blocking: no se sincronizan con
el default stream, así que el kernel 3 corre junto a los demás y el total
vuelve a T. Otra forma de lograrlo es compilar con
--default-stream per-thread. (5) Del orden de decenas de microsegundos para
5 lanzamientos, contra unos 20 ms del GPU: el host queda libre casi todo el
tiempo, y cualquier medición del host sin sincronizar mide solo eso. -->

---

## **Ejercicio 2: solapar copias y cómputo**

Descargar: [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu)

1. Ejecutar con $2$, $4$ y $8$ *streams* (`./cuda_pipelining.x 8`). ¿Cuánto se gana?
2. En la línea de tiempo que imprime el programa, ¿qué etapas se solapan? ¿Se solapan las dos copias entre sí?
3. Cambiar `cudaMallocHost` por `malloc` (y `cudaFreeHost` por `free`). ¿Qué pasa con el tiempo y con la línea de tiempo?
4. La copia H→D mueve `a` y `b`; la D→H, solo `c`. ¿Cuál es la mayor ganancia posible?

<!-- RESPUESTAS — TODO: medir en la T4. (1) Con más streams el tiempo baja,
pero la ganancia se satura: pasado cierto número de pedazos, cada uno es tan
chico que el costo fijo de cada copia y lanzamiento pesa más. (2) En la línea
de N streams, la fila del kernel empieza antes de que termine la fila H->D:
mientras el stream s calcula, el s+1 copia. Si asyncEngineCount >= 2, la
copia D->H de un pedazo también coincide con la H->D de otro. (3) Con memoria
paginable cudaMemcpyAsync deja de ser asincrónica: las copias pasan por el
buffer pinned del driver y el pipeline se rompe, así que el tiempo vuelve a
ser el de 1 stream o peor (las copias además son más lentas: 4.7 contra
12.4 GB/s en el ejemplo de memoria pinned), y las filas de la línea de tiempo
dejan de superponerse. (4) El total no puede bajar de la duración de la etapa
más larga. Si la copia H->D dura 2 unidades, la D->H 1 y el kernel k, la
ganancia máxima es (3 + k) / max(2, k): 2.5x si el kernel dura lo mismo que
la copia H->D, y menos cuanto más se desequilibren. -->

---

# Apéndice: para estudiar por cuenta propia

<!-- NOTA — lo que sigue (callbacks, prioridades, paralelismo dinámico, CUDA
con OpenMP, MPS y el costo de lanzar kernels) queda como referencia, como
dice la nota de ritmo del calendario. MPS además necesita sudo, así que no se
puede probar en Colab. -->

---

## **Compilar los ejemplos**

Los programas que incluyen `helper_timer.h` (`cuda_event_with_streams.cu`, `cuda_callback.cu`, `prioritized_cuda_stream.cu`, `openmp.cu`, `openmp_default_stream.cu` y `simpleMPI.cu`) necesitan los *headers* de los CUDA *samples*:

```bash
git clone --depth 1 https://github.com/NVIDIA/cuda-samples
nvcc -arch=sm_75 -I cuda-samples/Common programa.cu -o programa.x
```

<!-- NOTA — helper_timer.h es un archivo de los ejemplos oficiales de NVIDIA (cuda-
samples), no del toolkit, así que Colab no lo trae. Clonar el repositorio una
vez por sesión basta; --depth 1 evita bajar toda la historia.
dynamic_parallelism.cu, recursion.cu y cuda_kernel.cu además necesitan
-rdc=true -lcudadevrt (lanzan kernels desde el GPU); los OpenMP, -Xcompiler
-fopenmp; y simpleMPI.cu necesita MPI, que Colab no trae instalado. Estos
programas vienen del material original y están en inglés; varios usan como
kernel un ciclo for (i < 500) c = a + b, que el compilador reduce a una sola
suma, e imprimen un ancho de banda 4 veces mayor que el real
(3 * bufsize * sizeof(float), cuando bufsize ya está en bytes). Sirven para
ver la API, no para creerles los números. -->

---

## ***Callbacks***

- `cudaLaunchHostFunc(stream, funcion, datos)` pone una función del *host* en la fila del *stream*.
- Cuando el *stream* llega a ese punto, el *runtime* la llama desde un *thread* propio, en el *host*.
- Sirve para avisar que algo terminó; **no** para medir tiempos, y la función no puede llamar a la API de CUDA.
- Ejemplo: [cuda_callback.cu](../code/kernels/cuda_callback.cu), con la forma antigua `cudaStreamAddCallback`, hoy **obsoleta**.

<!-- NOTA — cudaStreamAddCallback está obsoleto; la función actual es
cudaLaunchHostFunc. La función corre en el host, así que su hora incluye la
latencia de la notificación: para medir, eventos. Que no pueda llamar a CUDA
es una regla de la documentación: el runtime podría quedar esperándose a sí
mismo. -->

---

## **Prioridad de los *streams***

- Se puede asociar una **prioridad** a los *streams*.
- Ejemplo: [prioritized_cuda_stream.cu](../code/kernels/prioritized_cuda_stream.cu)
  - `cudaDeviceGetStreamPriorityRange()`
  - `cudaStreamCreateWithPriority(&s, flags, prioridad)`, con `flags`:
    - `cudaStreamDefault`: se sincroniza con el *default stream*;
    - `cudaStreamNonBlocking`: no se sincroniza.

<!-- NOTA — la prioridad no interrumpe un kernel que ya está corriendo: cuando se
liberan recursos, el planificador prefiere los bloques de streams de mayor
prioridad. Importa cuando hay trabajo urgente y chico (por ejemplo, de
comunicación) compitiendo con kernels largos. En los números de prioridad,
menor es más prioritario. Los flags son los mismos de la barrera del default
stream. -->

---

## **Eventos en varios *streams***

- Ejemplo: [cuda_event_with_streams.cu](../code/kernels/cuda_event_with_streams.cu): cada *stream* mide su propio *kernel* con un par de eventos.
- El código pregunta: ¿qué pasa si se deja `cudaEventSynchronize(stop)` dentro del ciclo que lanza los *streams*?

<!-- NOTA — con cudaEventSynchronize dentro del ciclo, el host se queda esperando
el kernel de cada stream antes de lanzar las copias del siguiente: el
pipeline se pierde casi entero (solo la copia de vuelta de un stream puede
coincidir con la de ida del siguiente). Es el caso 3 del ejercicio 1, con
eventos en vez de cudaStreamSynchronize. -->

---

## **Paralelismo dinámico**

- Lanzar *kernels* dentro de un *kernel*:
  - permite el uso de *child grids*;
  - algoritmos recursivos;
  - *grids* adaptativos (¡simulaciones!).
- Ejemplos: [dynamic_parallelism.cu](../code/kernels/dynamic_parallelism.cu) · [recursion.cu](../code/kernels/recursion.cu)
- Compilar con `-rdc=true -lcudadevrt`.

<!-- NOTA — un kernel puede lanzar kernels hijos desde el GPU, útil para problemas
cuya forma se descubre durante el cálculo (mallas adaptativas, recursión).
En la práctica tiene un costo de lanzamiento alto y conviene usarlo poco; las
diapositivas de overhead lo muestran. -->

---

## **CUDA y OpenMP**

- Se puede combinar CUDA con OpenMP: lanzar *kernels* desde distintos *threads* de OpenMP.
- Ejemplo: [openmp.cu](../code/kernels/openmp.cu): cada *stream* corresponde a un *thread* de OpenMP.
- Ejemplo: [openmp_default_stream.cu](../code/kernels/openmp_default_stream.cu): todos los *threads* de OpenMP lanzan en el *default stream*.

<!-- NOTA — cada thread de OpenMP del host lanza su propio trabajo, idealmente en
su propio stream; si todos usan el default stream se serializan, que es lo que
contrasta el segundo ejemplo (la barrera del default stream otra vez). Se
compila con nvcc -Xcompiler -fopenmp. Necesita helper_timer.h. -->

---

## **MPS (*Multi-Process Service*)**

- Sin MPS, los *kernels* de **procesos distintos** no corren a la vez: el GPU se turna entre sus contextos.
- MPS es un *daemon* que junta los procesos en **un solo contexto**: así sus *kernels* sí pueden correr a la vez (útil con **MPI**). Solo en Linux.

<svg viewBox="0 0 1000 300" width="1000" style="display:block;margin:0 auto" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-mps" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<text x="250" y="24" font-size="20" font-weight="bold" text-anchor="middle" fill="#0288d1">Sin MPS</text>
<text x="750" y="24" font-size="20" font-weight="bold" text-anchor="middle" fill="#0288d1">Con MPS</text>
<rect x="60" y="40" width="110" height="40" rx="3" fill="#a5d6a7" stroke="#555"/>
<text x="115" y="65" font-size="15" text-anchor="middle" fill="#222">proceso 0</text>
<rect x="190" y="40" width="110" height="40" rx="3" fill="#90caf9" stroke="#555"/>
<text x="245" y="65" font-size="15" text-anchor="middle" fill="#222">proceso 1</text>
<rect x="320" y="40" width="110" height="40" rx="3" fill="#ffcc80" stroke="#555"/>
<text x="375" y="65" font-size="15" text-anchor="middle" fill="#222">proceso 2</text>
<line x1="115" y1="82" x2="115" y2="153" stroke="#444" stroke-width="2" marker-end="url(#fl-mps)"/>
<line x1="245" y1="82" x2="245" y2="153" stroke="#444" stroke-width="2" marker-end="url(#fl-mps)"/>
<line x1="375" y1="82" x2="375" y2="153" stroke="#444" stroke-width="2" marker-end="url(#fl-mps)"/>
<rect x="40" y="155" width="420" height="125" rx="3" fill="#eef3df" stroke="#555"/>
<text x="250" y="180" font-size="15" text-anchor="middle" fill="#222">GPU: un contexto a la vez</text>
<rect x="58" y="200" width="48" height="36" fill="#a5d6a7" stroke="#555"/>
<rect x="106" y="200" width="48" height="36" fill="#90caf9" stroke="#555"/>
<rect x="154" y="200" width="48" height="36" fill="#ffcc80" stroke="#555"/>
<rect x="202" y="200" width="48" height="36" fill="#a5d6a7" stroke="#555"/>
<rect x="250" y="200" width="48" height="36" fill="#90caf9" stroke="#555"/>
<rect x="298" y="200" width="48" height="36" fill="#ffcc80" stroke="#555"/>
<rect x="346" y="200" width="48" height="36" fill="#a5d6a7" stroke="#555"/>
<rect x="394" y="200" width="48" height="36" fill="#90caf9" stroke="#555"/>
<text x="442" y="265" font-size="13" text-anchor="end" fill="#666">tiempo →</text>
<rect x="560" y="40" width="110" height="40" rx="3" fill="#a5d6a7" stroke="#555"/>
<text x="615" y="65" font-size="15" text-anchor="middle" fill="#222">proceso 0</text>
<rect x="690" y="40" width="110" height="40" rx="3" fill="#90caf9" stroke="#555"/>
<text x="745" y="65" font-size="15" text-anchor="middle" fill="#222">proceso 1</text>
<rect x="820" y="40" width="110" height="40" rx="3" fill="#ffcc80" stroke="#555"/>
<text x="875" y="65" font-size="15" text-anchor="middle" fill="#222">proceso 2</text>
<line x1="615" y1="82" x2="615" y2="103" stroke="#444" stroke-width="2" marker-end="url(#fl-mps)"/>
<line x1="745" y1="82" x2="745" y2="103" stroke="#444" stroke-width="2" marker-end="url(#fl-mps)"/>
<line x1="875" y1="82" x2="875" y2="103" stroke="#444" stroke-width="2" marker-end="url(#fl-mps)"/>
<rect x="540" y="105" width="420" height="36" rx="3" fill="#ffe082" stroke="#555"/>
<text x="750" y="129" font-size="15" text-anchor="middle" fill="#222">daemon de MPS: un solo contexto</text>
<line x1="750" y1="143" x2="750" y2="153" stroke="#444" stroke-width="2" marker-end="url(#fl-mps)"/>
<rect x="540" y="155" width="420" height="125" rx="3" fill="#eef3df" stroke="#555"/>
<text x="750" y="180" font-size="15" text-anchor="middle" fill="#222">GPU: kernels a la vez</text>
<rect x="558" y="193" width="230" height="18" fill="#a5d6a7" stroke="#555"/>
<rect x="558" y="214" width="230" height="18" fill="#90caf9" stroke="#555"/>
<rect x="558" y="235" width="230" height="18" fill="#ffcc80" stroke="#555"/>
<text x="942" y="265" font-size="13" text-anchor="end" fill="#666">tiempo →</text>
</svg>

<!-- NOTA — sin MPS, cada proceso tiene su propio contexto en el GPU y el GPU
alterna entre contextos: aunque cada proceso use una fracción del GPU, sus
kernels no se ejecutan a la vez. MPS es un intermediario: los procesos le
mandan su trabajo y él lo entrega al GPU como si fuera un solo proceso, así
que kernels chicos de procesos distintos llenan juntos el GPU (y el total se
acorta, como en la figura). Importa en clusters donde varios procesos MPI
comparten un GPU. -->

---

## **MPS: cómo usarlo**

Ejemplo: [simpleMPI.cu](../code/kernels/simpleMPI.cu): cada proceso de MPI lanza varios *threads* de OpenMP, y cada *thread* lanza un *kernel*. Comparar el tiempo sin MPS y con MPS:

```sh
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c 3            # GPU en modo de proceso exclusivo
sudo nvidia-cuda-mps-control -d      # arrancar el daemon
# ... ejecutar el programa ...
echo "quit" | sudo nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c 0            # volver al modo por defecto
```

<!-- NOTA — los comandos necesitan sudo, así que en Colab no se puede; simpleMPI.cu
además necesita MPI y los headers de los samples. Queda como lectura. Hay que
acordarse de apagar MPS y devolver el GPU al modo por defecto (-c 0); si no,
otros usuarios del mismo GPU quedan afectados. -->

---

## **Costo de lanzar un *kernel***

Ejemplo: [cuda_kernel.cu](../code/kernels/cuda_kernel.cu): $16$ pasos de SAXPY sobre $2^{10}$ elementos, de tres formas:

1. un *kernel* por paso, lanzado desde un ciclo en el *host* (`saxpySimple`);
2. un solo *kernel* con el ciclo adentro (`saxpyIterativo`);
3. un *kernel* que se relanza desde el GPU (`saxpyRecursivo`).

```bash
nvcc -arch=sm_75 -rdc=true cuda_kernel.cu -o cuda_kernel.x -lcudadevrt
```

<!-- NOTA — las tres formas hacen exactamente las mismas cuentas, y cada paso
depende del anterior, así que el compilador no puede eliminar el ciclo. (La
versión original repetía y = alpha * x + beta, que no depende de y: el
compilador dejaba un solo paso y el "ciclo dentro del kernel" ganaba por
hacer 24 veces menos trabajo.) Con 1024 elementos el trabajo es mínimo y lo
que se mide son los lanzamientos. El programa calienta cada variante antes de
medirla y verifica el resultado. -->

---

## **Costo de lanzar un *kernel***

- Cada lanzamiento tiene un costo fijo (preparar el *grid*, repartir los bloques), aunque el *kernel* no haga casi nada.
- Si el trabajo por lanzamiento es chico, ese costo domina: el **ciclo dentro del *kernel*** es el más rápido.
- Lanzar desde el GPU (recursión) también cuesta.

<!-- NOTA — TODO: medir en la T4 y poner los tres tiempos acá. Cada lanzamiento
cuesta del orden de microsegundos. Por eso conviene poner el ciclo dentro del
kernel, con la misma lógica que el grid-stride loop de la clase de threads:
menos lanzamientos, más trabajo por lanzamiento. En la recursión cada grid
hijo espera a que se lance desde el GPU y el padre no termina hasta que
terminan todos sus descendientes. -->

---

# Fin Capítulo 4

## Próximo capítulo: librerías de CUDA y Python
