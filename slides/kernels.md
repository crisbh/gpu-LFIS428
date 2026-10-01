---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

<!-- Contenido reconstruido del PDF original (kernels.pdf). Solo texto: los
     diagramas del original (p. ej. los esquemas de pipelining) se omitieron. -->

# **Programación en GPUs**
## Invocación de los kernels

Streams, eventos y sincronización

---

## **Códigos**

Los códigos de esta clase están disponibles para descargar:

- Streams: [cuda_default_stream.cu](../code/kernels/cuda_default_stream.cu) · [cuda_multi_stream.cu](../code/kernels/cuda_multi_stream.cu) · [cuda_multi_stream_with_sync.cu](../code/kernels/cuda_multi_stream_with_sync.cu) · [cuda_multi_stream_with_default.cu](../code/kernels/cuda_multi_stream_with_default.cu) · [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu) · [prioritized_cuda_stream.cu](../code/kernels/prioritized_cuda_stream.cu)
- Callbacks/eventos: [cuda_callback.cu](../code/kernels/cuda_callback.cu) · [cuda_event.cu](../code/kernels/cuda_event.cu) · [cuda_event_with_streams.cu](../code/kernels/cuda_event_with_streams.cu)
- Paralelismo dinámico: [dynamic_parallelism.cu](../code/kernels/dynamic_parallelism.cu) · [recursion.cu](../code/kernels/recursion.cu)
- OpenMP / MPI: [openmp.cu](../code/kernels/openmp.cu) · [openmp_default_stream.cu](../code/kernels/openmp_default_stream.cu) · [simpleMPI.cu](../code/kernels/simpleMPI.cu)
- Overhead: [cuda_kernel.cu](../code/kernels/cuda_kernel.cu)

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

# Eventos

---

## **Eventos de CUDA**

- Un evento es una marca en la fila de un *stream*: el GPU anota cuándo la alcanza.
- La diferencia entre dos eventos mide el tiempo **en el GPU**, sin el ruido del *host*.

```cuda
cudaEventRecord(inicio);
kernel<<<grid, bloque>>>(...);
cudaEventRecord(fin);
cudaEventSynchronize(fin);              // esperar a que el GPU llegue a "fin"
cudaEventElapsedTime(&ms, inicio, fin);
```

Ejemplo: [cuda_event.cu](../code/kernels/cuda_event.cu) (lo compara con un cronómetro del *host*).

<!-- NOTA — hasta ahora medimos con segundos(), un cronómetro del host, y
vimos que los tiempos variaban bastante entre corridas en Colab. El
cronómetro del host mide también el lanzamiento, la sincronización y
cualquier pausa del CPU virtual. Los eventos los registra el propio GPU en la
fila del stream, así que miden solo lo que pasó entre las dos marcas. Siguen
sujetos a la variabilidad de los relojes del GPU (el calentamiento sigue
haciendo falta), pero sacan del medio al host. Los vamos a usar para medir los
streams. -->

---

# Streams

---

## **Streams**

- Un *stream* es una secuencia de comandos para el GPU.
- Normalmente los *kernels* se ejecutan en el *default stream* (número 0).
- Se puede especificar el *stream* con el cuarto argumento al lanzar el *kernel*:

```cuda
kernel<<< grid_size, block_size, shared_memory, stream >>>();
```

<!-- NOTA — un stream es una fila: lo que se pone en el mismo stream se ejecuta en
orden, y lo que está en streams distintos puede ejecutarse a la vez. Hasta
ahora todo lo hicimos en el default stream sin saberlo, por eso cada kernel
esperaba al anterior. El cuarto argumento de <<< >>> es el stream; el tercero
es la memoria compartida dinámica, que vimos en transpuestaCompDin. -->

---

## **Streams**

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
foo_kernel<<< grid_size, block_size, 0, stream >>>();
cudaStreamDestroy(stream);
```

Ejemplo: [cuda_default_stream.cu](../code/kernels/cuda_default_stream.cu) → con `nvprof --print-gpu-trace` se ve en qué *stream* corre cada operación.

<!-- NOTA — con nvprof --print-gpu-trace cada operación aparece con su stream y su
hora de inicio y fin, así que se puede ver si dos operaciones se solaparon.
Colab no tiene NVVP ni nsys, así que la traza de nvprof es la herramienta. -->

---

## **Streams**

- Ejemplo: [cuda_multi_stream.cu](../code/kernels/cuda_multi_stream.cu)
  - Los *kernels* se ejecutan de forma **asincrónica** con el *host*.
  - Las operaciones de CUDA en *streams* diferentes son **independientes**.
- Ejemplo: [cuda_multi_stream_with_sync.cu](../code/kernels/cuda_multi_stream_with_sync.cu)
  - Se pueden sincronizar los *streams* con `cudaStreamSynchronize(stream)`.
  - Esta función obliga al *host* a esperar hasta que el *stream* termina.
- Ejemplo: [cuda_multi_stream_with_default.cu](../code/kernels/cuda_multi_stream_with_default.cu)
  - Todos los demás *streams* son sincrónicos con el *default stream*.
  - Para tener *streams* operando en paralelo, mejor **no** usar el *default*.

<!-- NOTA — tres ejemplos cortos. En el primero los kernels se lanzan en streams
distintos y el host no espera. En el segundo, cudaStreamSynchronize hace que
el host espere a un stream en particular. El tercero muestra la trampa: el
default stream 'antiguo' se sincroniza con todos los demás streams
bloqueantes, así que cualquier cosa lanzada ahí rompe el paralelismo. Por eso
en el pipelining nada va al default stream. -->

---

## **Streams: pipelining**

Una aplicación de los *streams*:
- Las operaciones de transferencia de datos en unos *streams* coinciden con cómputos en otros.
- Otra manera de "esconder" el *latency*.

<!-- NOTA — la idea es la de una línea de producción: si el trabajo se divide en
pedazos, la copia del pedazo 2 puede hacerse mientras se calcula el pedazo 1.
Es la misma idea de esconder latencia que el occupancy, pero a nivel de
transferencias: el PCIe es lento (unos 12 GB/s en la T4) y así se esconde
detrás del cómputo. -->

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
stream por pedazo, las etapas de pedazos distintos se superponen: el motor de
copia y los SMs trabajan a la vez. El modelo supone que las tres etapas duran
lo mismo; en la realidad manda la etapa más larga, y el total tiende a
max(copia H->D, kernel, copia D->H) más el llenado y vaciado del pipeline. Las
dos copias solo se solapan entre sí si el GPU tiene más de un motor de copia
(cuda_pipelining.x imprime asyncEngineCount). -->

---

## **Streams: pipelining**

**3 requerimientos:**
- La memoria en el *host* debe ser ***pinned***: `cudaMallocHost()`.
- La transferencia de datos debe ser asincrónica: `cudaMemcpyAsync()`.
- Las operaciones de CUDA deben estar en *streams* diferentes (y nada en el *default stream*).

Ejemplo: [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu). Mide el mismo trabajo con $1$ *stream* y con varios.

<!-- NOTA — acá se cierra lo de la memoria pinned del capítulo de memoria: era
opcional para cudaMemcpy (solo más rápida), pero para solapar es obligatoria.
Desde memoria paginable, el driver tiene que pasar los datos por su buffer
pinned propio, y cudaMemcpyAsync se comporta como una copia sincrónica: el
pipeline se rompe. Es la pregunta 3 del ejercicio. -->

---

## **Eventos y streams**

- Ejemplo: [cuda_event_with_streams.cu](../code/kernels/cuda_event_with_streams.cu): eventos en varios *streams*.
- Los eventos también sirven para **sincronizar**: `cudaStreamWaitEvent(stream, evento)` hace que un *stream* espere a que otro llegue a un evento, sin bloquear al *host*.

<!-- NOTA — cudaStreamWaitEvent permite dependencias entre streams sin bloquear al
host: el stream B espera a que el stream A llegue a un evento, y el host sigue
lanzando trabajo. Es la forma de expresar 'este kernel necesita el resultado
de aquel otro' sin volver a un solo stream. -->

---

## **CUDA Callback**

- Una función de *callback* es una función llamada por el *host* en algún momento durante la ejecución del *stream*.
- Es útil para obtener información sobre el estatus de un *stream*.
- Ejemplo: [cuda_callback.cu](../code/kernels/cuda_callback.cu) — para obtener información del tiempo de ejecución de cada *stream*.
- Según la documentación de CUDA:
  - el uso de *callbacks* es **obsoleto**;
  - la alternativa es `cudaLaunchHostFunc`, que agrega una función del *host* a la "fila" del *stream*.

<!-- NOTA — cudaStreamAddCallback está obsoleto; la función actual es
cudaLaunchHostFunc, que encola una función del host en la fila del stream.
Sirve para avisar al host que algo terminó, pero no para medir tiempos con
precisión: corre en el host, así que su hora incluye la latencia de la
notificación. Para medir, eventos. -->

---

## **Stream priority**

- Se puede asociar una **prioridad** a los *streams*.
- Ejemplo: [prioritized_cuda_stream.cu](../code/kernels/prioritized_cuda_stream.cu)
  - `cudaDeviceGetStreamPriorityRange()`
  - `cudaStreamCreateWithPriority()`
    - sincrónico con el *default*: `cudaStreamDefault`
    - **no** sincrónico: `cudaStreamNonBlocking`

<!-- NOTA — la prioridad no interrumpe un kernel que ya está corriendo: cuando se
liberan recursos, el planificador prefiere los bloques de streams de mayor
prioridad. Importa cuando hay trabajo urgente y chico (por ejemplo, de
comunicación) compitiendo con kernels largos. En los números de prioridad,
menor es más prioritario. -->

---

## **Synchronization**

- **Sincronizar todo:**
  - `cudaDeviceSynchronize()` — bloquea el *host* hasta que todas las instrucciones de CUDA terminen.
- **Sincronizar respecto a un *stream*:**
  - `cudaStreamSynchronize(stream)` — bloquea el *host* hasta que terminen las instrucciones de ese *stream*.
- **Sincronizar con eventos:**
  - `cudaEventRecord(event, stream)`
  - `cudaEventSynchronize(event)`
  - `cudaStreamWaitEvent(stream, event)`
  - `cudaEventQuery(event)`

<!-- NOTA — resumen de las tres granularidades, de la más gruesa a la más fina:
todo el device, un stream, o un evento. La regla práctica es sincronizar lo
menos posible y lo más localmente posible: cudaDeviceSynchronize() después de
cada kernel elimina todo el paralelismo entre host y GPU. -->

---

# Ejercicios

---

## **Ejercicio 1: solapar copias y cómputo**

Descargar: [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu)

1. Ejecutar con $2$, $4$ y $8$ *streams* (`./cuda_pipelining.x 8`). ¿Cuánto se gana?
2. Con `nvprof --print-gpu-trace`, ¿qué operaciones se solapan?
3. Reemplazar `cudaMallocHost` por `malloc` (y `cudaFreeHost` por `free`). ¿Qué pasa?
4. ¿Cuál es la mayor ganancia posible?

```bash
nvcc -arch=sm_75 cuda_pipelining.cu -o cuda_pipelining.x
nvprof --print-gpu-trace ./cuda_pipelining.x 4
```

<!-- RESPUESTAS — TODO: medir en la T4. (1) Con más streams el tiempo baja,
pero la ganancia se satura: pasado cierto número de pedazos, cada uno es tan
chico que el costo fijo de cada copia y lanzamiento pesa más. (2) En la traza,
las copias de un stream aparecen al mismo tiempo que el kernel de otro (las
columnas de inicio y duración se superponen). (3) Con memoria paginable
cudaMemcpyAsync deja de ser asincrónica: las copias pasan por el buffer
pinned del driver y el pipeline se rompe, así que el tiempo vuelve a ser el
de 1 stream o peor (las copias además son más lentas: 4.7 contra 12.4 GB/s en
el ejemplo de memoria pinned). (4) Si las tres etapas duraran lo mismo, la
ganancia tendería a 3x; en la práctica la limita la etapa más larga: el
tiempo total no puede bajar de la duración de esa etapa. -->

---

# Apéndice: para estudiar por cuenta propia

<!-- NOTA — lo que sigue (paralelismo dinámico, CUDA con OpenMP, MPS y el
costo de lanzar kernels) queda como referencia, como dice la nota de ritmo
del calendario. MPS además necesita sudo, así que no se puede probar en
Colab. -->

---

## **CUDA Dynamic Parallelism**

- Lanzar *kernels* dentro de un *kernel*:
  - permite el uso de *child grids*;
  - algoritmos recursivos;
  - *grids* adaptativos (¡simulaciones!).
- Ejemplos: [dynamic_parallelism.cu](../code/kernels/dynamic_parallelism.cu) · [recursion.cu](../code/kernels/recursion.cu)

<!-- NOTA — un kernel puede lanzar kernels hijos desde el GPU, útil para problemas
cuya forma se descubre durante el cálculo (mallas adaptativas, recursión).
Requiere compilar con -rdc=true -lcudadevrt. En la práctica tiene un costo de
lanzamiento alto y conviene usarlo poco; las diapositivas de overhead lo
muestran. -->

---

## **CUDA / OpenMP**

- Se puede combinar CUDA con OpenMP: lanzar *kernels* desde distintos *threads* de OpenMP.
- Ejemplo: [openmp.cu](../code/kernels/openmp.cu) — cada *stream* corresponde a un *thread* de OpenMP.
- Ejemplo: [openmp_default_stream.cu](../code/kernels/openmp_default_stream.cu) — cada *thread* de OpenMP lanza el *default stream*.

<!-- NOTA — cada thread de OpenMP del host lanza su propio trabajo, idealmente en
su propio stream; si todos usan el default stream se serializan, que es lo que
contrasta el segundo ejemplo. Se compila con nvcc -Xcompiler -fopenmp.
Necesita helper_timer.h (ver Compilar los ejemplos). -->

---

# MPS (Multi-Process Service)

---

## **MPS**

- Se pueden ejecutar *kernels* de **procesos distintos**.
- Por la manera en que los procesos interactúan con el GPU, en la práctica los *kernels* se ejecutan de forma **secuencial**.
- Con el modo **Multi-Process Service (MPS)** se puede tener ejecución **simultánea** de los *kernels* de procesos distintos.
  - Así podemos combinar **MPI** con CUDA.
- MPS está disponible solamente en Linux.

<!-- NOTA — sin MPS, cada proceso tiene su propio contexto en el GPU y el GPU
alterna entre contextos: los kernels de procesos distintos no se ejecutan a la
vez. MPS junta los procesos en un solo contexto. Importa en clusters donde
varios procesos MPI comparten un GPU. -->

---

## **MPS**

Funciona como un *daemon* (un programa que corre en el *background*):
- todos los procesos mandan sus comandos al *daemon* de MPS;
- MPS manda los comandos de CUDA al GPU;
- para el GPU hay un solo proceso ocupando sus recursos (MPS).

<!-- NOTA — el daemon es un intermediario: los procesos le mandan su trabajo y él
lo entrega al GPU como si fuera un solo proceso. -->

---

## **MPS**

Ejemplo: [simpleMPI.cu](../code/kernels/simpleMPI.cu)
- modificación del programa con OpenMP;
- cada proceso de MPI lanza varios *threads* de OpenMP, y cada *thread* lanza un *kernel* en el GPU.

Primero, sin MPS...

<!-- NOTA — simpleMPI.cu necesita MPI y los headers de los samples; no corre en
Colab. Queda como lectura. -->

---

## **MPS**

Ahora inicializamos MPS:

```sh
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c 3
sudo nvidia-cuda-mps-control -d
```

El programa debería (en principio) ejecutar más rápido, ya que puede aprovechar el uso concurrente del GPU en cada proceso de MPI.

<!-- NOTA — los comandos necesitan sudo: nvidia-smi -c 3 pone el GPU en modo de
proceso exclusivo y nvidia-cuda-mps-control -d arranca el daemon. En Colab no
hay permisos para esto. -->

---

## **MPS**

Desactivamos MPS con:

```sh
echo "quit" | sudo nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c 0
```

<!-- NOTA — hay que acordarse de apagar MPS y devolver el GPU al modo por defecto
(-c 0); si no, otros usuarios del mismo GPU quedan afectados. -->

---

## **Kernel execution overhead**

Ejemplo: [cuda_kernel.cu](../code/kernels/cuda_kernel.cu) — operación de SAXPY de tres formas:

1. llamar un *kernel* dentro de un ciclo (`simple_saxpy_kernel`);
2. poner el ciclo dentro del *kernel* (`iterative_saxpy_kernel`);
3. un *kernel* recursivo (`recursive_saxpy_kernel`).

<!-- NOTA — cuda_kernel.cu compara tres formas de repetir SAXPY: lanzar el kernel
muchas veces desde un ciclo del host, poner el ciclo dentro del kernel, o usar
recursión con paralelismo dinámico. -->

---

## **Kernel execution overhead**

- Hay *overhead* al ejecutar un *kernel* dentro de un *loop* (asignación de recursos, etc.).
- También para la recursión (típicamente más).
- El caso de tener el **ciclo dentro del *kernel*** es el más rápido.

<!-- NOTA — cada lanzamiento de kernel cuesta del orden de microsegundos, aunque el
kernel no haga nada. Si el trabajo de cada lanzamiento es chico, ese costo
domina. Por eso conviene poner el ciclo dentro del kernel, con la misma lógica
que el grid-stride loop de la clase de threads: menos lanzamientos, más
trabajo por lanzamiento. La recursión con paralelismo dinámico es la más cara. -->

---

# ¡Gracias!

## Próxima clase: librerías de CUDA y Python
