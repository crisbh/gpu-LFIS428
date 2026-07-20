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

---

## **Streams**

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
foo_kernel<<< grid_size, block_size, 0, stream >>>();
cudaStreamDestroy(stream);
```

Ejemplo: [cuda_default_stream.cu](../code/kernels/cuda_default_stream.cu) → se pueden analizar los *streams* en NVVP.

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

---

## **Streams: pipelining**

Una aplicación de los *streams*:
- Las operaciones de transferencia de datos en unos *streams* coinciden con cómputos en otros.
- Otra manera de "esconder" el *latency*.

---

## **Streams: pipelining**

**3 requerimientos:**
- La memoria en el *host* debe ser *pinned*: `cudaMallocHost()`.
- La transferencia de datos debe ser asincrónica: `cudaMemcpyAsync()`.
- Las operaciones de CUDA deben estar en *streams* diferentes (y nada en el *default stream*).

Ejemplo: [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu)
- Programa escrito en C++; usa *object-oriented programming* para organizar la ejecución de los *kernels*.

---

## **CUDA Callback**

- Una función de *callback* es una función llamada por el *host* en algún momento durante la ejecución del *stream*.
- Es útil para obtener información sobre el estatus de un *stream*.
- Ejemplo: [cuda_callback.cu](../code/kernels/cuda_callback.cu) — para obtener información del tiempo de ejecución de cada *stream*.
- Según la documentación de CUDA:
  - el uso de *callbacks* es **obsoleto**;
  - la alternativa es `cudaLaunchHostFunc`, que agrega una función del *host* a la "fila" del *stream*.

---

## **Stream priority**

- Se puede asociar una **prioridad** a los *streams*.
- Ejemplo: [prioritized_cuda_stream.cu](../code/kernels/prioritized_cuda_stream.cu)
  - `cudaDeviceGetStreamPriorityRange()`
  - `cudaStreamCreateWithPriority()`
    - sincrónico con el *default*: `cudaStreamDefault`
    - **no** sincrónico: `cudaStreamNonBlocking`

---

## **CUDA events**

- Para grabar **eventos** del lado del *device* (GPU) en el *stream*.
- Ejemplo: [cuda_event.cu](../code/kernels/cuda_event.cu)
  - para determinar el tiempo de ejecución del *kernel*;
  - el ejemplo con *callback* tiene la desventaja de que el *callback* se llama en el *host*, no en el *device*.
- Ejemplo: [cuda_event_with_streams.cu](../code/kernels/cuda_event_with_streams.cu)
  - similar, pero con varios *streams*;
  - se pueden sincronizar los *streams* con *events*.

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

---

## **CUDA Dynamic Parallelism**

- Lanzar *kernels* dentro de un *kernel*:
  - permite el uso de *child grids*;
  - algoritmos recursivos;
  - *grids* adaptativos (¡simulaciones!).
- Ejemplos: [dynamic_parallelism.cu](../code/kernels/dynamic_parallelism.cu) · [recursion.cu](../code/kernels/recursion.cu)

---

## **CUDA / OpenMP**

- Se puede combinar CUDA con OpenMP: lanzar *kernels* desde distintos *threads* de OpenMP.
- Ejemplo: [openmp.cu](../code/kernels/openmp.cu) — cada *stream* corresponde a un *thread* de OpenMP.
- Ejemplo: [openmp_default_stream.cu](../code/kernels/openmp_default_stream.cu) — cada *thread* de OpenMP lanza el *default stream*.

---

# MPS (Multi-Process Service)

---

## **MPS**

- Se pueden ejecutar *kernels* de **procesos distintos**.
- Por la manera en que los procesos interactúan con el GPU, en la práctica los *kernels* se ejecutan de forma **secuencial**.
- Con el modo **Multi-Process Service (MPS)** se puede tener ejecución **simultánea** de los *kernels* de procesos distintos.
  - Así podemos combinar **MPI** con CUDA.
- MPS está disponible solamente en Linux.

---

## **MPS**

Funciona como un *daemon* (un programa que corre en el *background*):
- todos los procesos mandan sus comandos al *daemon* de MPS;
- MPS manda los comandos de CUDA al GPU;
- para el GPU hay un solo proceso ocupando sus recursos (MPS).

---

## **MPS**

Ejemplo: [simpleMPI.cu](../code/kernels/simpleMPI.cu)
- modificación del programa con OpenMP;
- cada proceso de MPI lanza varios *threads* de OpenMP, y cada *thread* lanza un *kernel* en el GPU.

Primero, sin MPS...

---

## **MPS**

Ahora inicializamos MPS:

```sh
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c 3
sudo nvidia-cuda-mps-control -d
```

El programa debería (en principio) ejecutar más rápido, ya que puede aprovechar el uso concurrente del GPU en cada proceso de MPI.

---

## **MPS**

Desactivamos MPS con:

```sh
echo "quit" | sudo nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c 0
```

---

## **Kernel execution overhead**

Ejemplo: [cuda_kernel.cu](../code/kernels/cuda_kernel.cu) — operación de SAXPY de tres formas:

1. llamar un *kernel* dentro de un ciclo (`simple_saxpy_kernel`);
2. poner el ciclo dentro del *kernel* (`iterative_saxpy_kernel`);
3. un *kernel* recursivo (`recursive_saxpy_kernel`).

---

## **Kernel execution overhead**

- Hay *overhead* al ejecutar un *kernel* dentro de un *loop* (asignación de recursos, etc.).
- También para la recursión (típicamente más).
- El caso de tener el **ciclo dentro del *kernel*** es el más rápido.

---

# ¡Gracias!

## Próxima clase: librerías de CUDA y Python
