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
<!-- librerías. -->
<!---->
<!-- --- -->

## **Organización y sincronización de los threads**

| Nivel | Se compone de | Sincronización |
|---|---|---|
| *Grid* | bloques | no hay entre bloques (solo al terminar el *kernel*) |
| Bloque | *warps* ($\leq 1024$ *threads*) | `__syncthreads()` |
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

CUDA es un modelo **SIMT** (*single instruction, multiple threads*). Es una mezcla entre **SIMD** (vectorización, AVX) y **SMT** (*threads* independientes, OpenMP):

- Escribimos el código de **un** *thread*; c/u tiene sus propios registros.
- Cada *thread* puede acceder a cualquier dirección (pero el acceso contiguo sigue siendo mejor).
- Cada *thread* puede tomar su propio camino en un `if`/`else` (a costa de producir **divergencia**).

En flexibilidad: **SIMD < SIMT < SMT**.

<!-- NOTA — la idea a transmitir: en SIMD el programador escribe operaciones sobre
vectores; en SIMT escribe el código de UN thread escalar y el hardware lo
ejecuta para 32 a la vez (un warp). Por eso CUDA se siente como programar
threads, pero rinde como vectorización mientras los 32 threads hagan lo mismo.
La flexibilidad tiene precio: acceso disperso (capítulo de memoria) y ramas
divergentes (más adelante en esta clase) cuestan rendimiento. -->

---

## **SIMT: desventajas**

- Con pocos *warps* activos (**occupancy** baja), no hay suficiente concurrencia para esconder la latencia.
- Un *warp* serializa las ramas de un `if` (divergencia).
- CUDA 9 agregó control fino: *warp primitives* (`__shfl_down_sync`, `__syncwarp`, ...) y *cooperative groups*.

<!-- NOTA — los dos costos de las primeras viñetas son la agenda de la clase:
occupancy viene ahora, divergencia dentro de la reducción. Las warp primitives de CUDA 9 también
aparecen en la reducción, como la forma correcta de trabajar dentro de un warp
desde Volta. Cooperative groups queda fuera del curso. -->

---

## **Diseño del GPU**

Latencia contra *throughput*:
- Los **CPUs**  optimizan *latency* (tiempo de espera).
- Los **GPUs**  optimizan *throughput* (cantidad de datos procesados).

Cada *core* de un GPU es mucho más lento que un *core* de un CPU... 

Pero hay miles de *cores*, así que un GPU puede usar miles de *threads* e intercambiar entre ellos (*context switching*) mucho más rápido que un CPU (son *threads* más "livianos").

<!-- NOTA — conviene conectar con el roofline del capítulo 1: un acceso a DRAM
tarda cientos de ciclos (300-600 en la T4, ver la jerarquía de memoria), y el
GPU no intenta reducir esa latencia como un CPU con caches grandes y ejecución
especulativa. La esconde: mientras un warp espera su dato, el SM ejecuta otro.
Para eso necesita muchos warps listos, y eso es exactamente lo que mide el
occupancy. Después de la comparación con el CPU, las diapositivas siguientes
lo muestran como una línea de tiempo. -->

---

## **CPU vs GPU**

<br>

<svg viewBox="0 0 1000 440" width="1000" style="display:block;margin:0 auto" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<text x="240" y="26" font-size="22" font-weight="bold" text-anchor="middle" fill="#0288d1">CPU</text>
<text x="760" y="26" font-size="22" font-weight="bold" text-anchor="middle" fill="#0288d1">GPU</text>
<rect x="30" y="45" width="200" height="105" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<text x="130" y="103.1" font-size="16" text-anchor="middle" fill="#222">Control</text>
<rect x="240" y="45" width="100" height="50" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<text x="290" y="75.6" font-size="16" text-anchor="middle" fill="#222">ALU</text>
<rect x="347" y="45" width="100" height="50" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<text x="397" y="75.6" font-size="16" text-anchor="middle" fill="#222">ALU</text>
<rect x="240" y="100" width="100" height="50" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<text x="290" y="130.6" font-size="16" text-anchor="middle" fill="#222">ALU</text>
<rect x="347" y="100" width="100" height="50" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<text x="397" y="130.6" font-size="16" text-anchor="middle" fill="#222">ALU</text>
<rect x="30" y="160" width="417" height="90" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<text x="238.5" y="210.6" font-size="16" text-anchor="middle" fill="#222">Caché</text>
<rect x="30" y="262" width="417" height="40" rx="3" fill="#ffcc80" stroke="#555" stroke-width="1"/>
<text x="238.5" y="287.6" font-size="16" text-anchor="middle" fill="#222">DRAM</text>
<rect x="550" y="45.0" width="18" height="22" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<rect x="570" y="45.0" width="18" height="22" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<rect x="592" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="616" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="640" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="664" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="688" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="712" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="736" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="760" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="784" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="808" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="832" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="856" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="880" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="904" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="928" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="952" y="45.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="550" y="70.6" width="18" height="22" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<rect x="570" y="70.6" width="18" height="22" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<rect x="592" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="616" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="640" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="664" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="688" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="712" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="736" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="760" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="784" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="808" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="832" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="856" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="880" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="904" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="928" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="952" y="70.6" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="550" y="96.2" width="18" height="22" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<rect x="570" y="96.2" width="18" height="22" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<rect x="592" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="616" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="640" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="664" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="688" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="712" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="736" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="760" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="784" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="808" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="832" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="856" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="880" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="904" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="928" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="952" y="96.2" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="550" y="121.80000000000001" width="18" height="22" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<rect x="570" y="121.80000000000001" width="18" height="22" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<rect x="592" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="616" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="640" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="664" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="688" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="712" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="736" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="760" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="784" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="808" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="832" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="856" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="880" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="904" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="928" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="952" y="121.80000000000001" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="550" y="147.4" width="18" height="22" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<rect x="570" y="147.4" width="18" height="22" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<rect x="592" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="616" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="640" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="664" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="688" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="712" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="736" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="760" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="784" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="808" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="832" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="856" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="880" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="904" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="928" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="952" y="147.4" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="550" y="173.0" width="18" height="22" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<rect x="570" y="173.0" width="18" height="22" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<rect x="592" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="616" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="640" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="664" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="688" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="712" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="736" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="760" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="784" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="808" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="832" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="856" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="880" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="904" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="928" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="952" y="173.0" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="550" y="198.60000000000002" width="18" height="22" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<rect x="570" y="198.60000000000002" width="18" height="22" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<rect x="592" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="616" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="640" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="664" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="688" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="712" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="736" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="760" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="784" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="808" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="832" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="856" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="880" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="904" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="928" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="952" y="198.60000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="550" y="224.20000000000002" width="18" height="22" rx="3" fill="#ffe082" stroke="#555" stroke-width="1"/>
<rect x="570" y="224.20000000000002" width="18" height="22" rx="3" fill="#ce93d8" stroke="#555" stroke-width="1"/>
<rect x="592" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="616" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="640" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="664" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="688" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="712" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="736" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="760" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="784" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="808" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="832" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="856" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="880" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="904" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="928" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="952" y="224.20000000000002" width="21" height="22" rx="3" fill="#a5d6a7" stroke="#555" stroke-width="1"/>
<rect x="550" y="262" width="426" height="40" rx="3" fill="#ffcc80" stroke="#555" stroke-width="1"/>
<text x="763" y="287.6" font-size="16" text-anchor="middle" fill="#222">DRAM</text>
<rect x="200" y="318" width="16" height="16" fill="#ffe082" stroke="#555"/>
<text x="222" y="331" font-size="15" fill="#333">control</text>
<rect x="340" y="318" width="16" height="16" fill="#ce93d8" stroke="#555"/>
<text x="362" y="331" font-size="15" fill="#333">caché</text>
<rect x="470" y="318" width="16" height="16" fill="#a5d6a7" stroke="#555"/>
<text x="492" y="331" font-size="15" fill="#333">cómputo (ALU)</text>
<rect x="650" y="318" width="16" height="16" fill="#ffcc80" stroke="#555"/>
<text x="672" y="331" font-size="15" fill="#333">memoria DRAM</text>
<text x="240" y="370" font-size="17" text-anchor="middle" fill="#333">Pocos núcleos grandes, mucho caché y control:</text>
<text x="240" y="394" font-size="17" text-anchor="middle" fill="#333">que <tspan font-weight="bold">un</tspan> thread espere lo menos posible</text>
<text x="240" y="418" font-size="17" text-anchor="middle" fill="#333">→ baja <tspan font-weight="bold">latencia</tspan></text>
<text x="760" y="370" font-size="17" text-anchor="middle" fill="#333">Miles de núcleos simples: el área va a cómputo.</text>
<text x="760" y="394" font-size="17" text-anchor="middle" fill="#333">La espera se <tspan font-weight="bold">esconde</tspan> con muchos threads</text>
<text x="760" y="418" font-size="17" text-anchor="middle" fill="#333">→ alto <tspan font-weight="bold">throughput</tspan></text>
</svg>

<!-- NOTA — el argumento es de área del chip: los transistores se gastan en cosas
distintas. El CPU los gasta en que UN thread avance lo más rápido posible:
caches grandes para que el dato casi siempre esté cerca, y control complejo
(predicción de saltos, ejecución fuera de orden, especulación) para no
detenerse nunca. El GPU los gasta en unidades de cómputo y en registros para
mantener miles de threads residentes al mismo tiempo, con muy poco control y
caché por núcleo. Cifras de la T4 que ya usamos en el curso: 40 SMs, 2560 CUDA
cores a unos 1.59 GHz, 300 GB/s, y unos 300-600 ciclos de latencia a la DRAM
(jerarquía de memoria, capítulo 2). Un CPU de escritorio típico, sólo como
orden de magnitud: decenas de núcleos, algunos GHz, decenas de GB/s. La
consecuencia es la clave de lo que sigue: el GPU no acorta la espera de un
acceso a memoria; si tuviera un solo warp, estaría casi siempre ocioso.
Necesita muchos warps en vuelo para esconder esa espera, que es lo que
muestran las líneas de tiempo siguientes y lo que después mide el occupancy.
El diagrama es esquemático: las proporciones no son a escala. -->

---

## **SM y organización de los threads**

![w:520px](images/warps_thread_blocks.png)

<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

Los *warps* son manejados por los *streaming multiprocessors* (SM).

---

## **Latencia: ejemplo con un solo warp**

```cuda
float a = x[i];     // L: cargar desde la memoria global (latencia alta)
y[i] = 2.0f * a;    // C: calcular y guardar resultado (necesita a)
```

Asumamos que una carga tarda $4$ unidades de tiempo (ciclos del reloj) en llegar, y el  *streaming multiprocessor* (SM) emite **una** instrucción por ciclo.

**Línea de tiempo:** 

<table class="timeline">
<tr><th></th><th class="t">t0</th><th class="t">t1</th><th class="t">t2</th><th class="t">t3</th><th class="t">t4</th></tr>
<tr><th>Warp 0</th><td class="s1">L</td><td class="espera"></td><td class="espera"></td><td class="espera"></td><td class="s1">C</td></tr>
<tr><th>SM emite</th><td class="s1">W0</td><td>—</td><td>—</td><td>—</td><td class="s1">W0</td></tr>
</table>

El SM está esperando (*idle*) durante $3$ de $5$ ciclos: el *warp* pasa casi todo el tiempo esperando.

<!-- NOTA — el modelo es de juguete a propósito: en la T4 un acceso a DRAM tarda
unos 300-600 ciclos (diapositiva de la jerarquía en el capítulo de memoria),
no 4. Con un solo warp, el SM estaría ocioso casi todo el tiempo: emite la
carga, y después no tiene nada que hacer hasta que llega el dato, porque la
instrucción siguiente depende de él. Las casillas rayadas son ese tiempo de
espera. -->


---

## **Esconder la latencia: varios warps**

```cuda
float a = x[i];     // L: cargar desde la memoria global (latencia alta)
y[i] = 2.0f * a;    // C: calcular y guardar resultado (necesita a)
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

**¿Cómo reparte el trabajo el SM para mantener el sistema ocupado?**

<!-- NOTA — dejar que la clase proponga antes de pasar a la diapositiva siguiente;
se puede llenar la tabla en la pizarra con sus ideas. Las respuestas que
buscamos son dos. (1) Más warps: mientras uno espera su carga, el SM emite la
instrucción de otro (paralelismo entre threads, TLP); es lo que muestra la
diapositiva que sigue, y lleva al occupancy. (2) Más trabajo
independiente dentro de cada thread: si cada thread carga dos elementos antes
de usar el primero (por ejemplo x[i] y x[i + N/2]), las dos cargas están en
vuelo a la vez (paralelismo de instrucciones, ILP); es lo que hace el loop
unrolling en la reducción. Las dos se combinan. Respuestas que conviene
descartar: hacer más rápido un solo thread no sirve, porque lo que domina es
la espera de la memoria, no el cálculo; y reordenar las dos instrucciones no
se puede, porque C depende de L. -->

---

## **Esconder la latencia: varios warps**

<br>

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

## **Orden en la ejecución de los threads**

**Importante:** 
- Desde Volta **no** hay *lock-step* garantizado: si un *thread* usa el resultado de otro, hay que usar `__syncwarp()`.
- Al igual que en la programación en paralelo con CPU (eg. con OpenMP), **no** podemos asumir un orden de ejecución de los *threads*.
- No hay garantías en el orden de ejecución de bloques y *warps*.

Si nuestro programa **asume** un cierto orden de ejecución para obtener resultados correctos, **tiene un bug**. Nosotros debemos imponerlo.

<!-- NOTA — como en OpenMP, el orden de los threads no está garantizado, pero aquí
la escala es otra: miles de threads repartidos por el hardware, sin que el
programador decida nada. Ojo con el matiz de la última frase: muchos programas
correctos SÍ necesitan un orden (la transpuesta escribe el tile antes de
leerlo; la reducción hace un nivel después del otro). Lo que está mal es
suponerlo; hay que imponerlo con __syncthreads(), __syncwarp(), atómicas o
lanzamientos de kernel separados. La diapositiva siguiente muestra el desorden
en la práctica. -->

---

## **Orden en la ejecución de los threads**

Ejemplo: [cuda_thread_block.cu](../code/threads/cuda_thread_block.cu).

```bash
nvcc -arch=sm_75 cuda_thread_block.cu -o cuda_thread_block.x
./cuda_thread_block.x 4 128     # argumentos: bloques, threads por bloque
```

- Ejecutarlo varias veces: el orden entre bloques y *warps* cambia.
<!-- - Dentro de un *warp* suelen imprimir en orden de *lane*: eso **no** prueba que avancen juntos. -->

<!-- NOTA — el programa recibe el tamaño del grid y del bloque como argumentos
(./cuda_thread_block.x 4 128) e imprime thread, bloque, warp y lane para
los lanes 0 y 16 de cada warp. Sin argumentos muestra cómo usarlo y termina.
Vale la pena ejecutarlo varias veces: el orden entre bloques y entre warps
cambia. Que dentro de un warp salga en orden es un detalle de
implementación de printf, no una garantía; desde Volta los threads de un warp
pueden avanzar por separado. Ojo con no confundir ORDEN con LOCK-STEP: el orden
es la secuencia en que aparecen las líneas; lock-step es que los 32 threads
ejecuten la misma instrucción al mismo tiempo, sin que ninguno empiece la
siguiente antes de que todos terminen la actual. Son independientes: un printf
suele escribir las líneas del warp en orden de lane avancen o no juntos, así
que ver el orden no prueba el lock-step; y perder el lock-step no desordena la
salida de este ejemplo, porque aquí los lanes no divergen. Lo que importa es la
regla: si un lane lee lo que escribió otro, hay que sincronizar con
__syncwarp() o una primitiva *_sync (como __shfl_down_sync en la
reducción). -->

<!-- Si alguien nota que dos corridas seguidas imprimen el MISMO orden y
después cambia: "no garantizado" no significa "aleatorio". Los
planificadores de bloques y de warps siguen reglas fijas, así que con el GPU
desocupado y el mismo lanzamiento suelen repetir el orden. Pequeñas
diferencias de tiempo (el reloj del GPU según temperatura y consumo, qué SMs
estaban libres, quién escribe primero en el buffer de printf) desempatan
distinto de vez en cuando. Moraleja: que un orden se repita no lo hace
garantizado; si la corrección depende del orden, hay que imponerlo con
sincronización (__syncthreads, __syncwarp, atómicas). Así pasan
desapercibidas las race conditions. -->

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
cada límite por separado.

Cómo obtenerlos en Colab: nvidia-smi solo da la capacidad de cómputo
(nvidia-smi --query-gpu=name,compute_cap --format=csv). simpleDeviceQuery.cu
(intro) imprime los threads por SM, pero no los bloques por SM, y da
regsPerBlock y sharedMemPerBlock (por bloque, no por SM). Basta un programa de
pocas líneas que imprima los cuatro campos de arriba con cudaGetDeviceProperties.
En la T4 deberían dar 1024 (32 warps), 16, 65536 y 65536 bytes. Para un kernel
concreto, ncu --section Occupancy --section LaunchStats ./prog muestra el
límite que impone cada recurso (Block Limit Registers / Shared Mem / Warps /
SM), que es la cuenta de las diapositivas siguientes. Referencia publicada: la
tabla "Compute Capabilities" de la CUDA C++ Programming Guide, columna 7.5.

Ojo con la memoria compartida: el SM tiene hasta 64 KB, pero cada bloque
recibe como máximo 48 KB por defecto (sharedMemPerBlock). Para pasar de 48 KB
hay que usar memoria compartida dinámica y pedirlo explícitamente con
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...),
hasta sharedMemPerBlockOptin (64 KB en la T4). Los ejemplos de la diapositiva
de memoria compartida (16 y 32 KB por bloque) quedan bajo ese límite. -->

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

**Notar:** Bloques de menos de $64$ *threads* no alcanzan a llenar el SM.

<!-- NOTA — el occupancy se cuenta en warps, no en threads: un bloque más chico que
un warp (por ejemplo 16 threads) ocupa igual un warp entero, con lanes
apagados. Con bloques de 16, como en el ejercicio de matriz_mult.cu, caben 16
bloques = 16 warps = 50% de occupancy, y además la mitad de cada warp no hace
nada. Por eso conviene que el tamaño del bloque sea múltiplo de 32. -->

<!-- NOTA — el caso de 32 threads es el que más sorprende: el bloque es un warp
completo, pero como solo caben 16 bloques por SM, quedan 16 warps de 32
posibles. Por eso la recomendación habitual es bloques de 128 a 256 threads:
llenan el SM sin quedar limitados por el número de bloques, y dejan margen
para que varios bloques convivan. El ejercicio de matriz_mult.cu, con bloques
de 16 threads, tiene el mismo occupancy que la primera fila (50%), pero es
peor: cada warp trabaja con solo 16 de sus 32 lanes. -->

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

## **Occupancy: cómo medirlo**

- **Teórico**: con los recursos del *kernel* (`nvcc --resource-usage` da registros y memoria compartida) y los límites de la T4.
- **Medido** (*achieved*): con el *profiler*. `ncu --section Occupancy` muestra el teórico, el medido y **qué límite** manda.
- La línea *Achieved Occupancy* de `ncu` es la métrica `sm__warps_active.avg.pct_of_peak_sustained_active`.

<!-- NOTA — --resource-usage imprime, por cada kernel, los registros por thread y
la memoria compartida y constante que usa; con eso y los límites de la T4 se
calcula el occupancy teórico. Para el medido basta ncu sin argumentos: sus
secciones por defecto ya incluyen Occupancy, y su línea Achieved Occupancy es
exactamente la métrica sm__warps_active.avg.pct_of_peak_sustained_active;
pedir la métrica sola (ncu --metrics ...) sólo sirve para obtener una línea
por kernel. El medido puede quedar bajo el teórico, por ejemplo cuando la
última tanda de bloques no llena todos los SMs. Ojo: ncu fija el reloj en su
valor base, así que sus tiempos no se comparan con los medidos con eventos. El
ejercicio de dos diapositivas más adelante aplica todo esto a matriz_mult.cu. -->

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

Descargar: [matriz_mult.cu](../code/threads/matriz_mult.cu). Por defecto usa bloques de $4 \times 4 = 16$ *threads*; el tamaño se puede pasar como argumento.

```bash
nvcc -arch=sm_75 --resource-usage matriz_mult.cu -o matriz_mult.x
ncu --section Occupancy ./matriz_mult.x         # bloques de 4 x 4
ncu --section Occupancy ./matriz_mult.x 16 16   # bloques de 16 x 16
```

1. Con los límites de la T4, ¿cuál es el *occupancy* teórico? (`--resource-usage` da los registros).
2. Medirlo con el *profiler*.
3. Repetir con bloques de $16 \times 16$.

<!-- RESPUESTAS medidas con ncu --section Occupancy en la T4 (2026-10). (1) Bloques
de 4x4: occupancy teórico 50%, no 25%. El occupancy se cuenta en WARPS, y un
bloque de 16 threads ocupa igual un warp completo (con la mitad de los lanes
apagados). Manda el límite de 16 bloques por SM: 16 bloques x 1 warp = 16 de
32 warps = 50%. Los registros no limitan: 52 por thread, cabrían 36 bloques.
ncu también nombra a la memoria compartida como límite aunque el kernel no la
usa: su límite coincide con los mismos 16 bloques, se puede ignorar. Se pierde
por partida doble: occupancy de 50%, y además cada warp trabaja con 16 de sus
32 lanes, así que sólo 256 de 1024 threads posibles hacen trabajo útil. (2)
Achieved 47.8% (15.3 warps activos por SM), cerca del teórico. Del reporte
completo de ncu: DRAM Throughput 1% pero L1/TEX 97%; las matrices de 256x256
(256 KB) caben en cache, así que el kernel está limitado por el L1, no por la
DRAM. (3) Bloques de 16x16 (./matriz_mult.x 16 16; el programa recibe el
tamaño de bloque como argumento y redondea el grid hacia arriba): teórico
100%. Ahora mandan a la vez los registros (52 x 256 threads: caben 4 bloques)
y los warps (8 por bloque: 4 bloques = 32 warps). Achieved 79.7%, y la
diferencia se explica casi exacta con las tandas (waves) de bloques: el grid
tiene 16x16 = 256 bloques y caben 40 SMs x 4 = 160 a la vez. La primera tanda
llena todos los SMs; la segunda tiene sólo 96 bloques, 2.4 por SM en promedio.
Si las dos tandas duran lo mismo, el promedio es (4 + 2.4)/2 = 3.2 bloques por
SM, o sea 3.2/4 = 80%. Con N más grande la última tanda pesa menos y el
achieved se acerca al 100%: el occupancy medido depende también del tamaño del
problema. Al comparar tiempos: ncu fija el reloj del GPU en su valor base (585
MHz contra unos 1590 de boost), así que sus duraciones no se comparan con
tiempos medidos con eventos; para eso, ncu --clock-control none. nvprof
--metrics achieved_occupancy sirve como alternativa a ncu. -->

---

# Reducción: una aplicación de la optimización

---

## **Reducción paralela**

- Muchas operaciones tienen como objetivo obtener un **solo valor a partir de un conjunto de datos**.
  - Este tipo de operaciones la denominamos **reducción**.
- Ahora discutiremos un cálculo de reducción en paralelo con GPU.

**Ejemplo:**
Calcular la sumatoria de todos los elementos en un *array* utilizando $N$ *threads*.

<svg viewBox="0 0 960 145" width="860" style="display:block;margin:0.6em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
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
<text x="760" y="38" font-size="22" fill="#455a64" font-style="italic">datos</text>
<path d="M28,58 Q28,72 42,72 L361,72 Q375,72 375,86 Q375,72 389,72 L708,72 Q722,72 722,58" stroke="#555" stroke-width="2.5" fill="none"/>
<text x="331" y="126" font-size="30" fill="#455a64" text-anchor="end" font-family="KaTeX_Main, 'Times New Roman', serif">&#931; = </text>
<rect x="343" y="97" width="64" height="40" rx="4" fill="#76b900"/>
<text x="375" y="125" font-size="24" fill="#fff" text-anchor="middle">134</text>
</svg>

<!-- NOTA — la reducción es un buen caso de estudio porque es simple de entender y
casi todo lo que cuesta es memoria: una suma por cada 4 bytes leídos, o sea la
intensidad aritmética más baja posible. Todas las optimizaciones que siguen
apuntan a lo mismo: leer cada elemento una vez, de forma contigua, y
desperdiciar la menor cantidad de threads. -->

---

## **Reducción paralela: árbol**

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
 - En cada paso (nivel del árbol), juntamos las sumas parciales (sincronizamos).

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

## **Reducción paralela: memoria global**

<svg viewBox="0 0 1044 496" width="900" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<line x1="192" y1="52" x2="192" y2="70" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="246" y1="52" x2="206.3" y2="77.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="192" y1="103" x2="192" y2="121" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="300" y1="52" x2="300" y2="70" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="354" y1="52" x2="314.3" y2="77.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="300" y1="103" x2="300" y2="121" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="408" y1="52" x2="408" y2="70" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="462" y1="52" x2="422.3" y2="77.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="408" y1="103" x2="408" y2="121" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="516" y1="52" x2="516" y2="70" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="570" y1="52" x2="530.3" y2="77.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="516" y1="103" x2="516" y2="121" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="624" y1="52" x2="624" y2="70" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="678" y1="52" x2="638.3" y2="77.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="624" y1="103" x2="624" y2="121" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="732" y1="52" x2="732" y2="70" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="786" y1="52" x2="746.3" y2="77.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="732" y1="103" x2="732" y2="121" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="840" y1="52" x2="840" y2="70" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="894" y1="52" x2="854.3" y2="77.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="840" y1="103" x2="840" y2="121" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="948" y1="52" x2="948" y2="70" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="1002" y1="52" x2="962.3" y2="77.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="948" y1="103" x2="948" y2="121" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="192" y1="162" x2="192" y2="180" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="300" y1="162" x2="208.2" y2="191.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="192" y1="213" x2="192" y2="231" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="408" y1="162" x2="408" y2="180" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="516" y1="162" x2="424.2" y2="191.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="408" y1="213" x2="408" y2="231" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="624" y1="162" x2="624" y2="180" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="732" y1="162" x2="640.2" y2="191.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="624" y1="213" x2="624" y2="231" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="840" y1="162" x2="840" y2="180" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="948" y1="162" x2="856.2" y2="191.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="840" y1="213" x2="840" y2="231" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="192" y1="272" x2="192" y2="290" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="408" y1="272" x2="208.8" y2="304.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="192" y1="323" x2="192" y2="341" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="624" y1="272" x2="624" y2="290" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="840" y1="272" x2="640.8" y2="304.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="624" y1="323" x2="624" y2="341" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="192" y1="382" x2="192" y2="400" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="624" y1="382" x2="208.9" y2="415.6" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<line x1="192" y1="433" x2="192" y2="451" stroke="#444" stroke-width="1.4" marker-end="url(#fl-red)"/>
<rect x="170" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="224" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="278" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="332" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="386" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="440" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="494" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="548" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="602" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="656" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="710" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="764" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="818" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="872" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="926" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="980" y="12" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="170" y="122" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="224" y="122" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="122" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="332" y="122" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="122" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="440" y="122" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="122" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="548" y="122" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="602" y="122" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="656" y="122" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="710" y="122" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="764" y="122" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="818" y="122" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="872" y="122" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="926" y="122" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="980" y="122" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="232" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="224" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="232" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="440" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="602" y="232" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="656" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="710" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="764" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="818" y="232" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="872" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="926" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="980" y="232" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="342" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="224" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="602" y="342" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="656" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="710" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="764" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="818" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="872" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="926" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="980" y="342" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="452" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<rect x="224" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="602" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="656" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="710" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="764" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="818" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="872" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="926" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="980" y="452" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<circle cx="192" cy="87" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="93" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="300" cy="87" r="16" fill="#e8641b" stroke="#444"/>
<text x="300" y="93" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="408" cy="87" r="16" fill="#e8641b" stroke="#444"/>
<text x="408" y="93" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="516" cy="87" r="16" fill="#e8641b" stroke="#444"/>
<text x="516" y="93" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="624" cy="87" r="16" fill="#e8641b" stroke="#444"/>
<text x="624" y="93" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="732" cy="87" r="16" fill="#e8641b" stroke="#444"/>
<text x="732" y="93" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="840" cy="87" r="16" fill="#e8641b" stroke="#444"/>
<text x="840" y="93" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="948" cy="87" r="16" fill="#e8641b" stroke="#444"/>
<text x="948" y="93" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="192" cy="197" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="203" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="408" cy="197" r="16" fill="#e8641b" stroke="#444"/>
<text x="408" y="203" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="624" cy="197" r="16" fill="#e8641b" stroke="#444"/>
<text x="624" y="203" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="840" cy="197" r="16" fill="#e8641b" stroke="#444"/>
<text x="840" y="203" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="192" cy="307" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="313" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="624" cy="307" r="16" fill="#e8641b" stroke="#444"/>
<text x="624" y="313" font-size="16" text-anchor="middle" fill="#fff">+</text>
<circle cx="192" cy="417" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="423" font-size="16" text-anchor="middle" fill="#fff">+</text>
<text x="156" y="38" font-size="16" text-anchor="end" fill="#333">Memoria global</text>
<text x="156" y="93" font-size="16" text-anchor="end" fill="#333">suma</text>
<text x="156" y="111" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
<text x="156" y="221" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<text x="156" y="331" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 4</text>
<text x="156" y="441" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 8</text>
</svg>

Asignamos a los *threads* la tarea de sumar pares usando un *stride*.

<!-- NOTA — en esta figura conviene marcar dos cosas que se repiten en todas las
versiones: cuántos threads trabajan en cada nivel (la mitad del anterior) y
qué distancia hay entre los dos elementos que se suman (el stride). Las
versiones que siguen difieren justamente en cómo se asignan threads a
elementos y en el orden de los strides. -->


<!-- NOTA — la figura (redibujada en SVG, en el estilo de las demás figuras de
la reducción). Dieciséis datos; en cada nivel el elemento i se suma con el
elemento i + stride y el resultado se guarda en i, con stride = 1, 2, 4, 8.
Cada círculo es una suma; las casillas lila son datos que ya no se usan. Al final la suma total queda en la casilla 0, después de log2(16) = 4
niveles. La figura original usa un operador genérico (*), porque el mismo
árbol sirve para cualquier operación asociativa (máximo, mínimo, producto);
aquí se dibuja + porque en el curso reducimos con sumas. Dos cosas para
marcar, que se repiten en todas las versiones: en cada nivel trabaja la mitad
de los threads del anterior, y el stride se duplica. Este patrón de pares
vecinos con stride creciente es el de reduccion_global y reduccion_global2;
las versiones con acceso contiguo recorren los strides al revés (de mayor a
menor). -->

---

## **Reducción paralela: memoria global**

Ejemplo: [reduccion_global.cu](../code/threads/reduccion_global.cu).

Implementa la estrategia de reducción anterior.
- El *kernel* es invocado dentro de un ciclo que recorre cada nivel del árbol.
- Al terminar cada iteración, se acumulan las sumas parciales, y el *stride* aumenta en un factor de $2$.

<!-- NOTA — la primera versión no usa memoria compartida ni bloques de forma
inteligente: el ciclo sobre el stride está en el host, y cada nivel del árbol
es un lanzamiento de kernel distinto. Con N = 2^24 son 24 lanzamientos. La
ventaja es que el fin de cada kernel sincroniza todo el grid, así que no hace
falta ninguna barrera dentro del kernel. Cada lanzamiento lee de un arreglo y
escribe en otro (la diapositiva del código explica por qué).

El host y el device no dan exactamente lo mismo, y el que se equivoca es el
CPU. Cada elemento vale a lo más 255/(2^31 - 1) ~ 1.2e-7 (media ~ 6e-8). Cuando
el acumulador float secuencial pasa de 1, la separación entre floats es
2^-23 ~ 1.2e-7: cada término nuevo es la mitad de esa separación, así que cada
suma redondea y el error se acumula, O(N). El GPU suma en árbol (por pares de
tamaño parecido) y el error crece como O(log N). Comparado con la suma exacta
en double (mismo srand(2019), rand de glibc):

  N      float secuencial (host)   double (exacta)   árbol (device)
  2^24   0.995958                  0.996006836       0.996007
  2^25   1.992238                  1.992247406       1.992247

Con 2^25 (una versión anterior del programa) la T4 imprimió
"host: 1.992238, device 1.992247": el device coincide con la suma exacta en
todas las cifras impresas y el host difiere en 9e-6. Con el N = 2^24 actual
la T4 imprime "host: 0.995958, device 0.996007" (2026-10), igual a la
predicción de la tabla: el host difiere en 4.9e-5. Lección: la reducción
paralela es más precisa; el CPU no es automáticamente la referencia. -->

---

## **Reducción paralela: memoria global**

El ciclo sobre el *stride* está en el *host*: un lanzamiento por nivel del árbol.

@include[cuda]{static/code/threads/reduccion_global.cu:60-71}

<!-- NOTA — con N = 2^24 el ciclo da 24 vueltas: stride = 1, 2, 4, ..., 2^23.
Cada lanzamiento usa n_bloques bloques de 1024 threads, es decir N threads en
todos los niveles, aunque cada nivel necesite la mitad de resultados que el
anterior (es el problema que plantea la diapositiva "Hay un problema con el
kernel"). Los lanzamientos en el mismo stream se ejecutan en orden, así que
cada uno ve completo el resultado del anterior sin sincronizar nada desde el
host. El cudaMemcpy final sí espera a que termine el último kernel. -->

---

## **Reducción paralela: memoria global**

Kernel invocado:

```cuda
__global__ void reduccion_memoria_global(float *data_out, const float *data_in,
                                         int stride, int N) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_x + stride < N)
    data_out[idx_x] = data_in[idx_x] + data_in[idx_x + stride];
  else if (idx_x < N)
    data_out[idx_x] = data_in[idx_x];
}
```

<!-- NOTA — ¿por qué dos arreglos? Con uno solo, data[idx] += data[idx + stride],
el thread idx leería data[idx + stride] mientras el thread idx + stride lo
está modificando en el MISMO lanzamiento: una race condition. Dentro de un
warp las lecturas ocurren antes que las escrituras, pero entre warps y entre
bloques no hay ningún orden garantizado, y el resultado puede salir mal (es
la regla de las diapositivas de orden de ejecución). Con data_in y data_out
separados ningún thread lee lo que otro escribe en el mismo lanzamiento;
después de cada lanzamiento se intercambian los punteros, y el siguiente
lanzamiento empieza recién cuando termina el anterior. El resultado queda en
d_in[0] después del último intercambio. -->

---

## **Reducción paralela: memoria global**

<svg viewBox="0 0 612 436" width="600" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-glob" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<text x="170" y="16" font-size="16" fill="#333">N = 8, n_threads = 4: n_bloques = 2</text>
<path d="M170,56 L170,50 L376,50 L376,56" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="273" y="44" font-size="15" text-anchor="middle" fill="#333">Bloque 0</text>
<path d="M386,56 L386,50 L592,50 L592,56" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="489" y="44" font-size="15" text-anchor="middle" fill="#333">Bloque 1</text>
<line x1="192" y1="102" x2="192" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="246" y1="102" x2="206.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="192" y1="153" x2="192" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="246" y1="102" x2="246" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="300" y1="102" x2="260.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="246" y1="153" x2="246" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="300" y1="102" x2="300" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="354" y1="102" x2="314.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="300" y1="153" x2="300" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="354" y1="102" x2="354" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="102" x2="368.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="354" y1="153" x2="354" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="102" x2="408" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="102" x2="422.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="153" x2="408" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="102" x2="462" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="102" x2="476.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="153" x2="462" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="102" x2="516" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="102" x2="530.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="153" x2="516" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="102" x2="570" y2="120" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="153" x2="570" y2="171" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="192" y1="212" x2="192" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="300" y1="212" x2="208.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="192" y1="263" x2="192" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="246" y1="212" x2="246" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="354" y1="212" x2="262.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="246" y1="263" x2="246" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="300" y1="212" x2="300" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="212" x2="316.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="300" y1="263" x2="300" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="354" y1="212" x2="354" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="212" x2="370.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="354" y1="263" x2="354" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="212" x2="408" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="212" x2="424.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="263" x2="408" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="212" x2="462" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="212" x2="478.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="263" x2="462" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="212" x2="516" y2="230" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="263" x2="516" y2="281" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="212" x2="570" y2="230" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="263" x2="570" y2="281" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="192" y1="322" x2="192" y2="340" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="322" x2="208.8" y2="354.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="192" y1="373" x2="192" y2="391" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="246" y1="322" x2="246" y2="340" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="322" x2="262.8" y2="354.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="246" y1="373" x2="246" y2="391" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="300" y1="322" x2="300" y2="340" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="322" x2="316.8" y2="354.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="300" y1="373" x2="300" y2="391" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="354" y1="322" x2="354" y2="340" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="322" x2="370.8" y2="354.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="354" y1="373" x2="354" y2="391" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="322" x2="408" y2="340" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="408" y1="373" x2="408" y2="391" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="322" x2="462" y2="340" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="462" y1="373" x2="462" y2="391" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="322" x2="516" y2="340" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="516" y1="373" x2="516" y2="391" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="322" x2="570" y2="340" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<line x1="570" y1="373" x2="570" y2="391" stroke="#444" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#fl-glob)"/>
<rect x="170" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="88" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="224" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="88" font-size="18" text-anchor="middle" fill="#222">1</text>
<rect x="278" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="88" font-size="18" text-anchor="middle" fill="#222">8</text>
<rect x="332" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="88" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="386" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="88" font-size="18" text-anchor="middle" fill="#222">7</text>
<rect x="440" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="462" y="88" font-size="18" text-anchor="middle" fill="#222">2</text>
<rect x="494" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="88" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="548" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="570" y="88" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="170" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="198" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="224" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="198" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="278" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="198" font-size="18" text-anchor="middle" fill="#222">12</text>
<rect x="332" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="198" font-size="18" text-anchor="middle" fill="#222">11</text>
<rect x="386" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="198" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="440" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="462" y="198" font-size="18" text-anchor="middle" fill="#222">11</text>
<rect x="494" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="198" font-size="18" text-anchor="middle" fill="#222">12</text>
<rect x="548" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="570" y="198" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="170" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="308" font-size="18" text-anchor="middle" fill="#222">16</text>
<rect x="224" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="308" font-size="18" text-anchor="middle" fill="#222">20</text>
<rect x="278" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="308" font-size="18" text-anchor="middle" fill="#222">21</text>
<rect x="332" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="308" font-size="18" text-anchor="middle" fill="#222">22</text>
<rect x="386" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="308" font-size="18" text-anchor="middle" fill="#222">21</text>
<rect x="440" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="462" y="308" font-size="18" text-anchor="middle" fill="#222">14</text>
<rect x="494" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="308" font-size="18" text-anchor="middle" fill="#222">12</text>
<rect x="548" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="570" y="308" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="170" y="392" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="418" font-size="18" text-anchor="middle" fill="#222">37</text>
<rect x="224" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<circle cx="192" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="143" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="246" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="246" y="143" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="300" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="300" y="143" font-size="16" text-anchor="middle" fill="#fff">2</text>
<circle cx="354" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="354" y="143" font-size="16" text-anchor="middle" fill="#fff">3</text>
<circle cx="408" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="408" y="143" font-size="16" text-anchor="middle" fill="#fff">4</text>
<circle cx="462" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="462" y="143" font-size="16" text-anchor="middle" fill="#fff">5</text>
<circle cx="516" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="516" y="143" font-size="16" text-anchor="middle" fill="#fff">6</text>
<circle cx="570" cy="137" r="16" fill="#bdbdbd" stroke="#444"/>
<text x="570" y="143" font-size="16" text-anchor="middle" fill="#fff">7</text>
<circle cx="192" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="253" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="246" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="246" y="253" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="300" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="300" y="253" font-size="16" text-anchor="middle" fill="#fff">2</text>
<circle cx="354" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="354" y="253" font-size="16" text-anchor="middle" fill="#fff">3</text>
<circle cx="408" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="408" y="253" font-size="16" text-anchor="middle" fill="#fff">4</text>
<circle cx="462" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="462" y="253" font-size="16" text-anchor="middle" fill="#fff">5</text>
<circle cx="516" cy="247" r="16" fill="#bdbdbd" stroke="#444"/>
<text x="516" y="253" font-size="16" text-anchor="middle" fill="#fff">6</text>
<circle cx="570" cy="247" r="16" fill="#bdbdbd" stroke="#444"/>
<text x="570" y="253" font-size="16" text-anchor="middle" fill="#fff">7</text>
<circle cx="192" cy="357" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="363" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="246" cy="357" r="16" fill="#e8641b" stroke="#444"/>
<text x="246" y="363" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="300" cy="357" r="16" fill="#e8641b" stroke="#444"/>
<text x="300" y="363" font-size="16" text-anchor="middle" fill="#fff">2</text>
<circle cx="354" cy="357" r="16" fill="#e8641b" stroke="#444"/>
<text x="354" y="363" font-size="16" text-anchor="middle" fill="#fff">3</text>
<circle cx="408" cy="357" r="16" fill="#bdbdbd" stroke="#444"/>
<text x="408" y="363" font-size="16" text-anchor="middle" fill="#fff">4</text>
<circle cx="462" cy="357" r="16" fill="#bdbdbd" stroke="#444"/>
<text x="462" y="363" font-size="16" text-anchor="middle" fill="#fff">5</text>
<circle cx="516" cy="357" r="16" fill="#bdbdbd" stroke="#444"/>
<text x="516" y="363" font-size="16" text-anchor="middle" fill="#fff">6</text>
<circle cx="570" cy="357" r="16" fill="#bdbdbd" stroke="#444"/>
<text x="570" y="363" font-size="16" text-anchor="middle" fill="#fff">7</text>
<text x="156" y="88" font-size="16" text-anchor="end" fill="#333">Memoria global</text>
<text x="156" y="143" font-size="16" text-anchor="end" fill="#333">ID del thread</text>
<text x="156" y="161" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
<text x="156" y="271" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<text x="156" y="381" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 4</text>
</svg>

- Cada nivel del árbol es **un lanzamiento** del *kernel* en el ciclo.
- Los *threads* se muestran en los círculos naranjos (gris: solo copia).

<!-- NOTA — propósito de esta diapositiva: seguir a mano, con N = 8, el ciclo
y el kernel de las dos diapositivas anteriores. Cada lanzamiento suma a
cada elemento el que está stride posiciones más adelante; después de los
niveles con stride 1, 2 y 4, data[0] tiene la suma de los 8 (37). Con
stride = N/2 se suma la última mitad sobre la primera; con stride = N ya no
queda nada que sumar, por eso el ciclo es stride < N.

Los threads 1, 2 y 3 también suman en el último nivel (20 + 14 = 34,
21 + 12 = 33, 22 + 3 = 25) y los threads 4 a 7 copian su valor; esos
resultados no le sirven a nadie. Es justamente la pregunta de la
diapositiva siguiente: conviene preguntar a los alumnos qué threads
trabajaron en cada fila antes de llegar ahí. -->

<!-- NOTA — la figura (redibujada en SVG a partir de la del libro, en el estilo
de la de acceso contiguo) dibuja lo que hace el código en TODOS los
threads: los círculos naranjos suman (data_out[i] = data_in[i] +
data_in[i + stride]) y los grises, con flechas punteadas, copian
(data_out[i] = data_in[i], cuando i + stride >= N). En el último nivel los
threads 1 a 3 también suman (20 + 14, 21 + 12, 22 + 3), pero nadie usa esos
resultados: por eso esas casillas son lila (datos que ya no se usan), y solo
data[0] = 37 queda con número. Las cuentas: stride 1 → 4, 9, 12, 11, 9, 11,
12, 3; stride 2 → 16, 20, 21, 22, 21, 14, 12, 3; stride 4 → 37 (y 34, 33,
25, 21, 14, 12, 3 en las casillas lila). Cada fila es un
arreglo distinto (d_in y d_out se intercambian), así que ninguna flecha lee
una casilla que se escribe en la misma fila. Suman 7 + 6 + 4 = 17 threads,
que es la cuenta de la diapositiva siguiente. -->

---

## **Reducción paralela: memoria global**

Este *kernel* tiene hace más trabajo del necesario...

... todos los *threads* calculan algo, ¡pero **la mayoría de los resultados no se usan**!

- Esto es evidente en el último nivel: los resultados de los threads 1, 2 y 3 no terminan en ninguna parte.

- En el nivel con *stride* $s$ solo sirven los índices múltiplos de $2s$.

Con $N = 8$: estamos haciendo $7 + 6 + 4 = 17$ sumas en total, cuando bastan $4 + 2 + 1 = 7$.

<!-- - En general: $\sim N \log_2 N$ sumas en vez de $N - 1$. -->

<!-- NOTA — la respuesta: en cada lanzamiento todos los threads hacen la suma, pero
en el nivel con stride s solo sirven los resultados de los índices múltiplos
de 2s; el resto escribe valores que nadie va a leer. En el último nivel
trabajan millones de threads para producir un único número útil. Además, todo
pasa por memoria global, y cada nivel lee y escribe el arreglo entero.

Las cuentas con N = 8: un thread suma si idx + stride < N, así que con
stride 1 suman 7 threads, con stride 2 suman 6 y con stride 4 suman 4. Un
árbol solo necesita N/2 + N/4 + ... + 1 = N - 1 = 7. Con N = 2^24 son 24
niveles de ~2^24 sumas cada uno, unas 4*10^8, contra 1.7*10^7 necesarias.
La próxima versión ataca exactamente esto. -->

---

## **Reducción paralela: menos threads trabajando**

Ejemplo: [reduccion_global2.cu](../code/threads/reduccion_global2.cu).

<svg viewBox="0 0 612 436" width="560" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-glob2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<text x="170" y="16" font-size="16" fill="#333">N = 8, n_threads = 4: n_bloques = 2</text>
<path d="M170,56 L170,50 L376,50 L376,56" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="273" y="44" font-size="15" text-anchor="middle" fill="#333">Bloque 0</text>
<path d="M386,56 L386,50 L592,50 L592,56" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="489" y="44" font-size="15" text-anchor="middle" fill="#333">Bloque 1</text>
<line x1="192" y1="102" x2="192" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="246" y1="102" x2="206.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="192" y1="153" x2="192" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="300" y1="102" x2="300" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="354" y1="102" x2="314.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="300" y1="153" x2="300" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="408" y1="102" x2="408" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="462" y1="102" x2="422.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="408" y1="153" x2="408" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="516" y1="102" x2="516" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="570" y1="102" x2="530.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="516" y1="153" x2="516" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="192" y1="212" x2="192" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="300" y1="212" x2="208.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="192" y1="263" x2="192" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="408" y1="212" x2="408" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="516" y1="212" x2="424.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="408" y1="263" x2="408" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="192" y1="322" x2="192" y2="340" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="408" y1="322" x2="208.8" y2="354.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<line x1="192" y1="373" x2="192" y2="391" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob2)"/>
<rect x="170" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="88" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="224" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="88" font-size="18" text-anchor="middle" fill="#222">1</text>
<rect x="278" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="88" font-size="18" text-anchor="middle" fill="#222">8</text>
<rect x="332" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="88" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="386" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="88" font-size="18" text-anchor="middle" fill="#222">7</text>
<rect x="440" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="462" y="88" font-size="18" text-anchor="middle" fill="#222">2</text>
<rect x="494" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="88" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="548" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="570" y="88" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="170" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="198" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="224" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="198" font-size="18" text-anchor="middle" fill="#222">12</text>
<rect x="332" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="198" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="440" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="198" font-size="18" text-anchor="middle" fill="#222">12</text>
<rect x="548" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="308" font-size="18" text-anchor="middle" fill="#222">16</text>
<rect x="224" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="308" font-size="18" text-anchor="middle" fill="#222">21</text>
<rect x="440" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="392" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="418" font-size="18" text-anchor="middle" fill="#222">37</text>
<rect x="224" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<circle cx="192" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="143" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="300" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="300" y="143" font-size="16" text-anchor="middle" fill="#fff">2</text>
<circle cx="408" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="408" y="143" font-size="16" text-anchor="middle" fill="#fff">4</text>
<circle cx="516" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="516" y="143" font-size="16" text-anchor="middle" fill="#fff">6</text>
<circle cx="192" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="253" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="408" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="408" y="253" font-size="16" text-anchor="middle" fill="#fff">4</text>
<circle cx="192" cy="357" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="363" font-size="16" text-anchor="middle" fill="#fff">0</text>
<text x="156" y="88" font-size="16" text-anchor="end" fill="#333">Memoria global</text>
<text x="156" y="143" font-size="16" text-anchor="end" fill="#333">ID del thread</text>
<text x="156" y="161" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
<text x="156" y="271" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<text x="156" y="381" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 4</text>
</svg>

Solo suman los *threads* cuyo resultado se usa: $4 + 2 + 1 = 7$ sumas.

<!-- NOTA — misma entrada (N = 8) y mismo resultado (37) que la figura de
reduccion_global, para compararlas lado a lado: ahora en cada fila trabajan
solo los threads múltiplos de 2*stride (0, 2, 4, 6; después 0, 4; después 0).
Los demás elementos quedan como estaban, porque nadie los vuelve a leer. -->

<!-- NOTA — la figura (redibujada en SVG a partir de la del libro). Las sumas
están verificadas: stride 1 → 3+1=4, 8+4=12, 7+2=9,
9+3=12; stride 2 → 4+12=16, 9+12=21; stride 4 → 16+21=37. En los círculos,
los threads que suman (múltiplos de 2*stride: 0, 2, 4, 6; después 0, 4;
después 0). Las casillas lila guardan datos que ya no se usan; los demás
threads no hacen nada y no se dibujan. Comparada con la versión de
reduccion_global, la entrada y el resultado son los mismos con 7 sumas en vez
de 17. Un solo arreglo: cada fila es el mismo data después del lanzamiento. -->

---

## **Reducción paralela: menos threads trabajando**

Nueva versión del *kernel*:
@include[cuda]{static/code/threads/reduccion_global2.cu:4-10}

- Solo suma si `idx % (2*stride) == 0`: $N - 1$ sumas en total.
- **1 solo arreglo**: escribe en múltiplos de $2s$, lee múltiplos impares de $s$.
- Igual se lanza el *kernel* con **$N$ *threads*** en cada nivel.

<!-- NOTA — los tres cambios respecto de reduccion_global:

(1) La condición idx % (2*stride) == 0 deja trabajar solo a los threads cuyo
resultado se usa: N/2, N/4, ..., 1 sumas, N - 1 en total, en vez de ~N log N.

(2) Con esa condición se puede volver a trabajar en un solo arreglo (in place):
en un lanzamiento se escriben los índices múltiplos de 2*stride y se leen los
múltiplos impares de stride, y ningún índice está en los dos grupos. Por eso
ya no hacen falta d_in, d_out ni el intercambio de punteros.

(3) Lo que NO cambia: el host lanza n_bloques * 1024 = N threads en cada
nivel. Los threads que no cumplen la condición no suman, pero igual ocupan su
lugar en un warp. Con stride = 1 trabajan los threads pares: en cada warp hay
16 threads activos y 16 esperando, y todos los warps siguen ejecutándose. Esto
sigue así mientras 2*stride <= 32 (strides 1 a 16): cada warp tiene algún
thread activo. Solo desde stride = 32 hay warps enteros sin trabajo.

O sea, se hizo mucho menos trabajo útil pero no necesariamente en menos
tiempo: es el tema de las diapositivas siguientes (divergencia de warps), y lo
que corrige reduccion_global3 juntando a los threads activos en índices
contiguos. Todavía sin medir en la T4. -->

---

## **Divergencia de warps**

Este método sufre de **divergencia de threads**.

Si en un programa de CUDA tenemos algo como:

```cuda
if (threadIdx.x % 2 == 0) {
  // rama 1
} else {
  // rama 2
}
```

Los *threads* pares ejecutan la **rama 1** y los impares la **rama 2**: es `reduccion_global2` con `stride = 1`.

<!-- NOTA — el if en sí no es el problema; el problema es que la condición tome
valores distintos DENTRO de un mismo warp. Si todos los threads de un warp van
por la misma rama, no hay costo extra. En reduccion_global2 la rama "else" está
vacía, pero los threads impares igual esperan a que los pares terminen la
suma. -->

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

<svg viewBox="0 0 992 320" width="1140" style="display:block;margin:0.6em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
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

En `ncu`, la sección `SourceCounters` la muestra como *Branch Efficiency*:

```bash
ncu --section SourceCounters -c 1 ./reduccion_global2.x
```

<!-- NOTA — branch_efficiency (nvprof) es el porcentaje de saltos en que todo el
warp tomó el mismo camino; 100% es sin divergencia. En ncu la métrica
equivalente mide el promedio de threads activos en las ramas. Se puede pedir
que comparen reduccion_global2 con reduccion_global3 con esta métrica.

La sección SourceCounters ("source metrics, including branch efficiency and
sampled warp stall reasons", según la Profiling Guide de Nsight Compute)
muestra la línea Branch Efficiency junto con el número de saltos y los saltos
divergentes promedio. Que esa línea sea exactamente
smsp__sass_average_branch_targets_threads_uniform.pct no está verificado: para
confirmarlo, agregar --metrics smsp__sass_average_branch_targets_threads_uniform.pct
al mismo comando y comparar los dos números. -c 1 perfila solo el primer
lanzamiento (stride = 1, la mayor divergencia); sin él ncu imprime los 24
lanzamientos. -s 5 -c 1 perfila el de stride = 32, desde donde hay warps
enteros sin threads activos.

Medido en la T4 (2026-10), reduccion_global (la primera versión, sin -c 1, los
24 lanzamientos): Branch Efficiency 100% en todos, Avg. Divergent Branches 0.01
en los 5 primeros (stride 1 a 16) y 0 después. Es lo esperado: el if/else de
esa versión no depende de la paridad de idx sino de idx + stride < N, y la
frontera N - stride cae dentro de un warp solo si stride < 32; entonces
diverge 1 warp de 524 288. Branch Instructions = 1 048 576 = 2 por warp, más 2
en los lanzamientos con ese warp divergente, y más stride/32 desde stride = 32
(los warps de la cola que evalúan también el else if). Lección: la primera
versión desperdicia trabajo pero NO tiene divergencia; sirve de referencia
para reduccion_global2, donde idx % (2*stride) separa a los threads de cada
warp. ncu además reporta accesos no contiguos solo con stride 1, 2 y 4:
data_in[idx + stride] queda desalineado respecto de los sectores de 32 bytes y
cada warp toca 5 sectores en vez de 4 en esa lectura (524 287 sectores de
más, 8% de 6 815 745 = 13 por warp: 4 + 5 leídos y 4 escritos). Desde
stride = 8 la lectura queda alineada. reduccion_global2 y 3 todavía sin medir. -->

---

## **Reducción paralela: sin warp divergence**

Ejemplo: [reduccion_global3.cu](../code/threads/reduccion_global3.cu).

<svg viewBox="0 0 612 436" width="540" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-glob3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<text x="170" y="16" font-size="16" fill="#333">N = 8, n_threads = 4: n_bloques = 2</text>
<path d="M170,56 L170,50 L376,50 L376,56" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="273" y="44" font-size="15" text-anchor="middle" fill="#333">Bloque 0</text>
<path d="M386,56 L386,50 L592,50 L592,56" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="489" y="44" font-size="15" text-anchor="middle" fill="#333">Bloque 1</text>
<line x1="192" y1="102" x2="192" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="246" y1="102" x2="206.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="192" y1="153" x2="192" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="300" y1="102" x2="300" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="354" y1="102" x2="314.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="300" y1="153" x2="300" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="408" y1="102" x2="408" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="462" y1="102" x2="422.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="408" y1="153" x2="408" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="516" y1="102" x2="516" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="570" y1="102" x2="530.3" y2="127.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="516" y1="153" x2="516" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="192" y1="212" x2="192" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="300" y1="212" x2="208.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="192" y1="263" x2="192" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="408" y1="212" x2="408" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="516" y1="212" x2="424.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="408" y1="263" x2="408" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="192" y1="322" x2="192" y2="340" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="408" y1="322" x2="208.8" y2="354.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<line x1="192" y1="373" x2="192" y2="391" stroke="#444" stroke-width="1.4" marker-end="url(#fl-glob3)"/>
<rect x="170" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="88" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="224" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="88" font-size="18" text-anchor="middle" fill="#222">1</text>
<rect x="278" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="88" font-size="18" text-anchor="middle" fill="#222">8</text>
<rect x="332" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="88" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="386" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="88" font-size="18" text-anchor="middle" fill="#222">7</text>
<rect x="440" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="462" y="88" font-size="18" text-anchor="middle" fill="#222">2</text>
<rect x="494" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="88" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="548" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="570" y="88" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="170" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="198" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="224" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="198" font-size="18" text-anchor="middle" fill="#222">12</text>
<rect x="332" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="198" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="440" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="198" font-size="18" text-anchor="middle" fill="#222">12</text>
<rect x="548" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="308" font-size="18" text-anchor="middle" fill="#222">16</text>
<rect x="224" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="308" font-size="18" text-anchor="middle" fill="#222">21</text>
<rect x="440" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="392" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="418" font-size="18" text-anchor="middle" fill="#222">37</text>
<rect x="224" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<circle cx="192" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="143" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="300" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="300" y="143" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="408" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="408" y="143" font-size="16" text-anchor="middle" fill="#fff">2</text>
<circle cx="516" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="516" y="143" font-size="16" text-anchor="middle" fill="#fff">3</text>
<circle cx="192" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="253" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="408" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="408" y="253" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="192" cy="357" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="363" font-size="16" text-anchor="middle" fill="#fff">0</text>
<text x="156" y="88" font-size="16" text-anchor="end" fill="#333">Memoria global</text>
<text x="156" y="143" font-size="16" text-anchor="end" fill="#333">ID del thread</text>
<text x="156" y="161" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
<text x="156" y="271" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<text x="156" y="381" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 4</text>
</svg>

Se parece al anterior, pero cambia la indexación de los threads.


<!-- Puede ser problemático usar el índice global de los *threads* para un *array* muy grande... -->

<!-- NOTA — reduccion_global3.cu cambia QUÉ thread suma qué par: el thread idx suma
la posición 2*stride*idx. Así los threads que trabajan son siempre los
primeros (índices bajos) y quedan juntos en los mismos warps: los warps están
o completos o vacíos, y no hay divergencia. El problema del final de la
diapositiva es real: 2*stride*idx crece rápido, y para arreglos muy grandes no
cabe en un int, por eso el código usa unsigned long. -->

<!-- NOTA — la figura (redibujada en SVG a partir de la del libro). Las sumas
son las de reduccion_global2 (stride 1 → 4, 12, 9, 12;
stride 2 → 16, 21; stride 4 → 37), y las casillas escritas son las mismas;
lo que cambia es el número del círculo, el thread que hace cada suma: el thread idx
trabaja en la posición 2*stride*idx. Con stride 1 suman los threads 0, 1, 2,
3 (posiciones 0, 2, 4, 6); con stride 2, los threads 0 y 1 (posiciones 0 y
4); con stride 4, el thread 0. Siempre los primeros, y por eso los warps
quedan completos o vacíos. Comparar con la figura de reduccion_global2,
donde con stride 1 suman los threads 0, 2, 4, 6. -->

---

## **¿Por qué no hay divergencia?**

`reduccion_global2`: suman los `idx` múltiplos de $2s$, en **todos** los *warps*.

@include[cuda]{static/code/threads/reduccion_global2.cu:7-9}

`reduccion_global3`: el *thread* `idx` suma en `index` $= 2s \cdot$ `idx`, y `index < N` deja activos solo los `idx` $< N/(2s)$: cada *warp* queda **completo o vacío**.

@include[cuda]{static/code/threads/reduccion_global3.cu:10-13}

<!-- NOTA — la divergencia depende de cómo se reparten los threads activos entre
los warps, no de cuántos son. Las dos versiones hacen las mismas N - 1 sumas.

reduccion_global2: suma el thread con idx % (2*stride) == 0. Con stride = 1
son los pares: en cada warp 16 activos y 16 esperando, y TODOS los warps
divergen. Mientras 2*stride <= 32 (strides 1 a 16) cada warp tiene
32/(2*stride) threads activos; después cada thread activo queda solo en su
warp, y siguen divergiendo N/(2*stride) warps.

reduccion_global3: el thread idx trabaja sobre index = 2*stride*idx, y la
condición index < N equivale a idx < N/(2*stride). Los threads activos son
siempre los primeros N/(2*stride), un rango contiguo que empieza en 0. Como un
warp son 32 idx consecutivos, un warp está entero dentro del rango (todos
suman) o entero fuera (ninguno suma, y el if es uniforme). A lo más el warp
de la frontera diverge, y solo cuando N/(2*stride) no es múltiplo de 32: con
N = 2^24, en los 5 últimos niveles (stride >= 2^19), donde quedan menos de 32
threads activos, todos en el warp 0.

Lo que se movió es el trabajo: el thread 1 ya no suma data[1] + data[2] (eso
no sirve), sino data[2] + data[3] con stride = 1, data[4] + data[6] con
stride = 2, etc. Mismo árbol, otra asignación de threads a nodos. Los warps
vacíos igual se lanzan y evalúan el if, pero salen enseguida sin dividir a
nadie. Para medirlo: ncu --section SourceCounters -c 1 sobre
reduccion_global2.x y reduccion_global3.x. Todavía sin medir en la T4. -->

---

## **Threads activos por warp**

<svg viewBox="0 0 1133 412" width="1140" style="display:block;margin:0.3em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<text x="220" y="20" font-size="15" text-anchor="middle" fill="#333">warp 0</text>
<path d="M102,32 L102,26 L338,26 L338,32" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="480" y="20" font-size="15" text-anchor="middle" fill="#333">warp 1</text>
<path d="M362,32 L362,26 L598,26 L598,32" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="740" y="20" font-size="15" text-anchor="middle" fill="#333">warp 2</text>
<path d="M622,32 L622,26 L858,26 L858,32" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="1000" y="20" font-size="15" text-anchor="middle" fill="#333">warp 3</text>
<path d="M882,32 L882,26 L1118,26 L1118,32" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="10" y="64" font-size="18" fill="#333"><tspan font-weight="bold">reduccion_global2</tspan>: suman los <tspan font-family="monospace">idx % (2s) == 0</tspan></text>
<text x="87" y="97" font-size="15" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
<circle cx="115" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="115" y="96" font-size="13" text-anchor="middle" fill="#fff">0</text>
<circle cx="145" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="145" y="96" font-size="13" text-anchor="middle" fill="#888">1</text>
<circle cx="175" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="175" y="96" font-size="13" text-anchor="middle" fill="#fff">2</text>
<circle cx="205" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="205" y="96" font-size="13" text-anchor="middle" fill="#888">3</text>
<circle cx="235" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="235" y="96" font-size="13" text-anchor="middle" fill="#fff">4</text>
<circle cx="265" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="265" y="96" font-size="13" text-anchor="middle" fill="#888">5</text>
<circle cx="295" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="295" y="96" font-size="13" text-anchor="middle" fill="#fff">6</text>
<circle cx="325" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="325" y="96" font-size="13" text-anchor="middle" fill="#888">7</text>
<circle cx="375" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="375" y="96" font-size="13" text-anchor="middle" fill="#fff">8</text>
<circle cx="405" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="405" y="96" font-size="13" text-anchor="middle" fill="#888">9</text>
<circle cx="435" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="435" y="96" font-size="13" text-anchor="middle" fill="#fff">10</text>
<circle cx="465" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="465" y="96" font-size="13" text-anchor="middle" fill="#888">11</text>
<circle cx="495" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="495" y="96" font-size="13" text-anchor="middle" fill="#fff">12</text>
<circle cx="525" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="525" y="96" font-size="13" text-anchor="middle" fill="#888">13</text>
<circle cx="555" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="555" y="96" font-size="13" text-anchor="middle" fill="#fff">14</text>
<circle cx="585" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="585" y="96" font-size="13" text-anchor="middle" fill="#888">15</text>
<circle cx="635" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="635" y="96" font-size="13" text-anchor="middle" fill="#fff">16</text>
<circle cx="665" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="665" y="96" font-size="13" text-anchor="middle" fill="#888">17</text>
<circle cx="695" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="695" y="96" font-size="13" text-anchor="middle" fill="#fff">18</text>
<circle cx="725" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="725" y="96" font-size="13" text-anchor="middle" fill="#888">19</text>
<circle cx="755" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="755" y="96" font-size="13" text-anchor="middle" fill="#fff">20</text>
<circle cx="785" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="785" y="96" font-size="13" text-anchor="middle" fill="#888">21</text>
<circle cx="815" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="815" y="96" font-size="13" text-anchor="middle" fill="#fff">22</text>
<circle cx="845" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="845" y="96" font-size="13" text-anchor="middle" fill="#888">23</text>
<circle cx="895" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="895" y="96" font-size="13" text-anchor="middle" fill="#fff">24</text>
<circle cx="925" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="925" y="96" font-size="13" text-anchor="middle" fill="#888">25</text>
<circle cx="955" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="955" y="96" font-size="13" text-anchor="middle" fill="#fff">26</text>
<circle cx="985" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="985" y="96" font-size="13" text-anchor="middle" fill="#888">27</text>
<circle cx="1015" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="1015" y="96" font-size="13" text-anchor="middle" fill="#fff">28</text>
<circle cx="1045" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1045" y="96" font-size="13" text-anchor="middle" fill="#888">29</text>
<circle cx="1075" cy="92" r="13" fill="#e8641b" stroke="#444"/>
<text x="1075" y="96" font-size="13" text-anchor="middle" fill="#fff">30</text>
<circle cx="1105" cy="92" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1105" y="96" font-size="13" text-anchor="middle" fill="#888">31</text>
<text x="220" y="124" font-size="15" text-anchor="middle" font-weight="bold" fill="#c62828">diverge</text>
<text x="480" y="124" font-size="15" text-anchor="middle" font-weight="bold" fill="#c62828">diverge</text>
<text x="740" y="124" font-size="15" text-anchor="middle" font-weight="bold" fill="#c62828">diverge</text>
<text x="1000" y="124" font-size="15" text-anchor="middle" font-weight="bold" fill="#c62828">diverge</text>
<text x="87" y="159" font-size="15" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<circle cx="115" cy="154" r="13" fill="#e8641b" stroke="#444"/>
<text x="115" y="158" font-size="13" text-anchor="middle" fill="#fff">0</text>
<circle cx="145" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="145" y="158" font-size="13" text-anchor="middle" fill="#888">1</text>
<circle cx="175" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="175" y="158" font-size="13" text-anchor="middle" fill="#888">2</text>
<circle cx="205" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="205" y="158" font-size="13" text-anchor="middle" fill="#888">3</text>
<circle cx="235" cy="154" r="13" fill="#e8641b" stroke="#444"/>
<text x="235" y="158" font-size="13" text-anchor="middle" fill="#fff">4</text>
<circle cx="265" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="265" y="158" font-size="13" text-anchor="middle" fill="#888">5</text>
<circle cx="295" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="295" y="158" font-size="13" text-anchor="middle" fill="#888">6</text>
<circle cx="325" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="325" y="158" font-size="13" text-anchor="middle" fill="#888">7</text>
<circle cx="375" cy="154" r="13" fill="#e8641b" stroke="#444"/>
<text x="375" y="158" font-size="13" text-anchor="middle" fill="#fff">8</text>
<circle cx="405" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="405" y="158" font-size="13" text-anchor="middle" fill="#888">9</text>
<circle cx="435" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="435" y="158" font-size="13" text-anchor="middle" fill="#888">10</text>
<circle cx="465" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="465" y="158" font-size="13" text-anchor="middle" fill="#888">11</text>
<circle cx="495" cy="154" r="13" fill="#e8641b" stroke="#444"/>
<text x="495" y="158" font-size="13" text-anchor="middle" fill="#fff">12</text>
<circle cx="525" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="525" y="158" font-size="13" text-anchor="middle" fill="#888">13</text>
<circle cx="555" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="555" y="158" font-size="13" text-anchor="middle" fill="#888">14</text>
<circle cx="585" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="585" y="158" font-size="13" text-anchor="middle" fill="#888">15</text>
<circle cx="635" cy="154" r="13" fill="#e8641b" stroke="#444"/>
<text x="635" y="158" font-size="13" text-anchor="middle" fill="#fff">16</text>
<circle cx="665" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="665" y="158" font-size="13" text-anchor="middle" fill="#888">17</text>
<circle cx="695" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="695" y="158" font-size="13" text-anchor="middle" fill="#888">18</text>
<circle cx="725" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="725" y="158" font-size="13" text-anchor="middle" fill="#888">19</text>
<circle cx="755" cy="154" r="13" fill="#e8641b" stroke="#444"/>
<text x="755" y="158" font-size="13" text-anchor="middle" fill="#fff">20</text>
<circle cx="785" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="785" y="158" font-size="13" text-anchor="middle" fill="#888">21</text>
<circle cx="815" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="815" y="158" font-size="13" text-anchor="middle" fill="#888">22</text>
<circle cx="845" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="845" y="158" font-size="13" text-anchor="middle" fill="#888">23</text>
<circle cx="895" cy="154" r="13" fill="#e8641b" stroke="#444"/>
<text x="895" y="158" font-size="13" text-anchor="middle" fill="#fff">24</text>
<circle cx="925" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="925" y="158" font-size="13" text-anchor="middle" fill="#888">25</text>
<circle cx="955" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="955" y="158" font-size="13" text-anchor="middle" fill="#888">26</text>
<circle cx="985" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="985" y="158" font-size="13" text-anchor="middle" fill="#888">27</text>
<circle cx="1015" cy="154" r="13" fill="#e8641b" stroke="#444"/>
<text x="1015" y="158" font-size="13" text-anchor="middle" fill="#fff">28</text>
<circle cx="1045" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1045" y="158" font-size="13" text-anchor="middle" fill="#888">29</text>
<circle cx="1075" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1075" y="158" font-size="13" text-anchor="middle" fill="#888">30</text>
<circle cx="1105" cy="154" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1105" y="158" font-size="13" text-anchor="middle" fill="#888">31</text>
<text x="220" y="186" font-size="15" text-anchor="middle" font-weight="bold" fill="#c62828">diverge</text>
<text x="480" y="186" font-size="15" text-anchor="middle" font-weight="bold" fill="#c62828">diverge</text>
<text x="740" y="186" font-size="15" text-anchor="middle" font-weight="bold" fill="#c62828">diverge</text>
<text x="1000" y="186" font-size="15" text-anchor="middle" font-weight="bold" fill="#c62828">diverge</text>
<line x1="10" y1="222" x2="1123" y2="222" stroke="#ccc" stroke-width="1.2"/>
<text x="10" y="250" font-size="18" fill="#333"><tspan font-weight="bold">reduccion_global3</tspan>: suman los <tspan font-family="monospace">idx < N/(2s)</tspan></text>
<text x="87" y="283" font-size="15" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
<circle cx="115" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="115" y="282" font-size="13" text-anchor="middle" fill="#fff">0</text>
<circle cx="145" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="145" y="282" font-size="13" text-anchor="middle" fill="#fff">1</text>
<circle cx="175" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="175" y="282" font-size="13" text-anchor="middle" fill="#fff">2</text>
<circle cx="205" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="205" y="282" font-size="13" text-anchor="middle" fill="#fff">3</text>
<circle cx="235" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="235" y="282" font-size="13" text-anchor="middle" fill="#fff">4</text>
<circle cx="265" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="265" y="282" font-size="13" text-anchor="middle" fill="#fff">5</text>
<circle cx="295" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="295" y="282" font-size="13" text-anchor="middle" fill="#fff">6</text>
<circle cx="325" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="325" y="282" font-size="13" text-anchor="middle" fill="#fff">7</text>
<circle cx="375" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="375" y="282" font-size="13" text-anchor="middle" fill="#fff">8</text>
<circle cx="405" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="405" y="282" font-size="13" text-anchor="middle" fill="#fff">9</text>
<circle cx="435" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="435" y="282" font-size="13" text-anchor="middle" fill="#fff">10</text>
<circle cx="465" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="465" y="282" font-size="13" text-anchor="middle" fill="#fff">11</text>
<circle cx="495" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="495" y="282" font-size="13" text-anchor="middle" fill="#fff">12</text>
<circle cx="525" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="525" y="282" font-size="13" text-anchor="middle" fill="#fff">13</text>
<circle cx="555" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="555" y="282" font-size="13" text-anchor="middle" fill="#fff">14</text>
<circle cx="585" cy="278" r="13" fill="#e8641b" stroke="#444"/>
<text x="585" y="282" font-size="13" text-anchor="middle" fill="#fff">15</text>
<circle cx="635" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="635" y="282" font-size="13" text-anchor="middle" fill="#888">16</text>
<circle cx="665" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="665" y="282" font-size="13" text-anchor="middle" fill="#888">17</text>
<circle cx="695" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="695" y="282" font-size="13" text-anchor="middle" fill="#888">18</text>
<circle cx="725" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="725" y="282" font-size="13" text-anchor="middle" fill="#888">19</text>
<circle cx="755" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="755" y="282" font-size="13" text-anchor="middle" fill="#888">20</text>
<circle cx="785" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="785" y="282" font-size="13" text-anchor="middle" fill="#888">21</text>
<circle cx="815" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="815" y="282" font-size="13" text-anchor="middle" fill="#888">22</text>
<circle cx="845" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="845" y="282" font-size="13" text-anchor="middle" fill="#888">23</text>
<circle cx="895" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="895" y="282" font-size="13" text-anchor="middle" fill="#888">24</text>
<circle cx="925" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="925" y="282" font-size="13" text-anchor="middle" fill="#888">25</text>
<circle cx="955" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="955" y="282" font-size="13" text-anchor="middle" fill="#888">26</text>
<circle cx="985" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="985" y="282" font-size="13" text-anchor="middle" fill="#888">27</text>
<circle cx="1015" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1015" y="282" font-size="13" text-anchor="middle" fill="#888">28</text>
<circle cx="1045" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1045" y="282" font-size="13" text-anchor="middle" fill="#888">29</text>
<circle cx="1075" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1075" y="282" font-size="13" text-anchor="middle" fill="#888">30</text>
<circle cx="1105" cy="278" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1105" y="282" font-size="13" text-anchor="middle" fill="#888">31</text>
<text x="220" y="310" font-size="15" text-anchor="middle" font-weight="bold" fill="#2e7d32">completo</text>
<text x="480" y="310" font-size="15" text-anchor="middle" font-weight="bold" fill="#2e7d32">completo</text>
<text x="740" y="310" font-size="15" text-anchor="middle" font-weight="bold" fill="#757575">vacío</text>
<text x="1000" y="310" font-size="15" text-anchor="middle" font-weight="bold" fill="#757575">vacío</text>
<text x="87" y="345" font-size="15" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<circle cx="115" cy="340" r="13" fill="#e8641b" stroke="#444"/>
<text x="115" y="344" font-size="13" text-anchor="middle" fill="#fff">0</text>
<circle cx="145" cy="340" r="13" fill="#e8641b" stroke="#444"/>
<text x="145" y="344" font-size="13" text-anchor="middle" fill="#fff">1</text>
<circle cx="175" cy="340" r="13" fill="#e8641b" stroke="#444"/>
<text x="175" y="344" font-size="13" text-anchor="middle" fill="#fff">2</text>
<circle cx="205" cy="340" r="13" fill="#e8641b" stroke="#444"/>
<text x="205" y="344" font-size="13" text-anchor="middle" fill="#fff">3</text>
<circle cx="235" cy="340" r="13" fill="#e8641b" stroke="#444"/>
<text x="235" y="344" font-size="13" text-anchor="middle" fill="#fff">4</text>
<circle cx="265" cy="340" r="13" fill="#e8641b" stroke="#444"/>
<text x="265" y="344" font-size="13" text-anchor="middle" fill="#fff">5</text>
<circle cx="295" cy="340" r="13" fill="#e8641b" stroke="#444"/>
<text x="295" y="344" font-size="13" text-anchor="middle" fill="#fff">6</text>
<circle cx="325" cy="340" r="13" fill="#e8641b" stroke="#444"/>
<text x="325" y="344" font-size="13" text-anchor="middle" fill="#fff">7</text>
<circle cx="375" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="375" y="344" font-size="13" text-anchor="middle" fill="#888">8</text>
<circle cx="405" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="405" y="344" font-size="13" text-anchor="middle" fill="#888">9</text>
<circle cx="435" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="435" y="344" font-size="13" text-anchor="middle" fill="#888">10</text>
<circle cx="465" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="465" y="344" font-size="13" text-anchor="middle" fill="#888">11</text>
<circle cx="495" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="495" y="344" font-size="13" text-anchor="middle" fill="#888">12</text>
<circle cx="525" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="525" y="344" font-size="13" text-anchor="middle" fill="#888">13</text>
<circle cx="555" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="555" y="344" font-size="13" text-anchor="middle" fill="#888">14</text>
<circle cx="585" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="585" y="344" font-size="13" text-anchor="middle" fill="#888">15</text>
<circle cx="635" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="635" y="344" font-size="13" text-anchor="middle" fill="#888">16</text>
<circle cx="665" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="665" y="344" font-size="13" text-anchor="middle" fill="#888">17</text>
<circle cx="695" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="695" y="344" font-size="13" text-anchor="middle" fill="#888">18</text>
<circle cx="725" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="725" y="344" font-size="13" text-anchor="middle" fill="#888">19</text>
<circle cx="755" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="755" y="344" font-size="13" text-anchor="middle" fill="#888">20</text>
<circle cx="785" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="785" y="344" font-size="13" text-anchor="middle" fill="#888">21</text>
<circle cx="815" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="815" y="344" font-size="13" text-anchor="middle" fill="#888">22</text>
<circle cx="845" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="845" y="344" font-size="13" text-anchor="middle" fill="#888">23</text>
<circle cx="895" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="895" y="344" font-size="13" text-anchor="middle" fill="#888">24</text>
<circle cx="925" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="925" y="344" font-size="13" text-anchor="middle" fill="#888">25</text>
<circle cx="955" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="955" y="344" font-size="13" text-anchor="middle" fill="#888">26</text>
<circle cx="985" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="985" y="344" font-size="13" text-anchor="middle" fill="#888">27</text>
<circle cx="1015" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1015" y="344" font-size="13" text-anchor="middle" fill="#888">28</text>
<circle cx="1045" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1045" y="344" font-size="13" text-anchor="middle" fill="#888">29</text>
<circle cx="1075" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1075" y="344" font-size="13" text-anchor="middle" fill="#888">30</text>
<circle cx="1105" cy="340" r="13" fill="#ececec" stroke="#aaa"/>
<text x="1105" y="344" font-size="13" text-anchor="middle" fill="#888">31</text>
<text x="220" y="372" font-size="15" text-anchor="middle" font-weight="bold" fill="#2e7d32">completo</text>
<text x="480" y="372" font-size="15" text-anchor="middle" font-weight="bold" fill="#757575">vacío</text>
<text x="740" y="372" font-size="15" text-anchor="middle" font-weight="bold" fill="#757575">vacío</text>
<text x="1000" y="372" font-size="15" text-anchor="middle" font-weight="bold" fill="#757575">vacío</text>
</svg>

Naranjo: *threads* que suman. *Warps* de $8$ *threads* para que quepan en la figura (en la T4 son $32$), con $N = 32$.

<!-- NOTA — la misma cantidad de sumas en las dos versiones (16 con stride 1, 8
con stride 2), repartida distinto. En reduccion_global2 los threads activos
son los idx múltiplos de 2s, intercalados: TODOS los warps tienen activos e
inactivos, y todos divergen (los dos lados del if se ejecutan en serie).
En reduccion_global3 los activos son los primeros N/(2s) idx, un rango
contiguo que empieza en 0: con stride 1 los warps 0 y 1 quedan completos y
el 2 y el 3 vacíos; con stride 2 solo el warp 0 trabaja. Un warp vacío evalúa
el if y sale, sin dividirse. Con warps de 32 la frontera N/(2s) cae justo
entre warps mientras N/(2s) sea múltiplo de 32; con N = 2^24 eso vale hasta
stride = 2^18, y en los últimos 5 niveles solo diverge el warp 0. Ojo: los
números de los círculos son el idx del thread, no la casilla de data que toca
(en reduccion_global3 el thread idx suma en index = 2s·idx). -->

---

## **¿Por qué usar `unsigned long` para `index`?**

@include[cuda]{static/code/threads/reduccion_global3.cu:6-13}

Con un arreglo de largo $N=2^{24}$:
- **Todos** los *threads* calculan `index`: hasta $2^{24} \cdot 2^{24} = 2^{48}$.
- En $32$ bits, `stride` $= 2^{23}$ e `idx` $= 256$ dan $2^{32} \to 0$.
- Ese *thread* también suma en `data[0]`: **resultado incorrecto**.

<!-- NOTA — la respuesta a "puede ser problemático usar el índice global" de la
diapositiva anterior. En reduccion_global3 el índice del elemento ya no es el
índice global del thread, sino index = 2*stride*idx, y se calcula en todos los
N threads lanzados, no solo en los N/(2*stride) que trabajan. Con N = 2^24,
idx llega a 2^24 - 1 y 2*stride a 2^24 en el último nivel, así que el producto
llega a ~2^48: no cabe en 32 bits.

El ejemplo del slide: en el último nivel (stride = 2^23), el thread idx = 256
calcula 2^24 * 2^8 = 2^32, que en 32 bits sin signo vale 0 (con int con signo
es comportamiento indefinido; en la práctica también da la vuelta). Entonces
index = 0 < N y el thread ejecuta data[0] += data[2^23], igual que el thread
0. Lo mismo pasa con todos los múltiplos de 256: 2^24/256 = 65 536 threads
sumando sobre data[0] a la vez, en una race condition y contando de nuevo la
segunda mitad. En niveles anteriores el producto da la vuelta a valores
menores que N (a veces 0, a veces no), y esos threads modifican elementos
que no les tocan.

Por qué basta unsigned long: idx es de 64 bits, y en 2 * stride * idx primero
se calcula 2 * stride en int (a lo más 2^24, cabe) y luego se multiplica por
idx convirtiendo a 64 bits. Si se declarara idx como int, el producto
completo se haría en 32 bits y el resultado saldría mal sin ningún aviso ni
error. El costo: la aritmética entera de 64 bits es más lenta en el GPU que la
de 32. Una alternativa es comprobar antes de multiplicar, idx < N / (2*stride),
que nunca desborda.

Otro problema de este índice, que no es de desborde: los threads de un warp
acceden a posiciones separadas por 2*stride, así que con stride grande cada
thread toca su propio sector de 32 bytes y los accesos no son contiguos. Las
versiones siguientes (con bloques y memoria compartida) lo atacan. -->

---

## **Reducción paralela: un árbol por bloque**

Ejemplo: [reduccion_global4.cu](../code/threads/reduccion_global4.cu).

- Antes: un árbol sobre **todo** el arreglo y un lanzamiento por nivel.
- Ahora: cada bloque reduce **su parte** en un solo lanzamiento.

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

<!-- <p class="credit">Fuente: adaptado de <em>Professional CUDA C Programming</em></p> -->

<!-- NOTA — reduccion_global4.cu cambia de estrategia: cada bloque reduce SU parte
del arreglo dentro de un solo kernel, sincronizando con __syncthreads() entre
niveles, y deja un resultado parcial por bloque; la suma de los parciales se
hace en el host. Detalle corregido en el código: los parciales se guardan en
un arreglo aparte, porque escribirlos en data[blockIdx.x] pisaba datos que el
bloque 0 todavía estaba sumando. Un solo lanzamiento en vez de 24. Las
versiones anteriores también lanzaban bloques, pero el árbol abarcaba el
arreglo entero y los niveles se sincronizaban relanzando el kernel; como
__syncthreads() solo sincroniza un bloque, hacerlo en un solo lanzamiento
obliga a un árbol por bloque. Ese es el cambio que nombra el título. -->

<!-- NOTA — la figura (redibujada en SVG a partir de la del libro). Arriba, en el device:
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

## **Reducción paralela: un árbol por bloque**

@include[cuda]{static/code/threads/reduccion_global4.cu:18-28}

- **Un solo lanzamiento**: el ciclo sobre el *stride* está dentro del *kernel*.
- `idata` apunta a la parte del bloque.
  - Al final: `parcial[blockIdx.x] = idata[0]`.

<!-- NOTA — los cambios respecto de reduccion_global3: (1) el ciclo sobre el
stride pasa del host al kernel, así que hay un lanzamiento en vez de 24;
(2) como los niveles ya no están separados por lanzamientos, hace falta
__syncthreads() para que un nivel empiece cuando todo el bloque terminó el
anterior, y __syncthreads() solo sincroniza un bloque: por eso cada bloque
reduce SU parte (idata apunta al comienzo de ella) y deja un parcial;
(3) el parcial va a un arreglo aparte, parcial[blockIdx.x], porque
escribirlo en data[blockIdx.x] pisaría datos que el bloque 0 todavía está
sumando. El índice sigue siendo el de reduccion_global3 (2*stride*t), ahora
con threadIdx.x. El comentario del código sobre el if vale la pena: la
barrera queda dentro de if (idx < N), lo que solo es correcto porque N es
múltiplo de blockDim.x; con otro N la barrera sería divergente y el
comportamiento no está definido (la diapositiva muestra solo el ciclo; el if
y la escritura del parcial están en el archivo). -->

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

Esta reducción es del tipo 'barrido' (desde derecha a izquierda).

<!-- <p class="credit">Fuente: adaptado de <em>Professional CUDA C Programming</em></p> -->

<!-- NOTA — reduccion_global5.cu invierte el orden de los strides: empieza con
stride = blockDim/2 y lo va dividiendo por 2, y suma el thread threadIdx.x con
threadIdx.x + stride. Así los threads activos son siempre los primeros (sin
divergencia) Y cada warp lee posiciones consecutivas (acceso contiguo), lo
que conecta con el capítulo de memoria. Es el patrón que se usa en todas las
versiones con memoria global que siguen (la de memoria compartida, que viene
ahora, vuelve a propósito a la indexación de reduccion_global2). -->

<!-- NOTA — la figura (redibujada en SVG a partir de la del libro); las sumas están verificadas: con stride 4, 3+4=7, 1+1=2, 7+6=13 y 0+3=3; con
stride 2, 7+13=20 y 2+3=5; con stride 1, 20+5=25, que es la suma de los ocho
datos. Es exactamente lo que hace reduccion_global5.cu: el stride empieza en
blockDim/2 y se divide por 2, y el thread t suma la casilla t + stride sobre
la casilla t. Dos cosas que mirar: los threads que trabajan son siempre los
primeros (0 a stride-1), así que no hay divergencia dentro de los warps
mientras stride sea al menos 32; y las lecturas de un paso son dos tramos
contiguos (t y t + stride), así que el acceso es contiguo. Las casillas lila
quedan con datos que ya no se usan. -->

---

## **Reducción paralela: acceso contiguo**

@include[cuda]{static/code/threads/reduccion_global5.cu:13-22}

- Respecto de la versión anterior (`reduccion_global4`) solo cambia el **orden de los *strides*** y el **índice**.
- Los *threads* activos son los primeros y leen **posiciones contiguas**.

<!-- NOTA — es el mismo kernel que reduccion_global4 (mismo idata, mismo
__syncthreads(), mismo parcial) con el ciclo invertido. Comparar las dos
versiones línea a línea: en global4 el thread t trabaja en 2*stride*t y lee
2*stride*t + stride, posiciones separadas por 2*stride; en global5 lee t y
t + stride, dos tramos contiguos. Con el mismo número de sumas y sin
divergencia en ninguno de los dos, la diferencia de tiempo es solo de acceso
a memoria. -->

---

## **Reducción paralela: memoria compartida**

Hasta ahora, hemos usado solo memoria global.
Ahora un ejemplo con memoria compartida: [reduccion_compartida.cu](../code/threads/reduccion_compartida.cu).

@include[cuda]{static/code/threads/reduccion_compartida.cu:9-16}

- Cada bloque copia su parte: **cada elemento se lee desde la memoria global una sola vez**.

<!-- NOTA — reduccion_compartida.cu copia el bloque a memoria compartida y hace el
árbol ahí, así la memoria global se lee una sola vez por elemento. Es una
versión más simple que las anteriores (usa la indexación con divergencia de
reduccion_global2) para aislar la idea de memoria compartida. Se lanza en
varias pasadas hasta que queda un número; desde la corrección, cada pasada
escribe sus parciales en otro arreglo y los punteros se intercambian.

La declaración extern __shared__ es la declaración dinámica del capítulo de
memoria (diapositiva "Memoria compartida — declaración dinámica", la misma
de transpuestaCompDin): el tamaño se fija al lanzar el kernel, no en el
código. Fuera del arreglo se copia un 0, que no cambia la suma. El
__syncthreads() es indispensable: ningún thread puede empezar a sumar
s_data[t + stride] antes de que el thread t + stride lo haya copiado. -->

---

## **Memoria compartida: el árbol**

@include[cuda]{static/code/threads/reduccion_compartida.cu:20-29}

- El mismo árbol de `reduccion_global2` (con su divergencia, a propósito), ahora sobre `s_data`.
- Al final, `salida[blockIdx.x] = s_data[0]`: un resultado por bloque.

<!-- NOTA — lo único que cambia respecto de reduccion_global2 es la memoria:
el árbol, la condición y la divergencia son los mismos, para que la
diferencia de tiempo se pueda atribuir a la memoria compartida. La
condición usa idx_x (el índice global) y no threadIdx.x, pero da lo mismo:
idx_x = blockIdx.x * blockDim.x + threadIdx.x, y blockDim.x es múltiplo de
2*stride, así que idx_x % (2*stride) == threadIdx.x % (2*stride). Una
diferencia con reduccion_global4 y 5: aquí el __syncthreads() está fuera de
cualquier if, y el último bloque se completa con ceros al copiar, así que el
kernel es correcto para cualquier N, no solo para múltiplos de blockDim.x.
El resultado del bloque va a un arreglo aparte (salida) por la misma razón
que en global4–8: escribirlo en data[blockIdx.x] pisaría datos que otro
bloque todavía no copia. -->

---

## **Memoria compartida: el recorrido de los datos**

<svg viewBox="0 0 1004 504" width="1000" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-comp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<rect x="172" y="114" width="406" height="302" rx="8" fill="#e3ecf9" stroke="#9fb6d8"/>
<text x="180" y="132" font-size="14" fill="#2f4f7f">memoria compartida del bloque 0 (<tspan font-family="monospace">s_data</tspan>)</text>
<text x="375" y="28" font-size="15" text-anchor="middle" fill="#333">bloque 0</text>
<rect x="586" y="114" width="406" height="302" rx="8" fill="#e3ecf9" stroke="#9fb6d8"/>
<text x="594" y="132" font-size="14" fill="#2f4f7f">memoria compartida del bloque 1 (<tspan font-family="monospace">s_data</tspan>)</text>
<text x="789" y="28" font-size="15" text-anchor="middle" fill="#333">bloque 1</text>
<line x1="200.0" y1="74.0" x2="200.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="250.0" y1="74.0" x2="250.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="300.0" y1="74.0" x2="300.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="350.0" y1="74.0" x2="350.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="400.0" y1="74.0" x2="400.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="450.0" y1="74.0" x2="450.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="500.0" y1="74.0" x2="500.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="550.0" y1="74.0" x2="550.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="614.0" y1="74.0" x2="614.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="664.0" y1="74.0" x2="664.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="714.0" y1="74.0" x2="714.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="764.0" y1="74.0" x2="764.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="814.0" y1="74.0" x2="814.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="864.0" y1="74.0" x2="864.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="914.0" y1="74.0" x2="914.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="964.0" y1="74.0" x2="964.0" y2="112.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="200.0" y1="178.0" x2="200.0" y2="184.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="250.0" y1="178.0" x2="213.8" y2="193.2" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="200.0" y1="213.0" x2="200.0" y2="219.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="300.0" y1="178.0" x2="300.0" y2="184.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="350.0" y1="178.0" x2="313.8" y2="193.2" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="300.0" y1="213.0" x2="300.0" y2="219.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="400.0" y1="178.0" x2="400.0" y2="184.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="450.0" y1="178.0" x2="413.8" y2="193.2" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="400.0" y1="213.0" x2="400.0" y2="219.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="500.0" y1="178.0" x2="500.0" y2="184.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="550.0" y1="178.0" x2="513.8" y2="193.2" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="500.0" y1="213.0" x2="500.0" y2="219.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="614.0" y1="178.0" x2="614.0" y2="184.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="664.0" y1="178.0" x2="627.8" y2="193.2" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="614.0" y1="213.0" x2="614.0" y2="219.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="714.0" y1="178.0" x2="714.0" y2="184.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="764.0" y1="178.0" x2="727.8" y2="193.2" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="714.0" y1="213.0" x2="714.0" y2="219.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="814.0" y1="178.0" x2="814.0" y2="184.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="864.0" y1="178.0" x2="827.8" y2="193.2" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="814.0" y1="213.0" x2="814.0" y2="219.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="914.0" y1="178.0" x2="914.0" y2="184.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="964.0" y1="178.0" x2="927.8" y2="193.2" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="914.0" y1="213.0" x2="914.0" y2="219.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<text x="160" y="204" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
<line x1="200.0" y1="254.0" x2="200.0" y2="260.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="300.0" y1="254.0" x2="214.7" y2="271.9" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="200.0" y1="289.0" x2="200.0" y2="295.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="400.0" y1="254.0" x2="400.0" y2="260.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="500.0" y1="254.0" x2="414.7" y2="271.9" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="400.0" y1="289.0" x2="400.0" y2="295.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="614.0" y1="254.0" x2="614.0" y2="260.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="714.0" y1="254.0" x2="628.7" y2="271.9" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="614.0" y1="289.0" x2="614.0" y2="295.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="814.0" y1="254.0" x2="814.0" y2="260.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="914.0" y1="254.0" x2="828.7" y2="271.9" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="814.0" y1="289.0" x2="814.0" y2="295.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<text x="160" y="280" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<line x1="200.0" y1="330.0" x2="200.0" y2="336.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="400.0" y1="330.0" x2="214.9" y2="349.4" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="200.0" y1="365.0" x2="200.0" y2="371.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="614.0" y1="330.0" x2="614.0" y2="336.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="814.0" y1="330.0" x2="628.9" y2="349.4" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<line x1="614.0" y1="365.0" x2="614.0" y2="371.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<text x="160" y="356" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 4</text>
<rect x="180" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="200" y="167" font-size="17" text-anchor="middle" fill="#222">3</text>
<rect x="230" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="250" y="167" font-size="17" text-anchor="middle" fill="#222">1</text>
<rect x="280" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="167" font-size="17" text-anchor="middle" fill="#222">8</text>
<rect x="330" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="350" y="167" font-size="17" text-anchor="middle" fill="#222">4</text>
<rect x="380" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="400" y="167" font-size="17" text-anchor="middle" fill="#222">7</text>
<rect x="430" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="450" y="167" font-size="17" text-anchor="middle" fill="#222">2</text>
<rect x="480" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="500" y="167" font-size="17" text-anchor="middle" fill="#222">9</text>
<rect x="530" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="550" y="167" font-size="17" text-anchor="middle" fill="#222">3</text>
<rect x="594" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="614" y="167" font-size="17" text-anchor="middle" fill="#222">2</text>
<rect x="644" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="664" y="167" font-size="17" text-anchor="middle" fill="#222">4</text>
<rect x="694" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="714" y="167" font-size="17" text-anchor="middle" fill="#222">6</text>
<rect x="744" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="764" y="167" font-size="17" text-anchor="middle" fill="#222">1</text>
<rect x="794" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="814" y="167" font-size="17" text-anchor="middle" fill="#222">8</text>
<rect x="844" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="864" y="167" font-size="17" text-anchor="middle" fill="#222">7</text>
<rect x="894" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="914" y="167" font-size="17" text-anchor="middle" fill="#222">9</text>
<rect x="944" y="144" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="964" y="167" font-size="17" text-anchor="middle" fill="#222">5</text>
<rect x="180" y="220" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="200" y="243" font-size="17" text-anchor="middle" fill="#222">4</text>
<rect x="230" y="220" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="280" y="220" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="243" font-size="17" text-anchor="middle" fill="#222">12</text>
<rect x="330" y="220" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="380" y="220" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="400" y="243" font-size="17" text-anchor="middle" fill="#222">9</text>
<rect x="430" y="220" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="480" y="220" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="500" y="243" font-size="17" text-anchor="middle" fill="#222">12</text>
<rect x="530" y="220" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="594" y="220" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="614" y="243" font-size="17" text-anchor="middle" fill="#222">6</text>
<rect x="644" y="220" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="694" y="220" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="714" y="243" font-size="17" text-anchor="middle" fill="#222">7</text>
<rect x="744" y="220" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="794" y="220" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="814" y="243" font-size="17" text-anchor="middle" fill="#222">15</text>
<rect x="844" y="220" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="894" y="220" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="914" y="243" font-size="17" text-anchor="middle" fill="#222">14</text>
<rect x="944" y="220" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="180" y="296" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="200" y="319" font-size="17" text-anchor="middle" fill="#222">16</text>
<rect x="230" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="280" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="330" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="380" y="296" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="400" y="319" font-size="17" text-anchor="middle" fill="#222">21</text>
<rect x="430" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="480" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="530" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="594" y="296" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="614" y="319" font-size="17" text-anchor="middle" fill="#222">13</text>
<rect x="644" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="694" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="744" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="794" y="296" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="814" y="319" font-size="17" text-anchor="middle" fill="#222">29</text>
<rect x="844" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="894" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="944" y="296" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="180" y="372" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="200" y="395" font-size="17" text-anchor="middle" fill="#222">37</text>
<rect x="230" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="280" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="330" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="380" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="430" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="480" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="530" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="594" y="372" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="614" y="395" font-size="17" text-anchor="middle" fill="#222">42</text>
<rect x="644" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="694" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="744" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="794" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="844" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="894" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="944" y="372" width="40" height="34" rx="3" fill="#e9e1f0" stroke="#555"/>
<circle cx="200" cy="199" r="14" fill="#e8641b" stroke="#444"/>
<text x="200" y="204" font-size="14" text-anchor="middle" fill="#fff">0</text>
<circle cx="300" cy="199" r="14" fill="#e8641b" stroke="#444"/>
<text x="300" y="204" font-size="14" text-anchor="middle" fill="#fff">2</text>
<circle cx="400" cy="199" r="14" fill="#e8641b" stroke="#444"/>
<text x="400" y="204" font-size="14" text-anchor="middle" fill="#fff">4</text>
<circle cx="500" cy="199" r="14" fill="#e8641b" stroke="#444"/>
<text x="500" y="204" font-size="14" text-anchor="middle" fill="#fff">6</text>
<circle cx="614" cy="199" r="14" fill="#e8641b" stroke="#444"/>
<text x="614" y="204" font-size="14" text-anchor="middle" fill="#fff">0</text>
<circle cx="714" cy="199" r="14" fill="#e8641b" stroke="#444"/>
<text x="714" y="204" font-size="14" text-anchor="middle" fill="#fff">2</text>
<circle cx="814" cy="199" r="14" fill="#e8641b" stroke="#444"/>
<text x="814" y="204" font-size="14" text-anchor="middle" fill="#fff">4</text>
<circle cx="914" cy="199" r="14" fill="#e8641b" stroke="#444"/>
<text x="914" y="204" font-size="14" text-anchor="middle" fill="#fff">6</text>
<circle cx="200" cy="275" r="14" fill="#e8641b" stroke="#444"/>
<text x="200" y="280" font-size="14" text-anchor="middle" fill="#fff">0</text>
<circle cx="400" cy="275" r="14" fill="#e8641b" stroke="#444"/>
<text x="400" y="280" font-size="14" text-anchor="middle" fill="#fff">4</text>
<circle cx="614" cy="275" r="14" fill="#e8641b" stroke="#444"/>
<text x="614" y="280" font-size="14" text-anchor="middle" fill="#fff">0</text>
<circle cx="814" cy="275" r="14" fill="#e8641b" stroke="#444"/>
<text x="814" y="280" font-size="14" text-anchor="middle" fill="#fff">4</text>
<circle cx="200" cy="351" r="14" fill="#e8641b" stroke="#444"/>
<text x="200" y="356" font-size="14" text-anchor="middle" fill="#fff">0</text>
<circle cx="614" cy="351" r="14" fill="#e8641b" stroke="#444"/>
<text x="614" y="356" font-size="14" text-anchor="middle" fill="#fff">0</text>
<rect x="180" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="200" y="63" font-size="17" text-anchor="middle" fill="#222">3</text>
<rect x="230" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="250" y="63" font-size="17" text-anchor="middle" fill="#222">1</text>
<rect x="280" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="63" font-size="17" text-anchor="middle" fill="#222">8</text>
<rect x="330" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="350" y="63" font-size="17" text-anchor="middle" fill="#222">4</text>
<rect x="380" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="400" y="63" font-size="17" text-anchor="middle" fill="#222">7</text>
<rect x="430" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="450" y="63" font-size="17" text-anchor="middle" fill="#222">2</text>
<rect x="480" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="500" y="63" font-size="17" text-anchor="middle" fill="#222">9</text>
<rect x="530" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="550" y="63" font-size="17" text-anchor="middle" fill="#222">3</text>
<rect x="594" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="614" y="63" font-size="17" text-anchor="middle" fill="#222">2</text>
<rect x="644" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="664" y="63" font-size="17" text-anchor="middle" fill="#222">4</text>
<rect x="694" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="714" y="63" font-size="17" text-anchor="middle" fill="#222">6</text>
<rect x="744" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="764" y="63" font-size="17" text-anchor="middle" fill="#222">1</text>
<rect x="794" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="814" y="63" font-size="17" text-anchor="middle" fill="#222">8</text>
<rect x="844" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="864" y="63" font-size="17" text-anchor="middle" fill="#222">7</text>
<rect x="894" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="914" y="63" font-size="17" text-anchor="middle" fill="#222">9</text>
<rect x="944" y="40" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="964" y="63" font-size="17" text-anchor="middle" fill="#222">5</text>
<line x1="200.0" y1="406.0" x2="200.0" y2="461.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<rect x="180" y="462" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="200" y="485" font-size="17" text-anchor="middle" fill="#222">37</text>
<line x1="614.0" y1="406.0" x2="250.0" y2="461.0" stroke="#444" stroke-width="1.4" marker-end="url(#fl-comp)"/>
<rect x="230" y="462" width="40" height="34" rx="3" fill="#eef3df" stroke="#555"/>
<text x="250" y="485" font-size="17" text-anchor="middle" fill="#222">42</text>
<text x="286" y="485" font-size="15" fill="#333">un resultado por bloque; la pasada siguiente reduce <tspan font-family="monospace">salida</tspan></text>
<text x="160" y="63" font-size="15" text-anchor="end" fill="#333">data (global)</text>
<text x="160" y="90" font-size="13" text-anchor="end" fill="#666" font-style="italic">copia</text>
<text x="160" y="106" font-size="13" text-anchor="end" fill="#666" font-style="italic">+ __syncthreads()</text>
<text x="160" y="167" font-size="15" text-anchor="end" fill="#333">s_data</text>
<text x="160" y="485" font-size="15" text-anchor="end" fill="#333">salida (global)</text>
</svg>

<!-- NOTA — la figura sigue a reduccion_compartida.cu con N = 16 y bloques de 8
(en el programa son bloques de 1024). Tres etapas: (1) cada thread copia UN
elemento de data (memoria global) a s_data, la memoria compartida de su
bloque, y __syncthreads() espera a que la copia esté completa; es la única
lectura de memoria global de cada elemento. (2) El árbol se hace entero en
s_data, con la misma indexación de reduccion_global2 (suman los threadIdx.x
múltiplos de 2*stride), así que con warps de 32 hay divergencia, a
propósito: lo único que cambia respecto de global2 es la memoria. Las
casillas lila ya no se usan. (3) El thread 0 de cada bloque escribe
s_data[0] en salida[blockIdx.x], en memoria global y en un arreglo aparte.
Las sumas: bloque 0 → 4, 12, 9, 12; 16, 21; 37. Bloque 1 → 6, 7, 15, 14;
13, 29; 42. Con N = 2^24 la primera pasada deja 16384 valores en salida, y
el host vuelve a lanzar el kernel sobre ellos (diapositiva siguiente) hasta
que queda uno. Cada bloque tiene su propia memoria compartida: el bloque 1
no ve el s_data del bloque 0, y por eso hace falta escribir en global para
juntar los resultados. -->

---

## **Memoria compartida: varias pasadas**

@include[cuda]{static/code/threads/reduccion_compartida.cu:83-92}

- **Tercer argumento** de `<<<...>>>`: memoria compartida por bloque (*bytes*).
- $2^{24} \to 2^{14} \to 2^{4} \to 1$: **3 lanzamientos**, y la suma termina en el GPU.

<!-- NOTA — el tercer argumento de la configuración de lanzamiento,
n_threads * sizeof(float), es el tamaño de s_data: la declaración extern
__shared__ no tiene tamaño, y se lo da el lanzamiento. Las cuentas con
N = 2^24 y bloques de 1024: la primera pasada deja 2^24/1024 = 16384 = 2^14
parciales, la segunda 2^14/1024 = 16, la tercera 1 (con un solo bloque, y
los 1008 threads que sobran copian ceros). Son 3 lanzamientos, contra 24 de
reduccion_global, y a diferencia de reduccion_global4 y 5 el host no suma
nada. entrada y salida se intercambian después de cada pasada por la misma
razón que en reduccion_global: ningún lanzamiento escribe en el arreglo que
está leyendo. Al terminar, el resultado queda en entrada[0].

n_bloques = (N + n_threads - 1) / n_threads es la división entera hacia
arriba, techo(N / n_threads): sumar n_threads - 1 antes de dividir hace que
cualquier resto empuje al siguiente múltiplo, y si N ya es múltiplo no cambia
nada. Con N / n_threads a secas faltaría un bloque cuando N no es múltiplo:
con n_threads = 1024, N = 5000 daría 4 bloques (904 elementos sin thread) en
vez de 5. Los threads que sobran en el último bloque (5120 - 5000 = 120 en
ese ejemplo) copian 0.f a s_data, que no cambia la suma. Es la misma fórmula
de SAXPY en el capítulo 1, y va siempre con su protección (if (i < n) allá,
el 0.f aquí). -->

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

Ciclo original:

```c
for (int i = 0; i < N; i++){
  a[i] = b[i] + c[i];
}
```

Versión equivalente:
```c
for (int i = 0; i < N; i+=2){
  a[i]   = b[i]   + c[i];
  a[i+1] = b[i+1] + c[i+1];
}
```

Esto corresponde a *loop unrolling* con un factor de $2$.

<!-- NOTA — las dos versiones hacen exactamente el mismo trabajo útil; la segunda
hace la mitad de las comparaciones y saltos del ciclo. Ojo: la versión
desenrollada supone que N es par; en código real hay que tratar el resto. El factor de unrolling es cuántas
iteraciones del ciclo original se hacen en una pasada. -->

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

## **Optimización: loop unrolling**

- Estos ciclos tienen menos iteraciones que el original.
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

<svg viewBox="0 0 1044 546" width="880" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-unr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<text x="170" y="16" font-size="16" fill="#333">N = 16, n_threads = 4, n_bloques = 2: cada bloque suma 2 &#215; 4 = 8 elementos</text>
<path d="M170,56 L170,50 L592,50 L592,56" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="381" y="44" font-size="15" text-anchor="middle" fill="#333">Bloque 0 (idata = data + 0)</text>
<path d="M602,56 L602,50 L1024,50 L1024,56" stroke="#555" stroke-width="1.4" fill="none"/>
<text x="813" y="44" font-size="15" text-anchor="middle" fill="#333">Bloque 1 (idata = data + 8)</text>
<line x1="192" y1="102" x2="192" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="408" y1="102" x2="208.8" y2="134.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="192" y1="153" x2="192" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="246" y1="102" x2="246" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="462" y1="102" x2="262.8" y2="134.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="246" y1="153" x2="246" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="300" y1="102" x2="300" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="516" y1="102" x2="316.8" y2="134.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="300" y1="153" x2="300" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="354" y1="102" x2="354" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="570" y1="102" x2="370.8" y2="134.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="354" y1="153" x2="354" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="624" y1="102" x2="624" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="840" y1="102" x2="640.8" y2="134.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="624" y1="153" x2="624" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="678" y1="102" x2="678" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="894" y1="102" x2="694.8" y2="134.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="678" y1="153" x2="678" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="732" y1="102" x2="732" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="948" y1="102" x2="748.8" y2="134.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="732" y1="153" x2="732" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="786" y1="102" x2="786" y2="120" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="1002" y1="102" x2="802.8" y2="134.3" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="786" y1="153" x2="786" y2="171" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="192" y1="212" x2="192" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="300" y1="212" x2="208.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="192" y1="263" x2="192" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="246" y1="212" x2="246" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="354" y1="212" x2="262.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="246" y1="263" x2="246" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="624" y1="212" x2="624" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="732" y1="212" x2="640.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="624" y1="263" x2="624" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="678" y1="212" x2="678" y2="230" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="786" y1="212" x2="694.2" y2="241.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="678" y1="263" x2="678" y2="281" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="192" y1="322" x2="192" y2="340" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="246" y1="322" x2="206.3" y2="347.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="192" y1="373" x2="192" y2="391" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="624" y1="322" x2="624" y2="340" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="678" y1="322" x2="638.3" y2="347.8" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="624" y1="373" x2="624" y2="391" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="192" y1="432" x2="192" y2="450" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="192" y1="483" x2="192" y2="501" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="624" y1="432" x2="624" y2="450" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<line x1="624" y1="483" x2="246" y2="501" stroke="#444" stroke-width="1.4" marker-end="url(#fl-unr)"/>
<rect x="170" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="88" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="224" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="88" font-size="18" text-anchor="middle" fill="#222">1</text>
<rect x="278" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="88" font-size="18" text-anchor="middle" fill="#222">8</text>
<rect x="332" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="88" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="386" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="408" y="88" font-size="18" text-anchor="middle" fill="#222">7</text>
<rect x="440" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="462" y="88" font-size="18" text-anchor="middle" fill="#222">2</text>
<rect x="494" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="516" y="88" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="548" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="570" y="88" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="602" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="624" y="88" font-size="18" text-anchor="middle" fill="#222">2</text>
<rect x="656" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="678" y="88" font-size="18" text-anchor="middle" fill="#222">4</text>
<rect x="710" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="732" y="88" font-size="18" text-anchor="middle" fill="#222">6</text>
<rect x="764" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="786" y="88" font-size="18" text-anchor="middle" fill="#222">1</text>
<rect x="818" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="840" y="88" font-size="18" text-anchor="middle" fill="#222">8</text>
<rect x="872" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="894" y="88" font-size="18" text-anchor="middle" fill="#222">7</text>
<rect x="926" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="948" y="88" font-size="18" text-anchor="middle" fill="#222">9</text>
<rect x="980" y="62" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="1002" y="88" font-size="18" text-anchor="middle" fill="#222">5</text>
<rect x="170" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="198" font-size="18" text-anchor="middle" fill="#222">10</text>
<rect x="224" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="198" font-size="18" text-anchor="middle" fill="#222">3</text>
<rect x="278" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="300" y="198" font-size="18" text-anchor="middle" fill="#222">17</text>
<rect x="332" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="354" y="198" font-size="18" text-anchor="middle" fill="#222">7</text>
<rect x="386" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="602" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="624" y="198" font-size="18" text-anchor="middle" fill="#222">10</text>
<rect x="656" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="678" y="198" font-size="18" text-anchor="middle" fill="#222">11</text>
<rect x="710" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="732" y="198" font-size="18" text-anchor="middle" fill="#222">15</text>
<rect x="764" y="172" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="786" y="198" font-size="18" text-anchor="middle" fill="#222">6</text>
<rect x="818" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="872" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="926" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="980" y="172" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="308" font-size="18" text-anchor="middle" fill="#222">27</text>
<rect x="224" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="308" font-size="18" text-anchor="middle" fill="#222">10</text>
<rect x="278" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="602" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="624" y="308" font-size="18" text-anchor="middle" fill="#222">25</text>
<rect x="656" y="282" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="678" y="308" font-size="18" text-anchor="middle" fill="#222">17</text>
<rect x="710" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="764" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="818" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="872" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="926" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="980" y="282" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="170" y="392" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="418" font-size="18" text-anchor="middle" fill="#222">37</text>
<rect x="224" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="278" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="332" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="386" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="440" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="494" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="548" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="602" y="392" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="624" y="418" font-size="18" text-anchor="middle" fill="#222">42</text>
<rect x="656" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="710" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="764" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="818" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="872" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="926" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<rect x="980" y="392" width="44" height="40" rx="3" fill="#e9e1f0" stroke="#555"/>
<circle cx="192" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="143" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="246" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="246" y="143" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="300" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="300" y="143" font-size="16" text-anchor="middle" fill="#fff">2</text>
<circle cx="354" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="354" y="143" font-size="16" text-anchor="middle" fill="#fff">3</text>
<circle cx="624" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="624" y="143" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="678" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="678" y="143" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="732" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="732" y="143" font-size="16" text-anchor="middle" fill="#fff">2</text>
<circle cx="786" cy="137" r="16" fill="#e8641b" stroke="#444"/>
<text x="786" y="143" font-size="16" text-anchor="middle" fill="#fff">3</text>
<circle cx="192" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="253" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="246" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="246" y="253" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="624" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="624" y="253" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="678" cy="247" r="16" fill="#e8641b" stroke="#444"/>
<text x="678" y="253" font-size="16" text-anchor="middle" fill="#fff">1</text>
<circle cx="192" cy="357" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="363" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="624" cy="357" r="16" fill="#e8641b" stroke="#444"/>
<text x="624" y="363" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="192" cy="467" r="16" fill="#e8641b" stroke="#444"/>
<text x="192" y="473" font-size="16" text-anchor="middle" fill="#fff">0</text>
<circle cx="624" cy="467" r="16" fill="#e8641b" stroke="#444"/>
<text x="624" y="473" font-size="16" text-anchor="middle" fill="#fff">0</text>
<rect x="170" y="502" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="192" y="528" font-size="18" text-anchor="middle" fill="#222">37</text>
<rect x="224" y="502" width="44" height="40" rx="3" fill="#eef3df" stroke="#555"/>
<text x="246" y="528" font-size="18" text-anchor="middle" fill="#222">42</text>
<text x="284" y="528" font-size="16" fill="#333">el <tspan font-style="italic">host</tspan> suma 37 + 42 = 79</text>
<text x="156" y="528" font-size="16" text-anchor="end" fill="#333">parcial</text>
<text x="156" y="88" font-size="16" text-anchor="end" fill="#333">Memoria global</text>
<text x="156" y="143" font-size="16" text-anchor="end" fill="#333">ID del thread</text>
<text x="156" y="161" font-size="13" text-anchor="end" fill="#666" font-style="italic">unrolling</text>
<text x="156" y="271" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 2</text>
<text x="156" y="381" font-size="13" text-anchor="end" fill="#666" font-style="italic">stride = 1</text>
</svg>

<!-- NOTA — reduccion_global6.cu: cada bloque procesa 2*blockDim elementos. Primero
cada thread suma su elemento con el que está un bloque más adelante, y después
se hace el árbol de siempre sobre blockDim elementos. Se lanza con
n_bloques/2. -->

<!-- NOTA — la figura (redibujada en SVG a partir de la del libro). Las sumas
están verificadas. Unrolling (data[idx] += data[idx +
blockDim.x]): bloque 0 → 3+7=10, 1+2=3, 8+9=17, 4+3=7; bloque 1 → 2+8=10,
4+7=11, 6+9=15, 1+5=6. Stride 2 → 27, 10 y 25, 17; stride 1 → 37 y 42. Los
círculos son threadIdx.x (0 a 3 en cada bloque); las casillas lila, datos que
ya no se usan. La última fila es el arreglo APARTE parcial, como en el
código actual; la figura original del libro escribía 42 en data[1], que es
la carrera corregida en reduccion_global4–8 (el bloque 1 podría
escribir ahí mientras el bloque 0 todavía lee data[1]). El host suma los
parciales: 37 + 42 = 79, la suma de los 16 datos. -->

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

# Warp primitives

---

## **Warp primitives**

- Las *warp primitives* (funciones primitivas de *warp*) son funciones básicas para trabajar directamente con los *threads* dentro de un *warp*.
- Un poco de jerga: un *thread* dentro de un *warp* se conoce como un *lane*.
- Hay muchas funciones disponibles: *CUDA Programming Guide*.
- Las usaremos para el último paso de la reducción, dentro de un *warp*, sin memoria compartida.

<!-- NOTA — lane es el índice del thread dentro de su warp, de 0 a 31 (threadIdx.x
% 32). Todas las primitivas modernas terminan en _sync y reciben una máscara:
son las versiones seguras desde Volta, porque sincronizan a los threads de la
máscara antes de intercambiar datos. Son la forma correcta de trabajar
dentro de un warp desde Volta: el idioma del libro ("warp unrolling", seis
sumas seguidas en memoria compartida confiando en el lock-step, sin
__syncwarp()) es una race condition en la T4. reduccion_global8.cu, en la
página de códigos, lo muestra corregido con volatile y dos __syncwarp() por
paso; no se ve en clase. -->

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

<!-- NOTA — los ejemplos de up y xor muestran las otras dos formas de mover valores:
warp_shuffle_up.cu calcula la suma acumulada (el lane t termina con la suma
de los lanes 0 a t), y warp_shuffle_xor.cu hace la reducción "en mariposa",
donde al final TODOS los lanes tienen la suma (no solo el 0). Otra opción para este nivel de control son los
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

<!-- NOTA — sobre el paso 1: el material original decía que el grid-stride "mejora el
occupancy". No es así: el occupancy depende de los recursos por bloque, no de
cuántos bloques se lanzan. Lo que sí hace es desacoplar el tamaño del grid del
tamaño de los datos (basta con llenar el GPU: en la T4 caben 4 bloques de
256 threads por SM; si se lanzan más, corren en tandas),
reducir el costo de lanzar y retirar bloques, y amortizar el trabajo fijo de
cada thread (calcular índices, inicializar) sobre muchos elementos. -->

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

<!-- --- -->
<!---->
<!-- # Ejercicios -->
<!---->
<!-- --- -->
<!---->
<!-- ## **Ejercicio 1: la reducción completa** -->
<!---->
<!-- Descargar: [reduccion-shuffle.cu](../code/threads/reduccion-shuffle.cu) y [reduccion_global8.cu](../code/threads/reduccion_global8.cu) -->
<!---->
<!-- 1. Ejecutar `reduccion-shuffle.x`. ¿Qué ancho de banda logra? Compararlo con los $300$ GB/s de la T4. -->
<!-- 2. Medir los dos programas con `nvprof` y comparar el tiempo de sus *kernels*. -->
<!-- 3. ¿Por qué el resultado del GPU puede cambiar en los últimos dígitos entre una ejecución y otra? -->
<!---->
<!-- ```bash -->
<!-- nvcc -arch=sm_75 reduccion-shuffle.cu -o reduccion-shuffle.x -->
<!-- nvprof ./reduccion-shuffle.x -->
<!-- ``` -->
<!---->
<!-- <!-- RESPUESTAS — TODO: medir en la T4. (1) Lo esperable es que se acerque al -->
<!-- peak de 300 GB/s: la reducción solo lee cada elemento una vez y hace una suma -->
<!-- por cada 4 bytes, así que es puramente memory bound (roofline del capítulo 1). -->
<!-- (2) reduccion_global8 lee y escribe la memoria global varias veces dentro de -->
<!-- cada bloque y copia los parciales al host; reduccion-shuffle lee cada -->
<!-- elemento una sola vez. (3) atomicAdd sobre float suma los aportes de los -->
<!-- bloques en el orden en que terminan, que cambia entre ejecuciones; la suma en -->
<!-- punto flotante no es asociativa, así que el redondeo cambia. Conecta con la -->
<!-- nota del ciclo del host de reduccion_global (la suma en árbol contra la -->
<!-- secuencial). --> -->
<!---->
<!-- --- -->
<!---->
<!-- ## **Ejercicio 2: histograma y contención** -->
<!---->
<!-- Descargar: [histograma.cu](../code/threads/histograma.cu) -->
<!---->
<!-- 1. Ejecutar el programa. Con datos **aleatorios**, ¿qué *kernel* es más rápido? -->
<!-- 2. Con datos **constantes**, ¿qué le pasa a `histGlobal`? ¿Por qué? -->
<!-- 3. ¿Por qué la versión con memoria compartida sufre menos con datos constantes? -->
<!---->
<!-- <!-- RESPUESTAS — TODO: medir en la T4. (1) histCompartida debería ganar: las -->
<!-- atómicas en memoria compartida son mucho más baratas que en global, y al final -->
<!-- cada bloque hace solo 256 atómicas globales. (2) Con datos constantes los 16 M -->
<!-- atomicAdd globales caen sobre el MISMO contador: el hardware los serializa y -->
<!-- el kernel se vuelve mucho más lento (contención máxima). (3) En la versión -->
<!-- compartida la contención queda dentro de cada bloque (solo compiten sus 256 -->
<!-- threads, en la memoria rápida del SM), y entre bloques hay una sola atómica -->
<!-- global por contador. Es la misma idea de la reducción: primero reducir -->
<!-- localmente, después una atómica por bloque. --> -->
<!---->
---

# Fin Capítulo 3

## Próximo capítulo: invocación de los kernels
