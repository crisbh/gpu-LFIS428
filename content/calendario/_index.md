---
title: "Calendario"
description: "Calendario de clases del curso"
---

# Calendario del curso

Sesiones de **90 minutos**, con teoría y una parte práctica (*hands-on* al menos cada dos sesiones, marcadas con 🛠️). Las **clases 1–18** cubren los contenidos; las **clases 19–30** se reservan para el desarrollo y la presentación del **proyecto final**.

Las fechas son referenciales (ajústalas desde la planilla del curso). El material apunta al tema; cuando una clase empieza a mitad de un tema, el enlace lleva directo a la diapositiva de inicio (p. ej. `memoria.html#24`).

## Contenidos (clases 1–18)

| Clase | Fecha | Tema | Material | Evaluación |
|:---:|:---|:---|:---|:---|
| 1 | — | Introducción: reglas, paralelismo GPU vs CPU, Colab (T4), primer kernel (hola mundo) | [Introducción](../slides/introduccion.html) | |
| 2 | — | 🛠️ Suma de vectores host vs. device; manejo de memoria (cudaMalloc, cudaMemcpy) | [Introducción](../slides/introduccion.html#25) | |
| 3 | — | SAXPY; reglas de kernels; organización de threads (blocks, grid); warps | [Introducción](../slides/introduccion.html#29) | |
| 4 | — | 🛠️ Ejercicio (bug de los límites → *grid-stride*); manejo de errores | [Introducción](../slides/introduccion.html#43) | |
| 5 | — | Profilers (Nsight); análisis de optimización; modelo roofline (con demo) | [Introducción](../slides/introduccion.html#50) | |
| 6 | — | 🛠️ Taller de ejercicios | | |
| 7 | — | **Quiz 1** (contenidos de las clases 1–6) | | Quiz 1 |
| 8 | — | Receso Fiestas Patrias | | |
| 9 | — | Receso Fiestas Patrias | | |
| 10 | — | Memoria: jerarquía y accesos; transpuesta de matrices; AoS vs. SoA | [Memoria](../slides/memoria.html) | |
| 11 | — | 🛠️ Memoria compartida y *padding*; memoria constante | [Memoria](../slides/memoria.html#24) | |
| 12 | — | 🛠️ Memoria unificada y *pinned* | [Memoria](../slides/memoria.html#52) | |
| 13 | — | Occupancy; reducción paralela; divergencia de *warps*; *loop unrolling* | [Threads](../slides/threads.html#16) | |
| 14 | — | *Grid-stride loops*; *warp primitives*; operaciones atómicas | [Threads](../slides/threads.html#59) | Tarea 1 |
| 15 | — | Invocación de kernels: *streams*, ejecución asíncrona, eventos | [Kernels](../slides/kernels.html#5) | |
| 16 | — | 🛠️ Librerías de CUDA y Python (cuBLAS/cuFFT/cuRAND, PyCUDA/Numba/CuPy) | [Librerías](../slides/librerias-python.html) | |
| 17 | — | 🛠️ Aplicaciones: N-cuerpos y OpenGL | [Aplicaciones](../slides/aplicaciones.html#3) | |
| 18 | — | Aplicaciones: *ray tracing* | [Aplicaciones](../slides/aplicaciones.html#37) | Tarea 2 |

> **Nota de ritmo:** la secuencia de optimización de la reducción (clase 13) recorre 8 variantes (`reduccion_global` … `reduccion_global8`). En clase se cubren **3–4 representativas** (p. ej. memoria global → sin divergencia → *unrolling* ×2 → *warp unrolling*); el resto queda como **referencia / autoestudio** en el deck de [Threads](../slides/threads.html#27). El deck de [Kernels](../slides/kernels.html#15) también incluye *dynamic parallelism*, CUDA/OpenMP, MPS y *kernel overhead* como **referencia / autoestudio** (más allá de *streams* y eventos de la clase 15).

## Proyecto final (clases 19–30)

| Clase | Fecha | Tema | Evaluación |
|:---:|:---|:---|:---|
| 19 | — | Introducción al proyecto final: temas, formación de grupos | |
| 20 | — | Definición de propuestas y lineamientos | |
| 21 | — | Desarrollo del proyecto (asesoría) | |
| 22 | — | Desarrollo del proyecto (asesoría) | |
| 23 | — | Desarrollo del proyecto (asesoría) | |
| 24 | — | **Presentaciones de avance** | Avance del proyecto |
| 25 | — | Desarrollo del proyecto (asesoría) | |
| 26 | — | Desarrollo del proyecto (asesoría) | |
| 27 | — | Desarrollo del proyecto (asesoría) | |
| 28 | — | Desarrollo del proyecto (asesoría) | |
| 29 | — | **Presentaciones finales** (parte 1) | Proyecto final |
| 30 | — | **Presentaciones finales** (parte 2) y cierre | Proyecto final |

<!--
Mantenimiento:
- Rellena "Fecha" desde la planilla del curso y ajusta los #N de inicio de cada clase.
- Los nombres/pesos/ubicación de las evaluaciones (Quiz 1, Tarea 1, Tarea 2, proyecto)
  deben reconciliarse con el esquema de evaluación en content/_index.md (fuente de verdad);
  no inventar pesos aquí.
- Todos los módulos están migrados (introduccion, kernels, memoria, threads, librerias-python,
  aplicaciones). Los #N asumen los decks actuales; re-derívalos si se reordenan diapositivas.
- Las clases 19–30 son un colchón generoso: si los contenidos se extienden, se pueden
  adelantar 2–3 sesiones de proyecto como prácticas extra.
-->
