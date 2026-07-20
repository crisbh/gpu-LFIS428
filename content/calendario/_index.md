---
title: "Calendario"
description: "Calendario de clases del curso"
---

# Calendario del curso

Sesiones de **90 minutos**, con teoría y una parte práctica (*hands-on* al menos cada dos sesiones, marcadas con 🛠️). Las **clases 1–16** cubren los contenidos; las **clases 17–28** se reservan para el desarrollo y la presentación del **proyecto final**.

Las fechas son referenciales (ajústalas desde la planilla del curso). El material apunta al tema; cuando una clase empieza a mitad de un tema, el enlace lleva directo a la diapositiva de inicio (p. ej. `memoria.html#21`).

## Contenidos (clases 1–16)

| Clase | Fecha | Tema | Material | Evaluación |
|:---:|:---|:---|:---|:---|
| 1 | — | Introducción: arquitectura GPU, modelo heterogéneo, NVCC | [Introducción](../slides/introduccion.html) | |
| 2 | — | 🛠️ Primer kernel, organización de threads e índices | [Introducción](../slides/introduccion.html#14) | |
| 3 | — | 🛠️ Manejo de errores y *profiling* (nvprof/ncu/nsys) | [Introducción](../slides/introduccion.html#32) | |
| 4 | — | Jerarquía de memoria; memoria global y acceso *coalesced* | [Memoria](../slides/memoria.html) | |
| 5 | — | 🛠️ Transpuesta de matrices; AoS vs. SoA | [Memoria](../slides/memoria.html#14) | |
| 6 | — | Memoria compartida y conflictos de bancos | [Memoria](../slides/memoria.html#21) | |
| 7 | — | 🛠️ Memoria constante, unificada y *pinned* | [Memoria](../slides/memoria.html#46) | |
| 8 | — | **Prueba** (contenidos de las clases 1–7) | | Prueba (25%) |
| 9 | — | SIMT y *occupancy* | [Threads](../slides/threads.html) | |
| 10 | — | 🛠️ Reducción paralela y divergencia de *warps* | [Threads](../slides/threads.html#24) | |
| 11 | — | 🛠️ *Loop/warp unrolling*, *grid-stride loops* | [Threads](../slides/threads.html#40) | |
| 12 | — | *Warp primitives*; repaso de optimización | [Threads](../slides/threads.html#63) | Tarea 1 (25%) |
| 13 | — | Invocación de kernels: *streams*, eventos, paralelismo dinámico | [Kernels](../slides/kernels.html) | |
| 14 | — | 🛠️ Librerías de CUDA y Python (cuBLAS/cuFFT/cuRAND, PyCUDA/Numba/CuPy) | [Librerías](../slides/librerias-python.html) | |
| 15 | — | 🛠️ Aplicaciones: N-cuerpos y OpenGL | [Aplicaciones](../slides/aplicaciones.html#3) | |
| 16 | — | Aplicaciones: *ray tracing* | [Aplicaciones](../slides/aplicaciones.html#37) | Tarea 2 (25%) |

> **Nota de ritmo:** la secuencia de optimización de la reducción (clases 10–11) recorre 8 variantes (`reduccion_global` … `reduccion_global8`). En clase se cubren **3–4 representativas** (p. ej. memoria global → sin divergencia → *unrolling* ×2 → *warp unrolling*); el resto queda como **referencia / autoestudio** en el deck de [Threads](../slides/threads.html#40).

## Proyecto final (clases 17–28)

| Clase | Fecha | Tema | Evaluación |
|:---:|:---|:---|:---|
| 17 | — | Introducción al proyecto final: temas (p. ej. *deep learning*), formación de grupos | |
| 18 | — | Definición de propuestas y lineamientos | |
| 19 | — | Desarrollo del proyecto (asesoría) | |
| 20 | — | Desarrollo del proyecto (asesoría) | |
| 21 | — | Desarrollo del proyecto (asesoría) | |
| 22 | — | **Presentaciones de avance** | Avance del proyecto |
| 23 | — | Desarrollo del proyecto (asesoría) | |
| 24 | — | Desarrollo del proyecto (asesoría) | |
| 25 | — | Desarrollo del proyecto (asesoría) | |
| 26 | — | Desarrollo del proyecto (asesoría) | |
| 27 | — | **Presentaciones finales** (parte 1) | Proyecto final (25%) |
| 28 | — | **Presentaciones finales** (parte 2) y cierre | Proyecto final (25%) |

<!--
Mantenimiento:
- Rellena "Fecha" desde la planilla del curso y ajusta los #N de inicio de cada clase.
- Las clases 13 y 14 (kernels, librerías de Python) aún no tienen material migrado
  (ver content/slides/_index.md y CLAUDE.md). Reemplaza "(próximamente)" al migrarlas.
- Las clases 17–28 son un colchón generoso: si los contenidos se extienden, se pueden
  adelantar 2–3 sesiones de proyecto como prácticas extra.
-->
