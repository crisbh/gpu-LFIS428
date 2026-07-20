---
title: "Códigos"
description: "Códigos de ejemplo del curso"
---

# Códigos del curso

Programas de ejemplo, agrupados por tema. Cada archivo se puede ver/descargar directamente.

## Introducción a CUDA

- [hola_mundo.cu](../code/intro/hola_mundo.cu)
- [suma_vectores_host.c](../code/intro/suma_vectores_host.c)
- [suma_vectores_gpu.cu](../code/intro/suma_vectores_gpu.cu)
- [mostrarIndices.cu](../code/intro/mostrarIndices.cu)
- [simpleDeviceQuery.cu](../code/intro/simpleDeviceQuery.cu)

## El uso de la memoria del GPU

- [variableGlobal.cu](../code/memoria/variableGlobal.cu) · [variableGlobalDin.cu](../code/memoria/variableGlobalDin.cu)
- [copiarFila.cu](../code/memoria/copiarFila.cu) · [copiarColumna.cu](../code/memoria/copiarColumna.cu)
- [transpuesta.cu](../code/memoria/transpuesta.cu) · [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu)
- [aos.cu](../code/memoria/aos.cu) · [soa.cu](../code/memoria/soa.cu) · [alineamiento_datos.c](../code/memoria/alineamiento_datos.c)
- [memoria_constante.cu](../code/memoria/memoria_constante.cu) · [memoriaPinned.cu](../code/memoria/memoriaPinned.cu) · [memoria_unificada.cu](../code/memoria/memoria_unificada.cu)
- [common.h](../code/memoria/common.h)

## Control de los threads

- [cuda_thread_block.cu](../code/threads/cuda_thread_block.cu) · [matriz_mult.cu](../code/threads/matriz_mult.cu)
- Reducción (memoria global): [reduccion_global.cu](../code/threads/reduccion_global.cu) · [reduccion_global2.cu](../code/threads/reduccion_global2.cu) · [reduccion_global3.cu](../code/threads/reduccion_global3.cu) · [reduccion_global4.cu](../code/threads/reduccion_global4.cu) · [reduccion_global5.cu](../code/threads/reduccion_global5.cu) · [reduccion_global6.cu](../code/threads/reduccion_global6.cu) · [reduccion_global7.cu](../code/threads/reduccion_global7.cu) · [reduccion_global8.cu](../code/threads/reduccion_global8.cu)
- Reducción (memoria compartida): [reduccion_compartida.cu](../code/threads/reduccion_compartida.cu) · [reduccion_compartida_new.cu](../code/threads/reduccion_compartida_new.cu) · [reduction.h](../code/threads/reduction.h)
- [grid_stride.cu](../code/threads/grid_stride.cu)
- Warp primitives: [warp_shuffle_down.cu](../code/threads/warp_shuffle_down.cu) · [warp_shuffle_up.cu](../code/threads/warp_shuffle_up.cu) · [warp_shuffle_xor.cu](../code/threads/warp_shuffle_xor.cu)
- Errores (PyCUDA): [gpu_suma_error.py](../code/threads/gpu_suma_error.py) · [gpu_producto_punto_error.py](../code/threads/gpu_producto_punto_error.py)

## Invocación de los kernels

- Streams: [cuda_default_stream.cu](../code/kernels/cuda_default_stream.cu) · [cuda_multi_stream.cu](../code/kernels/cuda_multi_stream.cu) · [cuda_multi_stream_with_sync.cu](../code/kernels/cuda_multi_stream_with_sync.cu) · [cuda_multi_stream_with_default.cu](../code/kernels/cuda_multi_stream_with_default.cu) · [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu) · [prioritized_cuda_stream.cu](../code/kernels/prioritized_cuda_stream.cu)
- Callbacks y eventos: [cuda_callback.cu](../code/kernels/cuda_callback.cu) · [cuda_event.cu](../code/kernels/cuda_event.cu) · [cuda_event_with_streams.cu](../code/kernels/cuda_event_with_streams.cu)
- Paralelismo dinámico: [dynamic_parallelism.cu](../code/kernels/dynamic_parallelism.cu) · [recursion.cu](../code/kernels/recursion.cu)
- OpenMP / MPI: [openmp.cu](../code/kernels/openmp.cu) · [openmp_default_stream.cu](../code/kernels/openmp_default_stream.cu) · [simpleMPI.cu](../code/kernels/simpleMPI.cu)
- Overhead de ejecución: [cuda_kernel.cu](../code/kernels/cuda_kernel.cu)

## Librerías de CUDA y Python

- cuBLAS: [cublasSgemm.cpp](../code/librerias-python/cublasSgemm.cpp)
- cuRAND: [curand_host.cpp](../code/librerias-python/curand_host.cpp) · [curand_device.cu](../code/librerias-python/curand_device.cu)
- cuFFT: [cufft.1d.cpp](../code/librerias-python/cufft.1d.cpp)
- Numba: [numba_saxpy.py](../code/librerias-python/numba_saxpy.py) · [numba_matmul.py](../code/librerias-python/numba_matmul.py)
- CuPy: [cupy_op.py](../code/librerias-python/cupy_op.py)
- PyCUDA: [pycuda_matmul.py](../code/librerias-python/pycuda_matmul.py)

## Aplicaciones

- N-cuerpos: [nbody.cu](../code/aplicaciones/nbody.cu) · [nbody.h](../code/aplicaciones/nbody.h) · [tipsy.h](../code/aplicaciones/tipsy.h)
- OpenGL: [simpleGL.cu](../code/aplicaciones/simpleGL.cu)
- Ray tracing (*Ray Tracing in One Weekend* en CUDA), por capítulo:
  - [Cap. 1 — salida básica](../code/aplicaciones/ray_tracing/c01_salida_basica/ch01_rt.cu)
  - [Cap. 2 — vectores](../code/aplicaciones/ray_tracing/c02_vectores/ch02_rt.cu)
  - [Cap. 3 — rayos](../code/aplicaciones/ray_tracing/c03_rayos/ch03_rt.cu)
  - [Cap. 4 — esferas](../code/aplicaciones/ray_tracing/c04_esferas/ch04_rt.cu)
  - [Cap. 5 — normales](../code/aplicaciones/ray_tracing/c05_normales/ch05_rt.cu)
  - [Cap. 6 — antialiasing](../code/aplicaciones/ray_tracing/c06_antialiasing/ch06_rt.cu)
  - [Cap. 7 — difuso](../code/aplicaciones/ray_tracing/c07_difuso/ch07_rt.cu)
  - [Cap. 8 — metal](../code/aplicaciones/ray_tracing/c08_metal/ch08_rt.cu)
  - [Cap. 9 — dieléctricos](../code/aplicaciones/ray_tracing/c09_dielectricos/ch09_rt.cu)
  - [Cap. 10 — cámara](../code/aplicaciones/ray_tracing/c10_camara/ch10_rt.cu)
  - [Cap. 11 — borroso](../code/aplicaciones/ray_tracing/c11_borroso/ch11_rt.cu)
  - [Cap. 12 — próximos pasos](../code/aplicaciones/ray_tracing/c12_proximos_pasos/ch12_rt.cu)

  Cada capítulo de ray tracing incluye también sus *headers* (`vec3.h`, `ray.h`, `sphere.h`, etc.) en la misma carpeta.
