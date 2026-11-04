---
title: "Códigos"
description: "Códigos de ejemplo del curso"
---

# Códigos del curso

Programas de ejemplo para las clases:

## 1. Introducción a CUDA

- [hola_mundo.cu](../code/intro/hola_mundo.cu)
- [suma_vectores_host.c](../code/intro/suma_vectores_host.c)
- [suma_vectores_gpu.cu](../code/intro/suma_vectores_gpu.cu)
- [saxpy.cu](../code/intro/saxpy.cu)
- [mostrarIndices.cu](../code/intro/mostrarIndices.cu)
- [suma_vectores_limites.cu](../code/intro/ejercicios/suma_vectores_limites.cu) (ejercicio)
- [matmul_roofline.cu](../code/intro/ejercicios/matmul_roofline.cu) (ejercicio)
- [matvec_roofline.cu](../code/intro/ejercicios/matvec_roofline.cu) (ejercicio)
- [errores.cu](../code/intro/errores.cu)
- [simpleDeviceQuery.cu](../code/intro/simpleDeviceQuery.cu)

<!-- ###################################### -->
## 2. El uso de la memoria del GPU

- [common.h](../code/memoria/common.h)
- [variableGlobal.cu](../code/memoria/variableGlobal.cu) · [variableGlobalDin.cu](../code/memoria/variableGlobalDin.cu)
- [copiarFila.cu](../code/memoria/copiarFila.cu) · [copiarColumna.cu](../code/memoria/copiarColumna.cu)
- [transpuesta.cu](../code/memoria/transpuesta.cu) · [transpuesta_compartida.cu](../code/memoria/transpuesta_compartida.cu)
- [memoria_constante.cu](../code/memoria/memoria_constante.cu) (necesita [common.h](../code/memoria/common.h)) · [memoriaPinned.cu](../code/memoria/memoriaPinned.cu) (necesita [common.h](../code/memoria/common.h)) · [memoria_unificada.cu](../code/memoria/memoria_unificada.cu) (necesita [common.h](../code/memoria/common.h))

<!-- ###################################### -->
## 3. Control de los threads

- [cuda_thread_block.cu](../code/threads/cuda_thread_block.cu) · [matriz_mult.cu](../code/threads/matriz_mult.cu)
- Reducción (memoria global): [reduccion_global.cu](../code/threads/reduccion_global.cu) · [reduccion_global2.cu](../code/threads/reduccion_global2.cu) · [reduccion_global3.cu](../code/threads/reduccion_global3.cu) · [reduccion_global4.cu](../code/threads/reduccion_global4.cu) · [reduccion_global5.cu](../code/threads/reduccion_global5.cu) · [reduccion_global6.cu](../code/threads/reduccion_global6.cu) · [reduccion_global7.cu](../code/threads/reduccion_global7.cu) · [reduccion_global8.cu](../code/threads/reduccion_global8.cu)
- Reducción (memoria compartida): [reduccion_compartida.cu](../code/threads/reduccion_compartida.cu)
- Warp primitives: [warp_shuffle_down.cu](../code/threads/warp_shuffle_down.cu) · [warp_shuffle_up.cu](../code/threads/warp_shuffle_up.cu) · [warp_shuffle_xor.cu](../code/threads/warp_shuffle_xor.cu)
- Reducción completa (con shuffle): [reduccion-shuffle.cu](../code/threads/reduccion-shuffle.cu) 
- Ejemplo: [histograma.cu](../code/threads/histograma.cu)
<!-- - [grid_stride.cu](../code/threads/grid_stride.cu) -->
<!-- - Errores (PyCUDA): [gpu_suma_error.py](../code/threads/gpu_suma_error.py) · [gpu_producto_punto_error.py](../code/threads/gpu_producto_punto_error.py) -->

<!-- ###################################### -->
## 4. Invocación de los kernels

- Eventos: [cuda_event.cu](../code/kernels/cuda_event.cu)
- *Default stream* y concurrencia: [streams-concurrencia.cu](../code/kernels/streams-concurrencia.cu) (ejercicio 1)
- *Pipelining*: [cuda_pipelining.cu](../code/kernels/cuda_pipelining.cu) (ejercicio 2)

Apéndice (para estudiar por cuenta propia; ver en las diapositivas cómo compilarlos):

- *Callbacks*: [cuda_callback.cu](../code/kernels/cuda_callback.cu)
- Prioridad de los *streams*: [prioritized_cuda_stream.cu](../code/kernels/prioritized_cuda_stream.cu)
- Eventos en varios *streams*: [cuda_event_with_streams.cu](../code/kernels/cuda_event_with_streams.cu)
- Paralelismo dinámico: [dynamic_parallelism.cu](../code/kernels/dynamic_parallelism.cu) · [recursion.cu](../code/kernels/recursion.cu)
- CUDA y OpenMP: [openmp.cu](../code/kernels/openmp.cu) · [openmp_default_stream.cu](../code/kernels/openmp_default_stream.cu)
- MPS: [simpleMPI.cu](../code/kernels/simpleMPI.cu)
- Costo de lanzar un *kernel*: [cuda_kernel.cu](../code/kernels/cuda_kernel.cu)

<!-- ###################################### -->
<!-- ## Aplicaciones -->
<!---->
- N-cuerpos: [nbody-colab.cu](../code/aplicaciones/nbody-colab.cu) (Colab) · [nbody-cpu.c](../code/aplicaciones/nbody-cpu.c) (CPU, para comparar)
- Estructuras de datos: [aos.cu](../code/memoria/aos.cu) · [soa.cu](../code/memoria/soa.cu) · [alineamiento_datos.c](../code/memoria/alineamiento_datos.c)
<!-- - Estructuras de datos: [aos.cu](../code/memoria/aos.cu) · [soa.cu](../code/memoria/soa.cu) · [alineamiento_datos.c](../code/memoria/alineamiento_datos.c) -->

<!-- ###################################### -->
<!-- ## Librerías de CUDA y Python -->
<!---->
<!-- - cuBLAS: [cublasSgemm.cpp](../code/librerias-python/cublasSgemm.cpp) -->
<!-- - cuRAND: [curand_host.cpp](../code/librerias-python/curand_host.cpp) · [curand_device.cu](../code/librerias-python/curand_device.cu) -->
<!-- - cuFFT: [cufft.1d.cpp](../code/librerias-python/cufft.1d.cpp) -->
<!-- - Numba: [numba_saxpy.py](../code/librerias-python/numba_saxpy.py) · [numba_matmul.py](../code/librerias-python/numba_matmul.py) -->
<!-- - CuPy: [cupy_op.py](../code/librerias-python/cupy_op.py) -->
<!-- - PyCUDA: [pycuda_matmul.py](../code/librerias-python/pycuda_matmul.py) -->

