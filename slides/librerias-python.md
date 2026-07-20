---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

<!-- Contenido reconstruido del PDF original (libs_python.pdf). Solo texto: los
     diagramas/capturas del original se omitieron. -->

# **Programación en GPUs**
## Librerías de CUDA y Python

cuBLAS · cuRAND · cuFFT · cuDNN · Numba · CuPy · PyCUDA

---

## **Códigos**

Los códigos de esta clase están disponibles para descargar:

- cuBLAS: [cublasSgemm.cpp](../code/librerias-python/cublasSgemm.cpp)
- cuRAND: [curand_host.cpp](../code/librerias-python/curand_host.cpp) · [curand_device.cu](../code/librerias-python/curand_device.cu)
- cuFFT: [cufft.1d.cpp](../code/librerias-python/cufft.1d.cpp)
- Numba: [numba_saxpy.py](../code/librerias-python/numba_saxpy.py) · [numba_matmul.py](../code/librerias-python/numba_matmul.py)
- CuPy: [cupy_op.py](../code/librerias-python/cupy_op.py)
- PyCUDA: [pycuda_matmul.py](../code/librerias-python/pycuda_matmul.py)

---

# cuBLAS

---

## **cuBLAS**

- **BLAS**: *Basic Linear Algebra Subroutines*.
- Permite usar operaciones de álgebra lineal optimizadas con el GPU (uno o varios):
  - Nivel 1: vector-vector.
  - Nivel 2: matriz-vector.
  - Nivel 3: matriz-matriz.

---

## **cuBLAS**

- Ejemplo: *Single-Precision General Matrix Multiplication* (**SGEMM**), nivel 3.
- cuBLAS es parte del *toolkit* y se usa como una librería externa de C/C++ (no requiere programar en CUDA).

---

## **cuBLAS**

Implementa la operación:

$$C = \alpha \, \mathrm{op}(A)\,\mathrm{op}(B) + \beta\, C$$

---

## **cuBLAS** — parámetros de `cublasSgemm`

- `transa`, `transb` — si se usan las transpuestas de $A$ y/o $B$.
- `m`, `n`, `k` — número de elementos en cada dimensión de las matrices.
- `alpha`, `beta` — parámetros de la operación SGEMM.
- `*A`, `*B`, `*C` — *linear buffer* para los datos de la matriz (*column-major order*).
- `lda` — *leading column dimension* de $A$ (cuBLAS alinea los elementos con este valor).
- `ldb` — *leading column dimension* de $B$.

---

## **cuBLAS**

- `cublasSetMatrix()` / `cublasGetMatrix()` — *wrappers* de `cudaMemcpy()`.
- Ejemplo: [cublasSgemm.cpp](../code/librerias-python/cublasSgemm.cpp) — las matrices están transpuestas.

---

## **cuBLAS**

- **cuBLAS-XT**: para varios GPUs.
- cuBLAS permite computación de **precisión mixta**.
- Si hay *tensor cores* en el GPU (Volta+), cuBLAS (v. 11.0) los usará para los cálculos.
  - Para versiones anteriores es posible especificar el uso de los *tensor cores*.

---

# cuRAND

---

## **cuRAND**

- Para la creación de números aleatorios de forma paralela.
- Hay varias funciones para generarlos:
  - `curandGenerateUniform()`, `curandGenerateNormal()`, `curandGenerateLogNormal()`, `curandGeneratePoisson()`
  - variantes de doble precisión: `curandGenerateUniformDouble()`, `curandGenerateNormalDouble()`, `curandGenerateLogNormalDouble()`

---

## **cuRAND**

Ejemplos:
- [curand_host.cpp](../code/librerias-python/curand_host.cpp) — números aleatorios en el *host*.
- [curand_device.cu](../code/librerias-python/curand_device.cu) — números aleatorios en el *device* (en los *kernels*).

---

# cuFFT

---

## **cuFFT**

- **FFT**: *Fast Fourier Transform* — algoritmo para calcular numéricamente la transformada de Fourier (discreta).
- Las funciones de cuFFT corresponden a las de **FFTW**, una librería estándar.

---

## **cuFFT**

Hay que crear un *"plan"*: la definición de toda la información necesaria para el problema.
- `cufftPlan1D()`, `cufftPlan2D()`, `cufftPlan3D()`
- `cufftPlanMany()` → varias transformadas en una llamada.
- Para datos de más de 4 GB se añade `64` al final del nombre de la función.
- También se puede usar multi-GPU.

---

## **cuFFT**

La operación del FFT se realiza con:
- `cufftExecC2C()` → complejo a complejo
- `cufftExecR2C()` → real a complejo
- `cufftExecC2R()` → complejo a real

$C \to Z$, $R \to D$ para precisión doble.

---

## **cuFFT**

- La operación puede ser *forward* (FFT) o *inverse* (IFFT); en el programa tenemos un par de operaciones.
- Para las transformadas C2R y R2C hay que crear **2 planes** (uno por dirección).
- Para C2C solo se requiere **un plan**.
- Ejemplo: [cufft.1d.cpp](../code/librerias-python/cufft.1d.cpp)

---

# cuDNN

---

## **cuDNN**

- Librería para *deep neural nets*.
- Integrada en:
  - PyTorch
  - TensorFlow
  - Keras

---

# Numba

---

## **Numba**

- Módulo de Python para compilación de código:
  - compilador *Just-in-Time* (JIT);
  - compatible con NumPy;
  - se puede compilar para el GPU (usa CUDA, pero no es necesario programar en CUDA).

---

## **Numba** — `@vectorize`

```python
from numba import vectorize

@vectorize(["float32(float32, float32, float32)"], target='cuda')
def saxpy(scala, a, b):
    return scala * a + b
```

- `@vectorize` es un *decorator*: especifica los tipos de los parámetros y el valor de retorno.
- Hay 3 *targets*: `cuda` → GPU, `parallel` → CPU multi-core, `cpu` → un *thread*.

---

## **Numba** — `@cuda.jit`

```python
from numba import cuda

@cuda.jit
def matmul(d_c, d_a, d_b):
    x, y = cuda.grid(2)
    if (x < d_c.shape[0] and y < d_c.shape[1]):
        sum = 0
        for k in range(d_a.shape[1]):
            sum += d_a[x, k] * d_b[k, y]
        d_c[x, y] = sum
```

- `cuda.grid()` da los índices de los *threads* a nivel del *grid*.
- Se llama con: `matmul[dimGrid, dimBlock](d_c, d_a, d_b)`.

---

## **Numba**

Ejemplos:
- [numba_saxpy.py](../code/librerias-python/numba_saxpy.py) — `@vectorize`.
- [numba_matmul.py](../code/librerias-python/numba_matmul.py) — `@cuda.jit`.

---

# CuPy

---

## **CuPy**

- Aceleración de álgebra lineal con el GPU (a través de CUDA).
- Compatible con NumPy.
- Se pueden aplicar 3 tipos de *kernel*:
  - **Elementwise**: aplica una operación a cada elemento de un vector/matriz.
  - **Reduction**: operación de reducción.
  - **Raw**: se puede definir un *kernel* de CUDA directamente en Python.
- Ejemplo: [cupy_op.py](../code/librerias-python/cupy_op.py)

---

# PyCUDA

---

## **PyCUDA**

- Acceso al API de CUDA en Python.
- Para escribir CUDA C/C++ en Python.
- Ejemplo: [pycuda_matmul.py](../code/librerias-python/pycuda_matmul.py)

---

# ¡Gracias!

## Próxima clase: aplicaciones
