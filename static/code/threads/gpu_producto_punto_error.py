# Taken from https://www.experoinc.com/post/a-quick-note-on-gpu-accuracy-and-double-precision

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from matplotlib import pyplot as plt

N = 10000000

d = np.random.rand(N)
f = d.astype(np.float32)
g = gpuarray.to_gpu(f)

pd = np.dot(d, d)
# Producto punto secuencial en float32: cumsum acumula en orden, como un ciclo for.
ps = np.cumsum(f * f, dtype=np.float32)[-1]
# np.dot NO es secuencial: llama a BLAS, que acumula en bloques.
pf = np.dot(f, f)
pg = gpuarray.dot(g, g).get()

serr = np.abs(pd - ps)
ferr = np.abs(pd - pf)
gerr = np.abs(pd - pg)

print("Producto punto de referencia (float64):", pd)
print("CPU float32, secuencial:", ps, "error:", serr)
print("CPU float32, BLAS:      ", pf, "error:", ferr)
print("GPU float32:            ", pg, "error:", gerr)

plt.bar(["CPU secuencial", "CPU BLAS", "GPU"], [serr, ferr, gerr], color=["b", "c", "g"])
plt.yscale("log")
plt.show()
