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

dsum = d.sum()
# Suma secuencial en float32: cumsum acumula en orden, como un ciclo for.
ssum = np.cumsum(f, dtype=np.float32)[-1]
# f.sum() NO es secuencial: NumPy usa suma por pares (un árbol, como el GPU).
fsum = f.sum()
gsum = gpuarray.sum(g).get()

serr = np.abs(dsum - ssum)
ferr = np.abs(dsum - fsum)
gerr = np.abs(dsum - gsum)

print("Suma de referencia (float64):", dsum)
print("CPU float32, secuencial:     ", ssum, "error:", serr)
print("CPU float32, NumPy por pares:", fsum, "error:", ferr)
print("GPU float32:                 ", gsum, "error:", gerr)

plt.bar(["CPU secuencial", "CPU por pares", "GPU"], [serr, ferr, gerr], color=["b", "c", "g"])
plt.yscale("log")
plt.show()
