"""
Grafica un archivo de salida de los proyectos y lo guarda como PNG.

Formato del archivo: ver comparar.py.

Uso:  python3 graficar.py salida_cpu.bin [--particulas]

- Campos 2D (ny > 1): una imagen por campo (salida_cpu_campo0.png, ...).
- Con --particulas (proyecto lennard-jones): dispersión de los campos 0 y 1
  (x, y) en salida_cpu_particulas.png.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comparar import leer


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    nombre = sys.argv[1]
    base = nombre.rsplit(".", 1)[0]
    datos, _ = leer(nombre)
    ncampos, ny, nx = datos.shape

    if "--particulas" in sys.argv:
        x, y = datos[0, 0], datos[1, 0]
        plt.figure(figsize=(7, 7))
        plt.scatter(x, y, s=0.2, c="k")
        plt.gca().set_aspect("equal")
        plt.savefig(f"{base}_particulas.png", dpi=150)
        print(f"Guardado {base}_particulas.png")
        return

    for c in range(ncampos):
        plt.figure(figsize=(8, 8 * ny / nx + 1))
        if ny > 1:
            plt.imshow(datos[c], origin="lower", cmap="viridis")
            plt.colorbar(shrink=0.8)
        else:
            plt.plot(datos[c, 0])
        plt.title(f"{nombre}: campo {c}")
        plt.savefig(f"{base}_campo{c}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Guardado {base}_campo{c}.png")


if __name__ == "__main__":
    main()
