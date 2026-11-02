"""
Compara dos archivos de salida de los proyectos (p. ej. CPU vs. GPU).

Formato de los archivos: una cabecera de 4 enteros de 32 bits
(nx, ny, ncampos, bytes por real) seguida de ncampos campos, cada uno con
nx*ny valores (float si bytes = 4, double si bytes = 8) en orden [y][x].
Es lo que escribe la función escribir_salida() de los códigos de CPU.

Uso:  python3 comparar.py salida_cpu.bin salida_gpu.bin

El primer archivo es la referencia (normalmente el de CPU). Para cada campo
imprime la diferencia máxima absoluta, la relativa (respecto al máximo valor
absoluto del primer archivo) y la diferencia media absoluta
(útil cuando sólo unos pocos puntos difieren, p. ej. en ray tracing). Los
archivos pueden tener distinta precisión (float vs. double).
"""

import sys

import numpy as np


def leer(nombre):
    cabecera = np.fromfile(nombre, dtype=np.int32, count=4)
    nx, ny, ncampos, bytes_real = (int(c) for c in cabecera)
    tipo = {4: np.float32, 8: np.float64}[bytes_real]
    datos = np.fromfile(nombre, dtype=tipo, offset=16)
    esperado = ncampos * ny * nx
    if datos.size != esperado:
        print(
            f"{nombre}: la cabecera dice {ncampos} campos de {nx} x {ny} "
            f"({esperado} valores), pero el archivo tiene {datos.size}. "
            "¿Se escribió completo, con el mismo tipo que dice la cabecera?"
        )
        sys.exit(1)
    return datos.reshape(ncampos, ny, nx).astype(np.float64), bytes_real


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    a, bytes_a = leer(sys.argv[1])
    b, bytes_b = leer(sys.argv[2])
    if a.shape != b.shape:
        print(f"Las dimensiones no coinciden: {a.shape} vs. {b.shape}")
        sys.exit(1)

    print(f"Dimensiones (campos, ny, nx): {a.shape}")
    print(f"Bytes por real: {bytes_a} vs. {bytes_b}")
    for c in range(a.shape[0]):
        dif = np.abs(a[c] - b[c]).max()
        escala = np.abs(a[c]).max()
        rel = dif / escala if escala > 0 else dif
        media = np.abs(a[c] - b[c]).mean()
        print(
            f"Campo {c}: diferencia máxima = {dif:.3e}, relativa = {rel:.3e}, "
            f"media = {media:.3e}"
        )


if __name__ == "__main__":
    main()
