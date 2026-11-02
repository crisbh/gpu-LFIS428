---
title: "Proyecto final"
description: "Proyecto final del curso: portar y optimizar un código de física en CUDA"
---

# Proyecto final

El proyecto final consiste en tomar un código de física que corre en la CPU, **portarlo a CUDA** y **optimizarlo**, midiendo y explicando cada mejora. Todos los temas son problemas donde la GPU tiene mucho que ofrecer: mallas grandes, muchas partículas o muchos eventos independientes.

## Modalidad

- Se trabaja en **parejas**. Excepcionalmente se puede trabajar de forma **individual** (se recomienda elegir un tema de dificultad ★).
- Cada grupo recibe **un tema** de la lista de abajo, asignado en la clase de inicio del proyecto: cada grupo indica tres temas en orden de preferencia y no se repiten temas entre grupos.
- Cada tema incluye un **código de referencia en C** (un solo archivo, sin dependencias) que resuelve el problema en la CPU y valida su resultado.

## Etapas

1. **Entender el código de CPU.** Compilarlo, correrlo, leerlo y entender la física y el método numérico. Identificar qué partes concentran el tiempo de cómputo.
2. **Portar a CUDA (versión 1).** Escribir los kernels para las partes costosas. Los datos deben **quedarse en la GPU** durante toda la simulación: sólo se copian al inicio y al final (o cuando se necesita imprimir algo). Validar el resultado contra la versión de CPU.
3. **Optimizar (versiones 2, 3, …).** Aplicar técnicas **vistas en el curso**: memoria compartida, accesos coalescentes (AoS vs. SoA), memoria constante, reducciones paralelas y *warp primitives*, operaciones atómicas, *occupancy*, *streams*, librerías (cuFFT, cuRAND), `float` vs. `double`. Se espera al menos **dos optimizaciones medidas** además de la versión 1, y al menos una de ellas no debe ser sólo cambiar `double` por `float`. Cada tema sugiere además técnicas **no vistas en clases** (marcadas así abajo), como la fusión de *kernels* (motivada por el [costo de lanzar un *kernel*](../slides/kernels.html#30), material de autoestudio): cuentan como optimizaciones, pero hay que estudiarlas por cuenta propia. Se pueden usar librerías no vistas (p. ej. Thrust o CUB para ordenar), explicando qué hacen y midiendo su costo. Cada versión debe seguir siendo correcta.
4. **Analizar el rendimiento.** Para cada versión, medir el tiempo y compararlo con la CPU en función del tamaño del problema. Usar los *profilers* disponibles en Colab, `ncu` (Nsight Compute) y `nvprof` (`nsys` normalmente no está disponible), y el modelo *roofline* para explicar **por qué** una versión es más rápida y qué tan lejos está del máximo del hardware. El ancho de banda efectivo se mide con `ncu --metrics dram__bytes.sum.per_second ./programa.x`, como en las clases.

> **Sobre el *speedup*:** el código de referencia usa un solo núcleo de CPU, así que el *speedup* respecto a él exagera la ventaja de la GPU. Por eso, además del *speedup*, hay que reportar el **ancho de banda efectivo** (GB/s) o los **FLOP/s** alcanzados y compararlos con los máximos de la T4 usados en las clases: {{< katex >}}\approx 300{{< /katex >}} GB/s de memoria y {{< katex >}}\approx 8.1{{< /katex >}} TFLOP/s en `float`. En `double` la T4 (CC 7.5) hace 1 operación por cada 32 de `float`, o sea {{< katex >}}\approx 8.1/32 \approx 0.25{{< /katex >}} TFLOP/s: los códigos de referencia usan `double`, y el *roofline* debe usar el techo que corresponde. El punto de quiebre del *roofline* queda en {{< katex >}}0.25\,\text{TFLOP/s} / 300\,\text{GB/s} \approx 0.8{{< /katex >}} FLOP/byte en `double` y {{< katex >}}\approx 27{{< /katex >}} FLOP/byte en `float`. Basta con contar a mano las operaciones de punto flotante de un paso (cada `+`, `-`, `*`, `/` cuenta 1, como en las clases); en los temas donde eso no tiene sentido (Ising, *ray tracing*) cada tema indica qué métrica reportar.

**Cómo medir**, para que los números de todos los grupos sean comparables:

- La CPU y la GPU se miden **en la misma máquina de Colab** (la del *runtime* con GPU), con el código de CPU compilado tal como viene (`gcc -O2`).
- Se mide **el ciclo principal**, igual que el código de CPU (ver dónde llama a `segundos()`). Las copias entre *host* y *device* y la reserva de memoria se reportan aparte.
- Para el *speedup* se usa el mismo método numérico y los mismos parámetros en ambos lados, salvo que el tema diga otra cosa (ver Poisson).

## Cronograma

Las clases corresponden al [calendario](../calendario/).

| Cuándo | Qué |
|:---|:---|
| Clase 19 | Asignación de temas y parejas |
| Clase 24 (~2 semanas después) | **Reunión 1:1** con el profesor (avance) |
| Antes de la clase 29 | **Entrega del código**, la misma fecha para todos los grupos |
| Clases 29 y 30 | **Presentaciones** de 10 minutos + preguntas |

**A la reunión 1:1** se trae: la versión 1 funcionando y validada contra la CPU (la salida de `comparar.py` o de la validación del tema), una primera tabla de tiempos CPU vs. GPU, un primer perfil con `ncu` o `nvprof`, y el plan de optimizaciones. No hace falta preparar diapositivas.

**La entrega** se hace por *(canal por definir)*, en un solo `.zip` con:

- un archivo por versión (`v1.cu`, `v2.cu`, …), cada uno compilable por sí solo;
- un `README` breve: cómo compilar y ejecutar cada versión, qué cambia de una versión a la siguiente, y una tabla con los tiempos medidos (CPU y cada versión, para varios tamaños);
- las figuras y las diapositivas de la presentación (PDF).

**La presentación** (10 minutos, del orden de 8 a 10 diapositivas, sin demostración en vivo) debe cubrir: el problema físico y el método, cómo se paralelizó, la validación, las optimizaciones y su efecto medido, y el análisis de rendimiento (*roofline*, cuello de botella). Ambos integrantes presentan.

## Pauta de evaluación

| Criterio | Peso |
|:---|:---:|
| Reunión 1:1 de avance: versión 1 correcta, comprensión del código y del plan de optimización | —% |
| Correctitud y validación de todas las versiones | —% |
| Optimización: técnicas aplicadas y mejora medida | —% |
| Análisis de rendimiento: *profiling*, *roofline*, explicación del cuello de botella | —% |
| Física: interpretación de los resultados y figuras | —% |
| Claridad del código, del `README` y de la presentación | —% |

## Herramientas comunes

- [comparar.py](../code/proyectos/comparar.py): compara dos archivos de salida campo por campo e imprime la diferencia máxima absoluta, la relativa y la media. El **primer** archivo es la referencia: `python3 comparar.py salida_cpu.bin salida_gpu.bin`. Acepta archivos en `float` o `double`.
- [graficar.py](../code/proyectos/graficar.py): genera imágenes PNG a partir de un archivo de salida (necesita `comparar.py` en la misma carpeta). Con `--rgb` arma una imagen en color (*ray tracing*) y con `--particulas` un gráfico de posiciones (Lennard-Jones).

Todos los códigos de CPU escriben su resultado con la función `escribir_salida()`: una cabecera con `(nx, ny, ncampos, bytes por real)` seguida de los campos. Schrödinger usa una variante, `escribir_salida_compleja()`, que separa la parte real e imaginaria en dos campos, y *ray tracing* escribe además la imagen en `imagen_cpu.ppm`. La versión GPU debe escribir en el **mismo formato** para poder compararla (basta con copiar la función).

Los códigos de CPU se compilan con `gcc -O2 archivo.c -lm -o archivo.x` y también compilan como C++, así que el archivo se puede renombrar a `.cu` y empezar a portarlo con `nvcc -arch=sm_75 archivo.cu -o archivo.x`, igual que los ejemplos del curso. Todos usan `typedef double real;`: cambiarlo a `float` es una optimización válida, siempre que se muestre que la validación sigue siendo aceptable.

Como en las clases:

- Revisar los errores de cada llamada a CUDA con el macro `CHECK` de [common.h](../code/memoria/common.h). Los lanzamientos de *kernels* no devuelven un código de error: después de cada uno, usar `CHECK(cudaGetLastError())`. `CHECK` sólo imprime el error y sigue, así que hay que mirar el **primer** error que aparece. Para cuFFT y cuRAND (*host*) están `CHECK_CUFFT` y `CHECK_CURAND`.
- Medir el tiempo de los *kernels* con eventos de CUDA (`cudaEventRecord`, `cudaEventElapsedTime`). Si se usa el reloj del host, llamar antes a `cudaDeviceSynchronize()`: los lanzamientos de *kernels* son asíncronos.

**Diferencias esperables entre CPU y GPU:** los kernels que sólo leen y escriben arreglos (*stencils*, colisiones) deberían reproducir la CPU hasta el redondeo ({{< katex >}}\sim 10^{-15}{{< /katex >}} en `double` después de pocos pasos). Las **reducciones** (energía, residuo, promedios) suman en otro orden, así que difieren en las últimas cifras. La suma en árbol de la GPU suele ser incluso **más** precisa que la suma secuencial de la CPU: cada valor pasa por {{< katex >}}\log_2 n{{< /katex >}} sumas en vez de hasta {{< katex >}}n{{< /katex >}}. Por ejemplo, en Gray-Scott con `float` los campos difieren de `double` en {{< katex >}}\sim 10^{-6}{{< /katex >}}, pero el promedio {{< katex >}}\langle u\rangle{{< /katex >}} sumado en serie en la CPU difiere en {{< katex >}}\sim 10^{-4}{{< /katex >}}: ahí el error está en la suma, no en la física. En sistemas caóticos (Lennard-Jones) las diferencias crecen con el tiempo: ahí se comparan observables, no campos.

Cada tema indica abajo los **valores de referencia** de la CPU con los parámetros por defecto (medidos en un Apple M2; los tiempos en Colab serán distintos) y qué cuenta como validado.

### Cómo trabajar en Colab

- Bajar el código de referencia y las herramientas directamente desde el sitio del curso, p. ej. para el tema del calor:

  ```
  !wget -q https://crisbh.github.io/gpu-LFIS428/code/proyectos/calor/calor_cpu.c
  !wget -q https://crisbh.github.io/gpu-LFIS428/code/proyectos/comparar.py
  !wget -q https://crisbh.github.io/gpu-LFIS428/code/proyectos/graficar.py
  !gcc -O2 calor_cpu.c -lm -o calor_cpu.x && ./calor_cpu.x
  ```

- Los archivos de Colab **se borran** cuando se reinicia el *runtime*. Guardar los `.cu` y los resultados en Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) o en un repositorio.
- Las corridas por defecto de CPU tardan de 4 a 30 s en un M2 (en Colab probablemente más). Guardar `salida_cpu.bin` en Drive para no repetirlas.
- Para ver una imagen en el *notebook*: `from IPython.display import Image; Image('salida_cpu_campo0.png')`.

## Temas

| # | Tema | Dificultad |
|:---:|:---|:---:|
| 1 | [Ecuación del calor 2D](#1-ecuación-del-calor-2d) | ★ |
| 2 | [Ecuación de onda 2D](#2-ecuación-de-onda-2d) | ★ |
| 3 | [Reacción-difusión de Gray-Scott](#3-reacción-difusión-de-gray-scott) | ★ |
| 4 | [Ecuación de Poisson 2D](#4-ecuación-de-poisson-2d) | ★★ |
| 5 | [Modelo de Ising 2D](#5-modelo-de-ising-2d) | ★★ |
| 6 | [Ecuación de Schrödinger 2D](#6-ecuación-de-schrödinger-2d) | ★★ |
| 7 | [Ray tracing](#7-ray-tracing) | ★★ |
| 8 | [Fluidos con Lattice Boltzmann](#8-fluidos-con-lattice-boltzmann) | ★★★ |
| 9 | [Dinámica molecular de Lennard-Jones](#9-dinámica-molecular-de-lennard-jones) | ★★★ |

### 1. Ecuación del calor 2D

**Dificultad:** ★ · **Código:** [calor_cpu.c](../code/proyectos/calor/calor_cpu.c)

Difusión del calor en una placa cuadrada con borde a temperatura cero, {{< katex >}}u_t = \alpha \nabla^2 u{{< /katex >}}, con diferencias finitas explícitas (FTCS) y el laplaciano de 5 puntos.

- **Validación:** la condición inicial {{< katex >}}\sin(\pi x)\sin(\pi y){{< /katex >}} decae exactamente como {{< katex >}}e^{-2\pi^2\alpha t}{{< /katex >}}; el código imprime el error máximo relativo respecto a esa solución, y también el error que daría un código que no hace nada. Con los parámetros por defecto la placa apenas se enfría (la amplitud baja de 1 a 0.9953), así que para validar se usa una malla menor con más pasos: `./calor_cpu.x 256 10000`, donde la amplitud baja a ~0.55. Validado = el mismo error relativo que la CPU y `comparar.py` con diferencias {{< katex >}}\lesssim 10^{-13}{{< /katex >}} en `double`.
- **Valores de referencia:** por defecto ({{< katex >}}N = 2048{{< /katex >}}, 5000 pasos): error relativo {{< katex >}}1.3\times 10^{-9}{{< /katex >}}, 10.3 s. Con `256 10000`: error relativo {{< katex >}}1.1\times 10^{-5}{{< /katex >}} (sin evolucionar sería 0.84), 0.2 s.
- **Métrica:** ancho de banda efectivo (GB/s).
- **Optimización mínima:** un kernel por paso con los datos en la GPU e intercambio de punteros (sin copias entre pasos).
- **Optimizaciones avanzadas:** *tiles* en memoria compartida, como en la transpuesta por bloques, pero cada *tile* necesita además un borde (*halo*) con los vecinos *(no visto en clases)*; varios pasos de tiempo por *kernel* (*temporal blocking*) *(no visto en clases)*; `float`; comparar el ancho de banda alcanzado con el máximo de la T4.
- **Extensiones:** fuentes de calor, conductividad variable, 3D.

### 2. Ecuación de onda 2D

**Dificultad:** ★ · **Código:** [onda_cpu.c](../code/proyectos/onda/onda_cpu.c)

Vibraciones de una membrana cuadrada fija en el borde, {{< katex >}}u_{tt} = c^2 \nabla^2 u{{< /katex >}}, con el esquema *leapfrog* (tres niveles de tiempo).

- **Validación:** el modo normal (1,1) oscila con {{< katex >}}\omega = \sqrt{2}\,\pi c{{< /katex >}}; por defecto se simula un período completo y se compara con la solución exacta. Además, el esquema conserva una energía discreta, cuya variación se imprime. Al final de un período la solución está en un máximo de {{< katex >}}\cos\omega t{{< /katex >}}, donde un error de fase casi no se nota: la prueba más estricta es comparar el campo con `comparar.py` después de pocos pasos (p. ej. `./onda_cpu.x 2048 100`), que debe coincidir hasta {{< katex >}}\sim 10^{-15}{{< /katex >}}. Ojo: si se da el número de pasos, el código usa {{< katex >}}\Delta t = 0.5\,h/c{{< /katex >}} en vez de dividir el período; la versión GPU debe repetir esa misma lógica.
- **Valores de referencia:** por defecto ({{< katex >}}N = 2048{{< /katex >}}, 5790 pasos): error {{< katex >}}6.9\times 10^{-14}{{< /katex >}}, variación relativa de la energía {{< katex >}}3.1\times 10^{-13}{{< /katex >}}, 16.2 s. Validado = errores del mismo orden en `double`; en `float`, medir y discutir cuánto empeoran.
- **Métrica:** ancho de banda efectivo (GB/s).
- **Optimización mínima:** kernel de un paso con rotación de tres punteros en la GPU (el primer paso, `primer_paso()`, también necesita su *kernel*).
- **Optimizaciones avanzadas:** memoria compartida; cálculo de la energía como reducción en la GPU; `float` (¿se sigue conservando la energía?).
- **Extensiones:** pulso inicial localizado, obstáculos (rendija doble), velocidad de propagación variable.

### 3. Reacción-difusión de Gray-Scott

**Dificultad:** ★ · **Código:** [reaccion-difusion_cpu.c](../code/proyectos/reaccion-difusion/reaccion-difusion_cpu.c)

Dos especies químicas que reaccionan y difunden forman patrones (manchas, laberintos, "mitosis"):

{{< katex display=true >}}
u_t = D_u \nabla^2 u - u v^2 + F(1-u), \qquad v_t = D_v \nabla^2 v + u v^2 - (F+k) v
{{< /katex >}}

- **Validación:** no hay solución exacta. El campo de la GPU debe coincidir con `salida_cpu.bin` (usar `comparar.py`): en `double`, hasta {{< katex >}}\sim 10^{-15}{{< /katex >}} tras 100 pasos y {{< katex >}}\lesssim 10^{-9}{{< /katex >}} tras los 10000 pasos por defecto (en la CPU, sólo cambiar el redondeo de las multiplicaciones-sumas da {{< katex >}}10^{-10}{{< /katex >}}). Además, comparar los promedios {{< katex >}}\langle u\rangle, \langle v\rangle{{< /katex >}} y los patrones. En `float`, el promedio sumado en serie en la CPU es el lado impreciso (ver arriba).
- **Valores de referencia:** por defecto ({{< katex >}}N = 1024{{< /katex >}}, 10000 pasos): {{< katex >}}\langle u\rangle = 0.7701597506{{< /katex >}}, {{< katex >}}\langle v\rangle = 0.0831220727{{< /katex >}}, 18.6 s.
- **Métrica:** ancho de banda efectivo (GB/s).
- **Optimización mínima:** un kernel que actualiza ambos campos con bordes periódicos.
- **Optimizaciones avanzadas:** memoria compartida para ambos campos; evitar el operador `%` en los bordes; `float`; guardar imágenes cada cierto número de pasos (llamando a `escribir_salida()` con otro nombre de archivo) sin frenar el cálculo, copiando con *streams* mientras la GPU sigue calculando.
- **Extensiones:** explorar el plano {{< katex >}}(F, k){{< /katex >}} y clasificar los patrones; animaciones.

### 4. Ecuación de Poisson 2D

**Dificultad:** ★★ · **Código:** [poisson_cpu.c](../code/proyectos/poisson/poisson_cpu.c)

Potencial electrostático a partir de una densidad de carga, {{< katex >}}\nabla^2 u = f{{< /katex >}}, resuelto iterativamente con Jacobi, Gauss-Seidel rojo-negro o SOR rojo-negro.

- **Validación:** con la fuente usada, la solución exacta es {{< katex >}}\sin(\pi x)\sin(\pi y){{< /katex >}}; el error debe bajar ~4 veces al duplicar {{< katex >}}N{{< /katex >}} (orden {{< katex >}}h^2{{< /katex >}}). Validado = mismo número de iteraciones (±100, porque el residuo se revisa cada 100) y el mismo error que la CPU con el mismo método.
- **Valores de referencia:** por defecto ({{< katex >}}N = 1024{{< /katex >}}, SOR): 4600 iteraciones, error {{< katex >}}7.9\times 10^{-7}{{< /katex >}}, 3.8 s. Con {{< katex >}}N = 128{{< /katex >}}: Jacobi necesita ~60300 iteraciones, Gauss-Seidel rojo-negro ~30700 y SOR ~600.
- **Qué comparar:** Jacobi necesita {{< katex >}}O(N^2){{< /katex >}} iteraciones, así que una GPU con Jacobi probablemente **pierde** contra la CPU con SOR en {{< katex >}}N = 1024{{< /katex >}}: eso es esperable y es parte de la historia. Reportar dos comparaciones: (a) el mismo método en CPU y GPU (Jacobi con {{< katex >}}N \le 256{{< /katex >}}, donde la CPU termina en segundos, y SOR una vez portado), y (b) el mejor método de la GPU contra el mejor de la CPU (SOR) en {{< katex >}}N = 1024{{< /katex >}} y mayores.
- **Métrica:** ancho de banda efectivo (GB/s) y tiempo por iteración; para comparar métodos, tiempo total hasta la tolerancia.
- **Optimización mínima:** Jacobi en la GPU (es trivialmente paralelo) con el residuo calculado como **reducción** en la GPU.
- **Optimizaciones avanzadas:** rojo-negro / SOR en la GPU *(no visto en clases)*, el código de CPU lo explica: ¿por qué Gauss-Seidel normal no se puede paralelizar? Para que la GPU le gane a la CPU con SOR, esta optimización es prácticamente necesaria. Revisar el residuo cada cierto número de iteraciones (la CPU lo hace cada 100; en la GPU cada revisión es una reducción más una copia al *host*: medir su costo y elegir la frecuencia); memoria compartida; comparar iteraciones vs. tiempo por iteración de cada método.
- **Extensiones:** condensador de placas paralelas u otras geometrías con condiciones de borde no nulas; *multigrid*.

### 5. Modelo de Ising 2D

**Dificultad:** ★★ · **Código:** [ising_cpu.c](../code/proyectos/ising/ising_cpu.c)

Transición de fase ferromagnética con Monte Carlo (algoritmo de Metropolis) en una red cuadrada, barriendo temperaturas alrededor de {{< katex >}}T_c = 2/\ln(1+\sqrt{2}) \approx 2.269{{< /katex >}}.

- **Validación:** la magnetización {{< katex >}}\langle|m|\rangle{{< /katex >}} y la energía por espín se comparan con la CPU y con la solución exacta de Onsager (el código la imprime al lado). Como la GPU usará otros números aleatorios, se comparan los observables, no las configuraciones. El código imprime el **error estándar** de cada uno (método de bloques) y acepta la semilla como último argumento, para ver cuánto cambian los resultados entre corridas. Validado = CPU y GPU coinciden dentro de {{< katex >}}3\sigma{{< /katex >}} (con {{< katex >}}\sigma^2 = \sigma_\text{CPU}^2 + \sigma_\text{GPU}^2{{< /katex >}}) lejos de {{< katex >}}T_c{{< /katex >}} ({{< katex >}}T \le 2.1{{< /katex >}} y {{< katex >}}T \ge 2.5{{< /katex >}}). Cerca de {{< katex >}}T_c{{< /katex >}} las medidas sucesivas están muy correlacionadas (ralentización crítica): las barras de error quedan subestimadas y dos semillas pueden dar valores muy distintos, así que ahí la comparación es sólo cualitativa. Sobre {{< katex >}}T_c{{< /katex >}} una red finita no llega a {{< katex >}}m = 0{{< /katex >}}: el {{< katex >}}\langle|m|\rangle{{< /katex >}} medido decrece con {{< katex >}}L{{< /katex >}}.
- **Valores de referencia:** por defecto ({{< katex >}}L = 512{{< /katex >}}, 1000 + 1000 barridos, semilla 12345): en {{< katex >}}T = 1.9{{< /katex >}}, {{< katex >}}\langle|m|\rangle = 0.93830 \pm 0.00006{{< /katex >}} y {{< katex >}}\langle e\rangle = -1.80954 \pm 0.00013{{< /katex >}} (Onsager: 0.938304 y −1.809490); en {{< katex >}}T = 2.3{{< /katex >}}, {{< katex >}}\langle|m|\rangle = 0.20 \pm 0.01{{< /katex >}} (error subestimado); 28.9 s.
- **Métrica:** intentos de volteo por nanosegundo (el código imprime ns por intento) y ancho de banda efectivo.
- **Optimización mínima:** actualización en tablero de ajedrez *(no visto en clases)*, el código de CPU lo explica, con un generador de números aleatorios por *thread* (cuRAND en el *device*, como en [curand_device.cu](../code/librerias-python/curand_device.cu)); magnetización y energía como reducciones en la GPU. Inicializar los estados de cuRAND (`curand_init`) es caro: hacerlo **una sola vez**, guardarlos en memoria global y reusarlos en cada lanzamiento. {{< katex >}}L{{< /katex >}} debe ser par (si no, el tablero de ajedrez con bordes periódicos tiene vecinos del mismo color).
- **Optimizaciones avanzadas:** espines como `char` (o bits); tabla de probabilidades en memoria constante; simular todas las temperaturas a la vez; memoria compartida.
- **Extensiones:** escalamiento de tamaño finito (cumulante de Binder para estimar {{< katex >}}T_c{{< /katex >}}), campo magnético externo.

### 6. Ecuación de Schrödinger 2D

**Dificultad:** ★★ · **Código:** [schrodinger_cpu.c](../code/proyectos/schrodinger/schrodinger_cpu.c)

Evolución temporal de un paquete de ondas en un oscilador armónico, {{< katex >}}i\,\partial_t\psi = -\tfrac{1}{2}\nabla^2\psi + V\psi{{< /katex >}}, con el método *split-step* de Fourier: la energía potencial se aplica en el espacio de posiciones y la cinética en el de momentos, pasando de uno a otro con FFTs.

- **Validación:** la norma se conserva y el centro del paquete sigue la trayectoria clásica {{< katex >}}\langle x\rangle = x_0\cos(\omega t){{< /katex >}}; ambas se imprimen durante la simulación. El paso final cae cerca de un máximo de {{< katex >}}\cos\omega t{{< /katex >}}, donde el error se nota poco: mirar también las filas intermedias. Validado = la misma tabla que la CPU y `comparar.py` sobre la función de onda final con diferencias {{< katex >}}\lesssim 10^{-10}{{< /katex >}} en `double`.
- **Valores de referencia:** por defecto ({{< katex >}}N = 512{{< /katex >}}, 628 pasos): variación relativa de la norma {{< katex >}}6\times 10^{-15}{{< /katex >}}, error final en {{< katex >}}\langle x\rangle{{< /katex >}} de {{< katex >}}2.5\times 10^{-7}{{< /katex >}} (a mitad de camino llega a {{< katex >}}\sim 6\times 10^{-5}{{< /katex >}}, dominado por {{< katex >}}\Delta t{{< /katex >}}), 10.0 s.
- **Métrica:** ancho de banda efectivo de los *kernels* propios y de cuFFT (con `ncu`).
- **Optimización mínima:** reemplazar la FFT propia del código por **cuFFT** y portar las multiplicaciones por fases a kernels. El arreglo complejo intercalado del código tiene el mismo formato que `cufftDoubleComplex`; en 2D se usa `cufftPlan2d(&plan, N, N, CUFFT_Z2Z)` (las diapositivas muestran el caso 1D). Ojo: la transformada **inversa** de cuFFT no está normalizada; hay que multiplicar por {{< katex >}}1/N^2{{< /katex >}}, por ejemplo en el mismo *kernel* que aplica la fase del potencial.
- **Sobre el *speedup*:** la FFT del código de CPU es simple y recalcula senos y cosenos en cada mariposa, así que es mucho más lenta que una FFT optimizada: el *speedup* respecto a ella exagera todavía más. Comparar también con `numpy.fft.fft2` es opcional pero bienvenido.
- **Optimizaciones avanzadas:** fusionar *kernels* *(no visto en clases)*; calcular las fases al vuelo en vez de leerlas de memoria; calcular los observables con reducciones y sólo de vez en cuando; `float` (`cufftComplex`). Como la mayor parte del tiempo se va en cuFFT, el análisis puede centrarse en qué fracción del ancho de banda máximo alcanza cuFFT y cuánto se gana al reducir los *kernels* propios.
- **Extensiones:** efecto túnel a través de una barrera; rendija doble; potencial dependiente del tiempo.

### 7. Ray tracing

**Dificultad:** ★★ · **Código:** [ray-tracing_cpu.c](../code/proyectos/ray-tracing/ray-tracing_cpu.c)

Generación de imágenes siguiendo rayos de luz (*path tracing*), basada en la escena final de *Ray Tracing in One Weekend* (P. Shirley): ~490 esferas difusas, metálicas y de vidrio. La óptica es geométrica: reflexión especular, refracción con la ley de Snell {{< katex >}}n_1 \sin\theta_1 = n_2 \sin\theta_2{{< /katex >}}, reflexión total interna y la reflectancia de Fresnel en la aproximación de Schlick. Cada píxel promedia muchos rayos aleatorios (Monte Carlo). El código está en C con *structs* y un `switch` por material (sin clases ni funciones virtuales, que complican el paso a la GPU).

- **Validación:** el modo 1 es la prueba del "horno blanco": esferas que no absorben luz bajo un cielo uniforme de intensidad 1. Por conservación de la energía todos los píxeles deben valer 1 (el código imprime el brillo mínimo y el promedio). En `double` el resultado es 1 hasta el redondeo; en `float` algunos rayos se pierden por redondeo y unos pocos píxeles aislados bajan a ~0.98–0.99 (un rayo de 50 o 100 muestras): eso es aceptable, mientras el promedio siga siendo 1 hasta {{< katex >}}\sim 10^{-3}{{< /katex >}}. En la escena del libro, cada píxel tiene su propio generador de números aleatorios, así que usando la misma función `aleatorio()` en la GPU la imagen debe coincidir con `salida_cpu.bin`: casi todos los píxeles quedan iguales y unos pocos difieren porque el redondeo cambia algún rebote, así que hay que mirar la diferencia **media** de `comparar.py`. Validado = diferencia media {{< katex >}}\lesssim 10^{-5}{{< /katex >}} en `double` y {{< katex >}}\sim 10^{-3}{{< /katex >}} en `float`.
- **Valores de referencia:** por defecto (400 × 225 píxeles, 100 muestras): brillo promedio 0.375943, 13.0 s (0.69 millones de rayos por segundo). Horno blanco (`./ray-tracing_cpu.x 1`): promedio y mínimo 1.000000 en `double`; con `float`, promedio 0.999999 y mínimo 0.990.
- **Métrica:** rayos por segundo (el código imprime los rayos de cámara por segundo). Contar FLOP a mano no tiene mucho sentido aquí (raíces, potencias, rebotes de largo variable): usar `ncu` para ver qué limita el *kernel*.
- **Imagen:** `python3 graficar.py salida_gpu.bin --rgb` arma la imagen en color; también se puede copiar la escritura del PPM del código de CPU.
- **Optimización mínima:** un *thread* por píxel con las esferas en la memoria de la GPU; cada *thread* inicializa su generador con el índice del píxel. Todas las funciones que usa el *kernel* necesitan `__host__ __device__` (o `__device__`).
- **Optimizaciones avanzadas:** esferas en **memoria constante** (todos los *threads* de un *warp* recorren las mismas esferas en el mismo orden, el caso ideal de la memoria constante; las 487 esferas ocupan ~35 KB en `double`, dentro de los 64 KB disponibles, pero escenas mucho más grandes ya no caben) o en memoria compartida; `float` (la T4 hace 32 veces menos operaciones por segundo en `double`); medir la **divergencia de *warps*** (rayos vecinos rebotan en materiales distintos o terminan en distinto número de rebotes) y probar bloques 2D pequeños para que los rayos de un *warp* sean coherentes; cuRAND en vez del *hash* propio. Más allá del curso: una jerarquía de volúmenes envolventes (BVH) para no revisar todas las esferas *(no visto en clases)*.
- **Extensiones:** escenas más grandes, otros objetos (planos, triángulos), luces de área, dispersión cromática (índice de refracción dependiente de la longitud de onda: arcoíris en una esfera de agua).

### 8. Fluidos con Lattice Boltzmann

**Dificultad:** ★★★ · **Código:** [lattice-boltzmann_cpu.c](../code/proyectos/lattice-boltzmann/lattice-boltzmann_cpu.c)

Simulación de un fluido con el método de Lattice Boltzmann D2Q9: cada nodo guarda 9 distribuciones que colisionan (localmente) y se propagan a los vecinos. Dos modos: flujo en un canal (Poiseuille) y flujo alrededor de un cilindro (calle de vórtices de von Kármán).

- **Validación:** en el modo canal, el perfil de velocidades debe ser la parábola de Poiseuille (el código imprime el error). El fluido parte del reposo y se acerca a la parábola como {{< katex >}}e^{-t/t_\text{rel}}{{< /katex >}}, con {{< katex >}}t_\text{rel} = H^2/(\pi^2\nu){{< /katex >}} pasos: si se agranda el canal hay que correr muchos más pasos (el código los ajusta solo si no se dan, y avisa si son pocos). Por eso: **validar con el tamaño por defecto** y medir rendimiento con tamaños mayores, donde no hace falta llegar al estado estacionario. En el modo cilindro, comparar los campos con `salida_cpu.bin` usando `comparar.py`. Validado = el mismo error de Poiseuille que la CPU y diferencias {{< katex >}}\lesssim 10^{-10}{{< /katex >}} en el modo cilindro (en `double`).
- **Valores de referencia:** modo 0 por defecto (512 × 66, 29900 pasos = 12 {{< katex >}}t_\text{rel}{{< /katex >}}): error relativo {{< katex >}}8.1\times 10^{-5}{{< /katex >}} (el error de discretización), 29.3 s, 34.5 MLUPS. Modo 1 (512 × 128, 10000 pasos): Reynolds ~123, {{< katex >}}\langle u_x\rangle = 0.051505{{< /katex >}}, 18.8 s.
- **Métrica:** ancho de banda efectivo (GB/s) y MLUPS (millones de nodos actualizados por segundo, la métrica usual de LBM; el código de CPU ya la imprime y cuenta también los nodos sólidos).
- **Optimización mínima:** kernels de colisión y propagación con los datos en la GPU. Las tablas de velocidades y pesos son constantes globales del *host*: en CUDA hay que pasarlas a memoria constante (`__constant__`) o copiarlas.
- **Optimizaciones avanzadas:** pasar de AoS a **SoA**. Según el criterio de la clase de aplicaciones, AoS conviene cuando cada *thread* usa todos los campos de **su** elemento; en la propagación, en cambio, cada nodo lee una sola {{< katex >}}f_i{{< /katex >}} de cada uno de 9 vecinos distintos, que es el caso donde SoA gana. Fusionar colisión y propagación en un solo *kernel* *(no visto en clases)*; `float`. LBM está limitado por memoria: comparar el ancho de banda efectivo con el máximo de la T4.
- **Extensiones:** número de Reynolds y frecuencia de desprendimiento de vórtices (número de Strouhal); otros obstáculos; condiciones de entrada y salida.

### 9. Dinámica molecular de Lennard-Jones

**Dificultad:** ★★★ · **Código:** [lennard-jones_cpu.c](../code/proyectos/lennard-jones/lennard-jones_cpu.c)

Un gas/líquido de partículas que interactúan con el potencial de Lennard-Jones de corto alcance, integrado con *velocity Verlet*. A diferencia del ejemplo de N-cuerpos del curso, la interacción se corta en {{< katex >}}r_c = 2.5{{< /katex >}} y se usan **listas de celdas**: cada partícula sólo mira las de su celda y las 8 vecinas.

- **Validación:** la energía total se conserva (el código imprime su variación). La variación no es cero: la fuerza no es continua en {{< katex >}}r_c{{< /katex >}} y {{< katex >}}\Delta t{{< /katex >}} es finito. Como la dinámica es caótica, las posiciones de la GPU sólo coinciden con la CPU durante pocos pasos: comparar con `comparar.py` después de 10 a 100 pasos (p. ej. `./lennard-jones_cpu.x 256 100`), donde deberían coincidir hasta {{< katex >}}\lesssim 10^{-10}{{< /katex >}} en `double` (la GPU suma las fuerzas en otro orden). Para corridas largas, comparar energía y temperatura. Validado = posiciones iguales a pocos pasos y, a 1000 pasos, una variación de la energía del mismo orden que la CPU.
- **Valores de referencia:** por defecto ({{< katex >}}N = 65536{{< /katex >}}, 1000 pasos): variación relativa de la energía {{< katex >}}6.1\times 10^{-4}{{< /katex >}}, casi toda en los primeros 100 pasos (la red inicial perfecta se desordena); después {{< katex >}}E/N{{< /katex >}} fluctúa alrededor de −0.49996 en {{< katex >}}\sim 2\times 10^{-5}{{< /katex >}}. 17.6 s.
- **Métrica:** pasos por segundo e interacciones de pares por segundo; ancho de banda efectivo con `ncu`.
- **Optimización mínima:** un *thread* por partícula para las fuerzas y la integración; construir las celdas en la GPU (las listas enlazadas necesitan **operaciones atómicas**). En una primera versión se acepta construir las celdas en la CPU (copiando las posiciones en cada paso), midiendo cuánto cuesta esa copia; la versión siguiente debe hacerlo en la GPU.
- **Optimizaciones avanzadas:** reemplazar las listas enlazadas por un ordenamiento de partículas por celda (accesos coalescentes) *(no visto en clases)*, por ejemplo con `thrust::sort_by_key`; memoria compartida por celda; `float`.
- **Extensiones:** termostato, función de distribución radial {{< katex >}}g(r){{< /katex >}}, transición líquido-sólido al bajar la temperatura.
