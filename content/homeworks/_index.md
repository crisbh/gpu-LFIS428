---
title: "Proyecto final"
description: "Proyecto final del curso: portar y optimizar un código de física en CUDA"
---

# Proyecto final

El proyecto final consiste en tomar un código de física que corre en la CPU, **portarlo a CUDA** y **optimizarlo**, midiendo y explicando cada mejora. Todos los temas son problemas donde la GPU tiene mucho que ofrecer: mallas grandes, muchas partículas o muchos eventos independientes.

## Modalidad

- Se trabaja en **parejas**. Excepcionalmente se puede trabajar de forma **individual** (se recomienda elegir un tema de dificultad ★).
- Cada grupo recibe **un tema** de la lista de abajo, asignado al inicio del proyecto: cada grupo indica tres temas distintos en orden de preferencia y no se repiten temas entre grupos. Si dos grupos piden el mismo tema, la asignación busca que ningún grupo quede fuera de sus tres preferencias y que la menor cantidad posible reciba la tercera; los empates se resuelven por sorteo.
- Cada tema incluye un **código de referencia en C** (un solo archivo, sin dependencias) que resuelve el problema en la CPU y valida su resultado. Se puede partir de **otro código** de CPU si se prefiere, indicando de dónde viene y mostrando una validación equivalente.

## Etapas

1. **Entender el código de CPU.** Compilarlo, correrlo, leerlo y entender la física y el método numérico. Identificar qué partes concentran el tiempo de cómputo.
2. **Portar a CUDA (versión 1).** Los datos deben **quedarse en la GPU** durante toda la simulación: sólo se copian al inicio y al final (o cuando se necesita imprimir algo). Validar el resultado contra la versión de CPU.
3. **Optimizar (versiones 2, 3, …).** Decidir qué optimizar y cómo, y justificarlo con mediciones. Se esperan al menos **dos optimizaciones medidas** además de la versión 1, y al menos una de ellas no debe ser sólo cambiar `double` por `float`. Vale usar técnicas o librerías no vistas en clases, explicando qué hacen. Cada versión debe seguir siendo correcta.
4. **Analizar el rendimiento.** Para cada versión, medir el tiempo y compararlo con la CPU en función del tamaño del problema. Usar `ncu` y el modelo *roofline* para explicar **por qué** una versión es más rápida y qué tan lejos está del máximo del hardware.

**Cómo medir**, para que los números de todos los grupos sean comparables:

- La CPU y la GPU se miden **en la misma máquina de Colab**, con el código de CPU compilado con `gcc -O2`, y con el mismo método numérico y los mismos parámetros.
- Se mide **el ciclo principal**; las copias entre *host* y *device* se reportan aparte. Para los *kernels*, usar eventos de CUDA.
- El código de referencia usa un solo núcleo de CPU, así que el *speedup* exagera la ventaja de la GPU. Reportar además el **ancho de banda efectivo** (GB/s) o los **FLOP/s** alcanzados, comparados con los máximos de la T4: {{< katex >}}\approx 300{{< /katex >}} GB/s de memoria, {{< katex >}}\approx 8.1{{< /katex >}} TFLOP/s en `float` y {{< katex >}}\approx 0.25{{< /katex >}} TFLOP/s en `double`.
- Las reducciones (energía, residuo, promedios) suman en otro orden en la GPU, así que difieren de la CPU en las últimas cifras: eso no es un error.

## Cronograma

Las fechas exactas se avisan en clases.

| Cuándo | Qué |
|:---|:---|
| Inicio del proyecto | Asignación de temas y parejas |
| ~2 semanas después | **Reunión 1:1** con el profesor (avance) |
| Antes de las presentaciones | **Entrega del código**, la misma fecha para todos los grupos |
| Últimas clases del curso | **Presentaciones** de 10 minutos + preguntas |

**A la reunión 1:1** se trae: la versión 1 funcionando y validada contra la CPU, una primera tabla de tiempos CPU vs. GPU, un primer perfil con `ncu`, y el plan de optimizaciones. No hace falta preparar diapositivas.

**La entrega** se hace por **correo electrónico** al profesor, con un solo `.zip` adjunto que contiene:

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

## Cómo trabajar en Colab

- Bajar el código de referencia desde el sitio del curso, p. ej. para el tema del calor:

  ```
  !wget -q https://crisbh.github.io/gpu-LFIS428/code/proyectos/calor/calor_cpu.c
  !gcc -O2 calor_cpu.c -lm -o calor_cpu.x && ./calor_cpu.x
  ```

- Los códigos compilan también como C++, así que se pueden renombrar a `.cu` y empezar a portarlos con `nvcc -arch=sm_75`. Todos usan `typedef double real;`.
- Los archivos de Colab **se borran** cuando se reinicia el *runtime*. Guardar los `.cu` y los resultados en Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) o en un repositorio.

## Temas

| # | Tema | Dificultad |
|:---:|:---|:---:|
| 1 | [Ecuación del calor 2D](#1-ecuación-del-calor-2d) | ★ |
| 2 | [Ecuación de onda 2D](#2-ecuación-de-onda-2d) | ★ |
| 3 | [Ecuación de Poisson 2D](#3-ecuación-de-poisson-2d) | ★★ |
| 4 | [Modelo de Ising 2D](#4-modelo-de-ising-2d) | ★★ |
| 5 | [Ecuación de Schrödinger 2D](#5-ecuación-de-schrödinger-2d) | ★★ |
| 6 | [Ray tracing](#6-ray-tracing) | ★★ |
| 7 | [Dinámica molecular de Lennard-Jones](#7-dinámica-molecular-de-lennard-jones) | ★★★ |

### 1. Ecuación del calor 2D

**Dificultad:** ★ · **Código:** [calor_cpu.c](../code/proyectos/calor/calor_cpu.c)

Difusión del calor en una placa cuadrada con borde a temperatura cero, {{< katex >}}u_t = \alpha \nabla^2 u{{< /katex >}}, con diferencias finitas explícitas (FTCS) y el laplaciano de 5 puntos.

- **Validación:** la condición inicial {{< katex >}}\sin(\pi x)\sin(\pi y){{< /katex >}} decae exactamente como {{< katex >}}e^{-2\pi^2\alpha t}{{< /katex >}}; el código imprime el error respecto a esa solución.

### 2. Ecuación de onda 2D

**Dificultad:** ★ · **Código:** [onda_cpu.c](../code/proyectos/onda/onda_cpu.c)

Vibraciones de una membrana cuadrada fija en el borde (un tambor), {{< katex >}}u_{tt} = c^2 \nabla^2 u{{< /katex >}}, con el esquema *leapfrog* (tres niveles de tiempo).

- **Validación:** el modo normal (1,1) oscila con {{< katex >}}\omega = \sqrt{2}\,\pi c{{< /katex >}} y el esquema conserva una energía discreta; el código imprime el error y la variación de la energía.

### 3. Ecuación de Poisson 2D

**Dificultad:** ★★ · **Código:** [poisson_cpu.c](../code/proyectos/poisson/poisson_cpu.c)

Potencial electrostático a partir de una densidad de carga, {{< katex >}}\nabla^2 u = f{{< /katex >}}. No hay evolución en el tiempo: se busca el equilibrio iterando, con Jacobi, Gauss-Seidel rojo-negro o SOR rojo-negro.

- **Validación:** la solución exacta es {{< katex >}}\sin(\pi x)\sin(\pi y){{< /katex >}}; con el mismo método, la GPU debe necesitar las mismas iteraciones y dar el mismo error que la CPU.
- **Ojo:** los métodos necesitan números de iteraciones muy distintos. Una GPU con Jacobi puede perder contra la CPU con SOR, y eso es esperable: comparar el mismo método en ambos lados, y además el mejor de cada uno.

### 4. Modelo de Ising 2D

**Dificultad:** ★★ · **Código:** [ising_cpu.c](../code/proyectos/ising/ising_cpu.c)

Un imán como una red de espines {{< katex >}}s_i = \pm 1{{< /katex >}} que prefieren alinearse con sus vecinos, simulado con Monte Carlo (Metropolis). Alrededor de la temperatura crítica {{< katex >}}T_c = 2/\ln(1+\sqrt{2}) \approx 2.269{{< /katex >}} se pierde el orden: una transición de fase.

- **Validación:** la magnetización y la energía por espín se comparan con la CPU y con la solución exacta de Onsager (el código imprime ambas, con su error estándar). La GPU usa otros números aleatorios: se comparan los observables, no las configuraciones, y lejos de {{< katex >}}T_c{{< /katex >}}.
- **Ojo:** inicializar los generadores de cuRAND (`curand_init`) es caro: hacerlo una sola vez y reusarlos. Con el tablero de ajedrez que explica el código, el lado {{< katex >}}L{{< /katex >}} debe ser par.

### 5. Ecuación de Schrödinger 2D

**Dificultad:** ★★ · **Código:** [schrodinger_cpu.c](../code/proyectos/schrodinger/schrodinger_cpu.c)

Evolución de un paquete de ondas en un oscilador armónico, {{< katex >}}i\,\partial_t\psi = -\tfrac{1}{2}\nabla^2\psi + V\psi{{< /katex >}}, con el método *split-step* de Fourier: el potencial se aplica en el espacio de posiciones y la energía cinética en el de momentos, pasando de uno a otro con FFTs.

- **Validación:** la norma se conserva y el centro del paquete sigue la trayectoria clásica {{< katex >}}\langle x\rangle = x_0\cos(\omega t){{< /katex >}}; el código imprime ambas durante la simulación.
- **Ojo:** la transformada inversa de cuFFT no está normalizada: hay que dividir por {{< katex >}}N^2{{< /katex >}}.

### 6. Ray tracing

**Dificultad:** ★★ · **Código:** [ray-tracing_cpu.c](../code/proyectos/ray-tracing/ray-tracing_cpu.c)

Imágenes generadas siguiendo rayos de luz (*path tracing*) en una escena de esferas difusas, metálicas y de vidrio (de *Ray Tracing in One Weekend*, P. Shirley). La óptica es geométrica: reflexión, refracción con la ley de Snell {{< katex >}}n_1 \sin\theta_1 = n_2 \sin\theta_2{{< /katex >}} y reflexión total interna. Cada píxel promedia muchos rayos al azar (Monte Carlo).

- **Validación:** en la prueba del "horno blanco" (`./ray-tracing_cpu.x 1`) las esferas no absorben luz y todos los píxeles deben valer 1; en la escena normal, la imagen debe coincidir con la de la CPU.

### 7. Dinámica molecular de Lennard-Jones

**Dificultad:** ★★★ · **Código:** [lennard-jones_cpu.c](../code/proyectos/lennard-jones/lennard-jones_cpu.c)

Un gas o líquido de átomos que se repelen de cerca y se atraen de lejos, con el potencial {{< katex >}}V(r) = 4\varepsilon\left[(\sigma/r)^{12} - (\sigma/r)^{6}\right]{{< /katex >}}, integrado con *velocity Verlet*. A diferencia del ejemplo de N-cuerpos del curso, la interacción es de corto alcance: se corta en {{< katex >}}r_c = 2.5{{< /katex >}} y cada partícula sólo mira las de su celda y las 8 vecinas.

- **Validación:** la energía total se conserva (el código imprime su variación).
- **Ojo:** la dinámica es caótica, así que las posiciones de la GPU sólo coinciden con las de la CPU durante pocos pasos; para corridas largas se comparan la energía y la temperatura.
