---
title: "Proyecto final"
description: "Proyecto final del curso: portar y optimizar un código de física en CUDA"
---

# Proyecto final

El proyecto final consiste en tomar un código de física que corre en la CPU, **portarlo a CUDA** y **optimizarlo**, midiendo y explicando cada mejora. Todos los temas son problemas donde la GPU tiene mucho que ofrecer: mallas grandes, muchas partículas o muchos eventos independientes.

## Modalidad

- Se trabaja en **parejas**. Excepcionalmente se puede trabajar de forma **individual** (se recomienda elegir un tema de dificultad ★).
- Cada grupo recibe **un tema** de la lista de abajo, asignado en la clase de inicio del proyecto.
- Cada tema incluye un **código de referencia en C** (un solo archivo, sin dependencias) que resuelve el problema en la CPU y valida su resultado.

## Etapas

1. **Entender el código de CPU.** Compilarlo, correrlo, leerlo y entender la física y el método numérico. Identificar qué partes concentran el tiempo de cómputo.
2. **Portar a CUDA (versión 1).** Escribir los kernels para las partes costosas. Los datos deben **quedarse en la GPU** durante toda la simulación: sólo se copian al inicio y al final (o cuando se necesita imprimir algo). Validar el resultado contra la versión de CPU.
3. **Optimizar (versiones 2, 3, …).** Aplicar al menos una técnica **vista en el curso**: memoria compartida, accesos coalescentes (AoS vs. SoA), memoria constante, reducciones paralelas y *warp primitives*, operaciones atómicas, *occupancy*, *streams*, librerías (cuFFT, cuRAND), `float` vs. `double`. Cada tema sugiere además técnicas **no vistas en clases** (marcadas así abajo), como la fusión de *kernels* (motivada por el [costo de lanzar un *kernel*](../slides/kernels.html)): son bienvenidas, pero hay que estudiarlas por cuenta propia. Cada versión debe seguir siendo correcta.
4. **Analizar el rendimiento.** Para cada versión, medir el tiempo y compararlo con la CPU en función del tamaño del problema. Usar los *profilers* disponibles en Colab, `ncu` (Nsight Compute) y `nvprof` (`nsys` normalmente no está disponible), y el modelo *roofline* para explicar **por qué** una versión es más rápida y qué tan lejos está del máximo del hardware. El ancho de banda efectivo se mide con `ncu --metrics dram__bytes.sum.per_second ./programa.x`, como en las clases.

> **Sobre el *speedup*:** el código de referencia usa un solo núcleo de CPU, así que el *speedup* respecto a él exagera la ventaja de la GPU. Por eso, además del *speedup*, hay que reportar el **ancho de banda efectivo** (GB/s) o los **FLOP/s** alcanzados y compararlos con los máximos de la T4 usados en las clases: {{< katex >}}\approx 300{{< /katex >}} GB/s de memoria y {{< katex >}}\approx 8.1{{< /katex >}} TFLOP/s en `float`. En `double` la T4 (CC 7.5) hace 1 operación por cada 32 de `float`, o sea {{< katex >}}\approx 8.1/32 \approx 0.25{{< /katex >}} TFLOP/s: los códigos de referencia usan `double`, y el *roofline* debe usar el techo que corresponde.

## Cronograma

| Cuándo | Qué |
|:---|:---|
| Semana 0 | Asignación de temas y parejas |
| ~Semana 2 | **Reunión 1:1** con el profesor (avance). Mínimo esperado: versión 1 funcionando y validada contra la CPU, y un primer perfil con `ncu` o `nvprof` |
| ~Semana 3.5 | **Entrega del código** y **presentación** de 10 minutos + preguntas |

La entrega incluye los códigos (.cu) y un `README` breve: cómo compilar y ejecutar cada versión, y una tabla con los tiempos medidos.

La presentación (10 minutos) debe cubrir: el problema físico y el método, cómo se paralelizó, la validación, las optimizaciones y su efecto medido, y el análisis de rendimiento (*roofline*, cuello de botella). Ambos integrantes presentan.

## Pauta de evaluación

| Criterio | Peso |
|:---|:---:|
| Reunión de avance (evaluación 3): versión 1 correcta, comprensión del código y del plan de optimización | —% |
| Correctitud y validación de todas las versiones | —% |
| Optimización: técnicas aplicadas y mejora medida | —% |
| Análisis de rendimiento: *profiling*, *roofline*, explicación del cuello de botella | —% |
| Física: interpretación de los resultados y figuras | —% |
| Claridad del código, del `README` y de la presentación | —% |

## Herramientas comunes

- [comparar.py](../code/proyectos/comparar.py): compara dos archivos de salida (p. ej. `salida_cpu.bin` y `salida_gpu.bin`) campo por campo e imprime la diferencia máxima absoluta y relativa. Acepta archivos en `float` o `double`.
- [graficar.py](../code/proyectos/graficar.py): genera imágenes PNG a partir de un archivo de salida (necesita `comparar.py` en la misma carpeta).

Todos los códigos de CPU escriben su resultado con la función `escribir_salida()`: una cabecera con `(nx, ny, ncampos, bytes por real)` seguida de los campos. La versión GPU debe escribir en el **mismo formato** para poder compararla (basta con copiar la función).

Los códigos de CPU se compilan con `gcc -O2 archivo.c -lm -o archivo.x` y también compilan como C++, así que el archivo se puede renombrar a `.cu` y empezar a portarlo con `nvcc -arch=sm_75 archivo.cu -o archivo.x`, igual que los ejemplos del curso. Todos usan `typedef double real;`: cambiarlo a `float` es una optimización válida, siempre que se muestre que la validación sigue siendo aceptable.

Como en las clases:

- Revisar los errores de cada llamada a CUDA con el macro `CHECK` de [common.h](../code/memoria/common.h).
- Medir el tiempo de los *kernels* con eventos de CUDA (`cudaEventRecord`, `cudaEventElapsedTime`). Si se usa el reloj del host, llamar antes a `cudaDeviceSynchronize()`: los lanzamientos de *kernels* son asíncronos.

**Diferencias esperables entre CPU y GPU:** los kernels que sólo leen y escriben arreglos (*stencils*, colisiones) deberían reproducir la CPU hasta el redondeo ({{< katex >}}\sim 10^{-15}{{< /katex >}} en `double`). Las **reducciones** (energía, residuo, promedios) suman en otro orden, así que difieren en las últimas cifras; como se vio en la clase de reducción, la suma en árbol de la GPU suele ser incluso **más** precisa que la suma secuencial de la CPU. En sistemas caóticos (Gray-Scott en tiempos largos, Lennard-Jones) esas diferencias crecen con el tiempo: ahí se comparan observables, no campos.

## Temas

| # | Tema | Dificultad |
|:---:|:---|:---:|
| 1 | [Ecuación del calor 2D](#1-ecuación-del-calor-2d) | ★ |
| 2 | [Ecuación de onda 2D](#2-ecuación-de-onda-2d) | ★ |
| 3 | [Reacción-difusión de Gray-Scott](#3-reacción-difusión-de-gray-scott) | ★ |
| 4 | [Ecuación de Poisson 2D](#4-ecuación-de-poisson-2d) | ★★ |
| 5 | [Modelo de Ising 2D](#5-modelo-de-ising-2d) | ★★ |
| 6 | [Ecuación de Schrödinger 2D](#6-ecuación-de-schrödinger-2d) | ★★ |
| 7 | [Fluidos con Lattice Boltzmann](#7-fluidos-con-lattice-boltzmann) | ★★★ |
| 8 | [Dinámica molecular de Lennard-Jones](#8-dinámica-molecular-de-lennard-jones) | ★★★ |
| 9 | [Ray tracing](#9-ray-tracing) | ★★ |

### 1. Ecuación del calor 2D

**Dificultad:** ★ · **Código:** [calor_cpu.c](../code/proyectos/calor/calor_cpu.c)

Difusión del calor en una placa cuadrada con borde a temperatura cero, {{< katex >}}u_t = \alpha \nabla^2 u{{< /katex >}}, con diferencias finitas explícitas (FTCS) y el laplaciano de 5 puntos.

- **Validación:** la condición inicial {{< katex >}}\sin(\pi x)\sin(\pi y){{< /katex >}} decae exactamente como {{< katex >}}e^{-2\pi^2\alpha t}{{< /katex >}}; el código imprime el error máximo respecto a esa solución.
- **Optimización mínima:** un kernel por paso con los datos en la GPU e intercambio de punteros (sin copias entre pasos).
- **Optimizaciones avanzadas:** *tiles* en memoria compartida, como en la transpuesta por bloques, pero cada *tile* necesita además un borde (*halo*) con los vecinos *(no visto en clases)*; varios pasos de tiempo por *kernel* (*temporal blocking*) *(no visto en clases)*; `float`; comparar el ancho de banda alcanzado con el máximo de la T4.
- **Extensiones:** fuentes de calor, conductividad variable, 3D.

### 2. Ecuación de onda 2D

**Dificultad:** ★ · **Código:** [onda_cpu.c](../code/proyectos/onda/onda_cpu.c)

Vibraciones de una membrana cuadrada fija en el borde, {{< katex >}}u_{tt} = c^2 \nabla^2 u{{< /katex >}}, con el esquema *leapfrog* (tres niveles de tiempo).

- **Validación:** el modo normal (1,1) oscila con {{< katex >}}\omega = \sqrt{2}\,\pi c{{< /katex >}}; por defecto se simula un período completo y se compara con la solución exacta. Además, el esquema conserva una energía discreta, cuya variación se imprime.
- **Optimización mínima:** kernel de un paso con rotación de tres punteros en la GPU.
- **Optimizaciones avanzadas:** memoria compartida; cálculo de la energía como reducción en la GPU; `float` (¿se sigue conservando la energía?).
- **Extensiones:** pulso inicial localizado, obstáculos (rendija doble), velocidad de propagación variable.

### 3. Reacción-difusión de Gray-Scott

**Dificultad:** ★ · **Código:** [reaccion-difusion_cpu.c](../code/proyectos/reaccion-difusion/reaccion-difusion_cpu.c)

Dos especies químicas que reaccionan y difunden forman patrones (manchas, laberintos, "mitosis"):

{{< katex display=true >}}
u_t = D_u \nabla^2 u - u v^2 + F(1-u), \qquad v_t = D_v \nabla^2 v + u v^2 - (F+k) v
{{< /katex >}}

- **Validación:** no hay solución exacta. Para pocos pasos, el campo de la GPU debe coincidir con `salida_cpu.bin` hasta el redondeo (usar `comparar.py`); para corridas largas, comparar los promedios {{< katex >}}\langle u\rangle, \langle v\rangle{{< /katex >}} y los patrones.
- **Optimización mínima:** un kernel que actualiza ambos campos con bordes periódicos.
- **Optimizaciones avanzadas:** memoria compartida para ambos campos; evitar el operador `%` en los bordes; `float`; generar imágenes durante la simulación sin frenar el cálculo (*streams*).
- **Extensiones:** explorar el plano {{< katex >}}(F, k){{< /katex >}} y clasificar los patrones; animaciones.

### 4. Ecuación de Poisson 2D

**Dificultad:** ★★ · **Código:** [poisson_cpu.c](../code/proyectos/poisson/poisson_cpu.c)

Potencial electrostático a partir de una densidad de carga, {{< katex >}}\nabla^2 u = f{{< /katex >}}, resuelto iterativamente con Jacobi, Gauss-Seidel rojo-negro o SOR rojo-negro.

- **Validación:** con la fuente usada, la solución exacta es {{< katex >}}\sin(\pi x)\sin(\pi y){{< /katex >}}; el error debe bajar ~4 veces al duplicar {{< katex >}}N{{< /katex >}} (orden {{< katex >}}h^2{{< /katex >}}).
- **Optimización mínima:** Jacobi en la GPU (es trivialmente paralelo) con el residuo calculado como **reducción** en la GPU.
- **Optimizaciones avanzadas:** rojo-negro / SOR en la GPU *(no visto en clases)*, el código de CPU lo explica: ¿por qué Gauss-Seidel normal no se puede paralelizar?; calcular el residuo con menos frecuencia; memoria compartida; comparar iteraciones vs. tiempo por iteración de cada método.
- **Extensiones:** condensador de placas paralelas u otras geometrías con condiciones de borde no nulas; *multigrid*.

### 5. Modelo de Ising 2D

**Dificultad:** ★★ · **Código:** [ising_cpu.c](../code/proyectos/ising/ising_cpu.c)

Transición de fase ferromagnética con Monte Carlo (algoritmo de Metropolis) en una red cuadrada, barriendo temperaturas alrededor de {{< katex >}}T_c = 2/\ln(1+\sqrt{2}) \approx 2.269{{< /katex >}}.

- **Validación:** la magnetización {{< katex >}}\langle|m|\rangle{{< /katex >}} y la energía por espín se comparan con la solución exacta de Onsager (el código la imprime al lado). Como la GPU usará otros números aleatorios, se comparan los observables, no las configuraciones.
- **Optimización mínima:** actualización en tablero de ajedrez *(no visto en clases)*, el código de CPU lo explica, con un generador de números aleatorios por *thread* (cuRAND en el *device*, como en [curand_device.cu](../code/librerias-python/curand_device.cu)); magnetización y energía como reducciones en la GPU.
- **Optimizaciones avanzadas:** espines como `char` (o bits); tabla de probabilidades en memoria constante; simular todas las temperaturas a la vez; memoria compartida.
- **Extensiones:** escalamiento de tamaño finito (cumulante de Binder para estimar {{< katex >}}T_c{{< /katex >}}), campo magnético externo.

### 6. Ecuación de Schrödinger 2D

**Dificultad:** ★★ · **Código:** [schrodinger_cpu.c](../code/proyectos/schrodinger/schrodinger_cpu.c)

Evolución temporal de un paquete de ondas en un oscilador armónico, {{< katex >}}i\,\partial_t\psi = -\tfrac{1}{2}\nabla^2\psi + V\psi{{< /katex >}}, con el método *split-step* de Fourier: la energía potencial se aplica en el espacio de posiciones y la cinética en el de momentos, pasando de uno a otro con FFTs.

- **Validación:** la norma se conserva y el centro del paquete sigue la trayectoria clásica {{< katex >}}\langle x\rangle = x_0\cos(\omega t){{< /katex >}}; ambas se imprimen durante la simulación.
- **Optimización mínima:** reemplazar la FFT propia del código por **cuFFT** y portar las multiplicaciones por fases a kernels.
- **Optimizaciones avanzadas:** fusionar *kernels* *(no visto en clases)*; calcular las fases al vuelo en vez de leerlas de memoria; calcular los observables con reducciones y sólo de vez en cuando; `float` (`cufftComplex`).
- **Extensiones:** efecto túnel a través de una barrera; rendija doble; potencial dependiente del tiempo.

### 7. Fluidos con Lattice Boltzmann

**Dificultad:** ★★★ · **Código:** [lattice-boltzmann_cpu.c](../code/proyectos/lattice-boltzmann/lattice-boltzmann_cpu.c)

Simulación de un fluido con el método de Lattice Boltzmann D2Q9: cada nodo guarda 9 distribuciones que colisionan (localmente) y se propagan a los vecinos. Dos modos: flujo en un canal (Poiseuille) y flujo alrededor de un cilindro (calle de vórtices de von Kármán).

- **Validación:** en el modo canal, el perfil de velocidades debe ser la parábola de Poiseuille (el código imprime el error). En el modo cilindro, comparar con `salida_cpu.bin`.
- **Optimización mínima:** kernels de colisión y propagación con los datos en la GPU.
- **Optimizaciones avanzadas:** pasar de AoS a **SoA**. Según el criterio de la clase de aplicaciones, AoS conviene cuando cada *thread* usa todos los campos de **su** elemento; en la propagación, en cambio, cada nodo lee una sola {{< katex >}}f_i{{< /katex >}} de cada uno de 9 vecinos distintos, que es el caso donde SoA gana. Fusionar colisión y propagación en un solo *kernel* *(no visto en clases)*; `float`. LBM está limitado por memoria: reportar el ancho de banda efectivo frente al máximo de la T4, y también MLUPS (millones de nodos actualizados por segundo, la métrica usual de LBM) *(no visto en clases)*.
- **Extensiones:** número de Reynolds y frecuencia de desprendimiento de vórtices (número de Strouhal); otros obstáculos; condiciones de entrada y salida.

### 8. Dinámica molecular de Lennard-Jones

**Dificultad:** ★★★ · **Código:** [lennard-jones_cpu.c](../code/proyectos/lennard-jones/lennard-jones_cpu.c)

Un gas/líquido de partículas que interactúan con el potencial de Lennard-Jones de corto alcance, integrado con *velocity Verlet*. A diferencia del ejemplo de N-cuerpos del curso, la interacción se corta en {{< katex >}}r_c = 2.5{{< /katex >}} y se usan **listas de celdas**: cada partícula sólo mira las de su celda y las 8 vecinas.

- **Validación:** la energía total se conserva (el código imprime su variación). Como la dinámica es caótica, las posiciones de la GPU sólo coinciden con la CPU durante pocos pasos; para corridas largas, comparar energía y temperatura.
- **Optimización mínima:** un *thread* por partícula para las fuerzas y la integración; construir las celdas en la GPU (las listas enlazadas necesitan **operaciones atómicas**).
- **Optimizaciones avanzadas:** reemplazar las listas enlazadas por un ordenamiento de partículas por celda (accesos coalescentes) *(no visto en clases)*; memoria compartida por celda; `float`.
- **Extensiones:** termostato, función de distribución radial {{< katex >}}g(r){{< /katex >}}, transición líquido-sólido al bajar la temperatura.

### 9. Ray tracing

**Dificultad:** ★★ · **Código:** [ray-tracing_cpu.c](../code/proyectos/ray-tracing/ray-tracing_cpu.c)

Generación de imágenes siguiendo rayos de luz (*path tracing*), basada en la escena final de *Ray Tracing in One Weekend* (P. Shirley): ~490 esferas difusas, metálicas y de vidrio. La óptica es geométrica: reflexión especular, refracción con la ley de Snell {{< katex >}}n_1 \sin\theta_1 = n_2 \sin\theta_2{{< /katex >}}, reflexión total interna y la reflectancia de Fresnel en la aproximación de Schlick. Cada píxel promedia muchos rayos aleatorios (Monte Carlo). El código está en C con *structs* y un `switch` por material (sin clases ni funciones virtuales, que complican el paso a la GPU).

- **Validación:** el modo 1 es la prueba del "horno blanco": esferas que no absorben luz bajo un cielo uniforme de intensidad 1. Por conservación de la energía todos los píxeles deben valer exactamente 1 (el código imprime el brillo mínimo y el promedio). En la escena del libro, cada píxel tiene su propio generador de números aleatorios, así que usando la misma función `aleatorio()` en la GPU la imagen debe coincidir con `salida_cpu.bin`: casi todos los píxeles quedan iguales y unos pocos difieren porque el redondeo cambia algún rebote, así que hay que mirar la diferencia **media** de `comparar.py` (con `float` en vez de `double` en la CPU, la media es {{< katex >}}\sim 10^{-3}{{< /katex >}}).
- **Optimización mínima:** un *thread* por píxel con las esferas en la memoria de la GPU; cada *thread* inicializa su generador con el índice del píxel.
- **Optimizaciones avanzadas:** esferas en **memoria constante** (todos los *threads* de un *warp* recorren las mismas esferas en el mismo orden, el caso ideal de la memoria constante) o en memoria compartida; `float` (la T4 hace 32 veces menos operaciones por segundo en `double`); medir la **divergencia de *warps*** (rayos vecinos rebotan en materiales distintos o terminan en distinto número de rebotes) y probar bloques 2D pequeños para que los rayos de un *warp* sean coherentes; cuRAND en vez del *hash* propio. Más allá del curso: una jerarquía de volúmenes envolventes (BVH) para no revisar todas las esferas *(no visto en clases)*.
- **Extensiones:** escenas más grandes, otros objetos (planos, triángulos), luces de área, dispersión cromática (índice de refracción dependiente de la longitud de onda: arcoíris en una esfera de agua).
