---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

# **Programación en GPUs**
## Aplicaciones

N-cuerpos

---

## **Códigos**

Códigos disponibles para descargar:

- N-cuerpos: [nbody-colab.cu](../code/aplicaciones/nbody-colab.cu) (Colab) · [nbody.cu](../code/aplicaciones/nbody.cu) · [nbody.h](../code/aplicaciones/nbody.h) · [tipsy.h](../code/aplicaciones/tipsy.h) (OpenGL)
- Estructuras de datos: [aos.cu](../code/memoria/aos.cu) · [soa.cu](../code/memoria/soa.cu) · [alineamiento_datos.c](../code/memoria/alineamiento_datos.c)

---

# Simulaciones: N-cuerpos

Ejemplo del CUDA Toolkit

---

## **Simulaciones: N-cuerpos**

- Una simulación de N-cuerpos trata de $N$ masas, representadas como partículas.
- La fuerza gravitacional entre estas masas se calcula en la simulación.
- De la fuerza obtenemos la **aceleración** que actúa en cada partícula.
- De la aceleración actualizamos la **velocidad** de cada partícula, y su **posición**.
- Repetimos este proceso cada *timestep* para evolucionar las partículas.

---

## **Simulaciones: N-cuerpos**

Hay varios métodos para obtener la fuerza gravitacional. El más directo es sumar todas las fuerzas entre cada par de partículas (**suma directa**):

- Requiere una expresión para la fuerza entre un par de partículas (ley de gravitación Universal de Newton).
- Desventaja: complejidad algorítmica $\mathcal{O}(N^2)$.

---

## **Simulaciones: N-cuerpos**

Fuerza entre $2$ cuerpos:

$$\mathbf{F}_{ij} = G \frac{m_i m_j}{r_{ij}^2}\,\frac{\mathbf{r}_{ij}}{r_{ij}}$$

donde $m_i$, $m_j$ son las masas $i$ y $j$, y $\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i$.

La fuerza total $\mathbf{F}_i$ en la partícula $i$:

$$\mathbf{F}_i = \sum_{j \neq i}^N \mathbf{F}_{ij} = G m_i \sum_{j \neq i}^N \frac{m_j\, \mathbf{r}_{ij}}{r_{ij}^3}$$

---

## **Simulaciones: N-cuerpos**

- La fuerza crece arbitrariamente cuando dos partículas están muy cerca, lo cual produce resultados no físicos.
- Para evitar esto, podemos introducir un término que la regulariza:

$$\mathbf{F}_i \approx G m_i \sum_{j}^N \frac{m_j\, \mathbf{r}_{ij}}{(r_{ij}^2 + \epsilon^2)^{3/2}}$$

El factor $\epsilon$ se llama *softening length*. En la práctica necesitamos la aceleración: $\mathbf{a}_i = \mathbf{F}_i / m_i$.

---

## **Simulaciones: N-cuerpos**

Para actualizar las velocidades y posiciones usamos un integrador *leapfrog-Verlet*:

$$v_{i+1/2} = v_{i-1/2} + a_i \, \Delta t$$
$$x_{i+1} = x_i + v_{i+1/2} \, \Delta t$$

<svg viewBox="0 0 1100 285" width="880" style="display:block;margin:0.4em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-lf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<text x="10" y="76" font-size="22" fill="#0288d1" font-weight="bold">x, a</text>
<text x="10" y="181" font-size="22" fill="#2e7d32" font-weight="bold">v</text>
<line x1="150" y1="250" x2="1060" y2="250" stroke="#444" stroke-width="1.5" marker-end="url(#fl-lf)"/>
<text x="1075" y="256" font-size="22" fill="#333">t</text>
<line x1="392" y1="70" x2="666" y2="70" stroke="#0288d1" stroke-width="2.5" marker-end="url(#fl-lf)"/>
<line x1="692" y1="70" x2="966" y2="70" stroke="#0288d1" stroke-width="2.5" marker-end="url(#fl-lf)"/>
<line x1="242" y1="175" x2="516" y2="175" stroke="#2e7d32" stroke-width="2.5" marker-end="url(#fl-lf)"/>
<line x1="542" y1="175" x2="816" y2="175" stroke="#2e7d32" stroke-width="2.5" marker-end="url(#fl-lf)"/>
<line x1="160" y1="175" x2="216" y2="175" stroke="#2e7d32" stroke-width="2.5" stroke-dasharray="4 4" marker-end="url(#fl-lf)"/>
<line x1="380" y1="82" x2="380" y2="169" stroke="#0288d1" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#fl-lf)"/>
<line x1="530" y1="163" x2="530" y2="76" stroke="#2e7d32" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#fl-lf)"/>
<line x1="680" y1="82" x2="680" y2="169" stroke="#0288d1" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#fl-lf)"/>
<line x1="830" y1="163" x2="830" y2="76" stroke="#2e7d32" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#fl-lf)"/>
<circle cx="380" cy="70" r="10" fill="#0288d1"/>
<text x="380" y="50" font-size="22" text-anchor="middle" fill="#222"><tspan font-weight="bold">x</tspan><tspan font-size="16" dy="5">i</tspan><tspan dy="-5">, </tspan><tspan font-weight="bold">a</tspan><tspan font-size="16" dy="5">i</tspan></text>
<line x1="380" y1="244" x2="380" y2="256" stroke="#444"/>
<text x="380" y="276" font-size="20" text-anchor="middle" fill="#333"><tspan>t</tspan><tspan font-size="16" dy="5">i</tspan></text>
<circle cx="680" cy="70" r="10" fill="#0288d1"/>
<text x="680" y="50" font-size="22" text-anchor="middle" fill="#222"><tspan font-weight="bold">x</tspan><tspan font-size="16" dy="5">i+1</tspan><tspan dy="-5">, </tspan><tspan font-weight="bold">a</tspan><tspan font-size="16" dy="5">i+1</tspan></text>
<line x1="680" y1="244" x2="680" y2="256" stroke="#444"/>
<text x="680" y="276" font-size="20" text-anchor="middle" fill="#333"><tspan>t</tspan><tspan font-size="16" dy="5">i+1</tspan></text>
<circle cx="980" cy="70" r="10" fill="#0288d1"/>
<text x="980" y="50" font-size="22" text-anchor="middle" fill="#222"><tspan font-weight="bold">x</tspan><tspan font-size="16" dy="5">i+2</tspan><tspan dy="-5">, </tspan><tspan font-weight="bold">a</tspan><tspan font-size="16" dy="5">i+2</tspan></text>
<line x1="980" y1="244" x2="980" y2="256" stroke="#444"/>
<text x="980" y="276" font-size="20" text-anchor="middle" fill="#333"><tspan>t</tspan><tspan font-size="16" dy="5">i+2</tspan></text>
<circle cx="230" cy="175" r="10" fill="#2e7d32"/>
<text x="230" y="207" font-size="22" text-anchor="middle" fill="#222"><tspan font-weight="bold">v</tspan><tspan font-size="16" dy="5">i−½</tspan></text>
<line x1="230" y1="244" x2="230" y2="256" stroke="#444"/>
<text x="230" y="276" font-size="20" text-anchor="middle" fill="#333"><tspan>t</tspan><tspan font-size="16" dy="5">i−½</tspan></text>
<circle cx="530" cy="175" r="10" fill="#2e7d32"/>
<text x="530" y="207" font-size="22" text-anchor="middle" fill="#222"><tspan font-weight="bold">v</tspan><tspan font-size="16" dy="5">i+½</tspan></text>
<line x1="530" y1="244" x2="530" y2="256" stroke="#444"/>
<text x="530" y="276" font-size="20" text-anchor="middle" fill="#333"><tspan>t</tspan><tspan font-size="16" dy="5">i+½</tspan></text>
<circle cx="830" cy="175" r="10" fill="#2e7d32"/>
<text x="830" y="207" font-size="22" text-anchor="middle" fill="#222"><tspan font-weight="bold">v</tspan><tspan font-size="16" dy="5">i+3⁄2</tspan></text>
<line x1="830" y1="244" x2="830" y2="256" stroke="#444"/>
<text x="830" y="276" font-size="20" text-anchor="middle" fill="#333"><tspan>t</tspan><tspan font-size="16" dy="5">i+3⁄2</tspan></text>
<text x="390" y="135" font-size="19" fill="#0288d1">+ <tspan font-weight="bold">a</tspan><tspan font-size="14" dy="4">i</tspan><tspan dy="-4"> Δt</tspan></text>
<text x="540" y="115" font-size="19" fill="#2e7d32">+ <tspan font-weight="bold">v</tspan><tspan font-size="14" dy="4">i+½</tspan><tspan dy="-4"> Δt</tspan></text>
</svg>

<!-- NOTA — leer el diagrama: posiciones y aceleraciones viven en los tiempos enteros, velocidades en los medios; cada cantidad "salta por encima" de la otra (de ahí leapfrog). La flecha punteada azul dice que la aceleración a_i (calculada con las posiciones x_i: es el paso O(N^2), el kernel) actualiza la velocidad de i−1/2 a i+1/2; la verde, que esa velocidad nueva lleva la posición de x_i a x_{i+1}. Una sola evaluación de fuerzas por paso, segundo orden en Δt y simpléctico: la energía no deriva en simulaciones largas, a diferencia de Euler. Para arrancar hace falta v_{1/2} = v_0 + a_0 Δt/2 (la flecha punteada de la izquierda). -->

---

## **La matriz de fuerzas $\mathbf{F}_{ij}$**

<svg viewBox="0 0 1000 362" width="820" style="display:block;margin:0.2em auto 0" xmlns="http://www.w3.org/2000/svg" font-family="Arial, sans-serif">
<defs><marker id="fl-fij" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444"/></marker></defs>
<text x="404" y="20" font-size="18" text-anchor="middle" fill="#0288d1" font-weight="bold">j (cuerpo que ejerce la fuerza)</text>
<text x="40" y="194" font-size="18" text-anchor="middle" fill="#0288d1" font-weight="bold" transform="rotate(-90 40 194)">i (cuerpo que la recibe)</text>
<text x="209.0" y="52" font-size="16" text-anchor="middle" fill="#333">1</text>
<text x="156" y="90.0" font-size="16" text-anchor="end" fill="#333">1</text>
<text x="287.0" y="52" font-size="16" text-anchor="middle" fill="#333">2</text>
<text x="156" y="134.0" font-size="16" text-anchor="end" fill="#333">2</text>
<text x="365.0" y="52" font-size="16" text-anchor="middle" fill="#333">3</text>
<text x="156" y="178.0" font-size="16" text-anchor="end" fill="#333">3</text>
<text x="443.0" y="52" font-size="16" text-anchor="middle" fill="#333">4</text>
<text x="156" y="222.0" font-size="16" text-anchor="end" fill="#333">4</text>
<text x="521.0" y="52" font-size="16" text-anchor="middle" fill="#333">⋯</text>
<text x="156" y="266.0" font-size="16" text-anchor="end" fill="#333">⋮</text>
<text x="599.0" y="52" font-size="16" text-anchor="middle" fill="#333">N</text>
<text x="156" y="310.0" font-size="16" text-anchor="end" fill="#333">N</text>
<rect x="170" y="62" width="78" height="44" fill="#e0e0e0" stroke="#555"/>
<text x="209.0" y="90.0" font-size="17" text-anchor="middle" fill="#222">0</text>
<rect x="248" y="62" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="287.0" y="90.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">12</tspan></text>
<rect x="326" y="62" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="365.0" y="90.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">13</tspan></text>
<rect x="404" y="62" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="443.0" y="90.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">14</tspan></text>
<rect x="482" y="62" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="521.0" y="90.0" font-size="17" text-anchor="middle" fill="#222">⋯</text>
<rect x="560" y="62" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="599.0" y="90.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">1N</tspan></text>
<rect x="170" y="106" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="209.0" y="134.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">21</tspan></text>
<rect x="248" y="106" width="78" height="44" fill="#e0e0e0" stroke="#555"/>
<text x="287.0" y="134.0" font-size="17" text-anchor="middle" fill="#222">0</text>
<rect x="326" y="106" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="365.0" y="134.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">23</tspan></text>
<rect x="404" y="106" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="443.0" y="134.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">24</tspan></text>
<rect x="482" y="106" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="521.0" y="134.0" font-size="17" text-anchor="middle" fill="#222">⋯</text>
<rect x="560" y="106" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="599.0" y="134.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">2N</tspan></text>
<rect x="170" y="150" width="78" height="44" fill="#a5d6a7" stroke="#555"/>
<text x="209.0" y="178.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">31</tspan></text>
<rect x="248" y="150" width="78" height="44" fill="#a5d6a7" stroke="#555"/>
<text x="287.0" y="178.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">32</tspan></text>
<rect x="326" y="150" width="78" height="44" fill="#e0e0e0" stroke="#555"/>
<text x="365.0" y="178.0" font-size="17" text-anchor="middle" fill="#222">0</text>
<rect x="404" y="150" width="78" height="44" fill="#a5d6a7" stroke="#555"/>
<text x="443.0" y="178.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">34</tspan></text>
<rect x="482" y="150" width="78" height="44" fill="#a5d6a7" stroke="#555"/>
<text x="521.0" y="178.0" font-size="17" text-anchor="middle" fill="#222">⋯</text>
<rect x="560" y="150" width="78" height="44" fill="#a5d6a7" stroke="#555"/>
<text x="599.0" y="178.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">3N</tspan></text>
<rect x="170" y="194" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="209.0" y="222.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">41</tspan></text>
<rect x="248" y="194" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="287.0" y="222.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">42</tspan></text>
<rect x="326" y="194" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="365.0" y="222.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">43</tspan></text>
<rect x="404" y="194" width="78" height="44" fill="#e0e0e0" stroke="#555"/>
<text x="443.0" y="222.0" font-size="17" text-anchor="middle" fill="#222">0</text>
<rect x="482" y="194" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="521.0" y="222.0" font-size="17" text-anchor="middle" fill="#222">⋯</text>
<rect x="560" y="194" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="599.0" y="222.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">4N</tspan></text>
<rect x="170" y="238" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="209.0" y="266.0" font-size="17" text-anchor="middle" fill="#222">⋮</text>
<rect x="248" y="238" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="287.0" y="266.0" font-size="17" text-anchor="middle" fill="#222">⋮</text>
<rect x="326" y="238" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="365.0" y="266.0" font-size="17" text-anchor="middle" fill="#222">⋮</text>
<rect x="404" y="238" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="443.0" y="266.0" font-size="17" text-anchor="middle" fill="#222">⋮</text>
<rect x="482" y="238" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="521.0" y="266.0" font-size="17" text-anchor="middle" fill="#222">⋱</text>
<rect x="560" y="238" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="599.0" y="266.0" font-size="17" text-anchor="middle" fill="#222">⋮</text>
<rect x="170" y="282" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="209.0" y="310.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">N1</tspan></text>
<rect x="248" y="282" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="287.0" y="310.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">N2</tspan></text>
<rect x="326" y="282" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="365.0" y="310.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">N3</tspan></text>
<rect x="404" y="282" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="443.0" y="310.0" font-size="17" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="12" dy="5">N4</tspan></text>
<rect x="482" y="282" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="521.0" y="310.0" font-size="17" text-anchor="middle" fill="#222">⋯</text>
<rect x="560" y="282" width="78" height="44" fill="#fff" stroke="#555"/>
<text x="599.0" y="310.0" font-size="17" text-anchor="middle" fill="#222">0</text>
<line x1="646" y1="172.0" x2="718" y2="172.0" stroke="#444" stroke-width="2" marker-end="url(#fl-fij)"/>
<text x="682" y="162.0" font-size="16" text-anchor="middle" fill="#333">Σ<tspan font-size="11" dy="4">j</tspan></text>
<rect x="726" y="148.0" width="150" height="48" rx="3" fill="#a5d6a7" stroke="#555"/>
<text x="801" y="179.0" font-size="19" text-anchor="middle" fill="#222"><tspan font-weight="bold">F</tspan><tspan font-size="13" dy="5">3</tspan><tspan dy="-5"> (total)</tspan></text>
<text x="404" y="354" font-size="15" text-anchor="middle" fill="#555">diagonal en gris: un cuerpo no se atrae a sí mismo</text>
</svg>

- Fila $i$: las fuerzas que los $N$ cuerpos ejercen sobre el cuerpo $i$. Su suma es $\mathbf{F}_i$.
- Cada elemento es un vector ($3$ componentes) que depende solo de los cuerpos $i$ y $j$: son independientes entre sí.

<!-- NOTA — la matriz nunca se guarda en memoria: es la forma de ver el trabajo. Son N^2 interacciones y cada fila se reduce a un solo vector F_i, así que basta acumular la suma mientras se recorre la fila (es lo que hace bodyBodyInteraction con ai). La matriz es antisimétrica, F_ji = -F_ij (tercera ley de Newton), lo que permitiría hacer la mitad de los cálculos; esta implementación no lo aprovecha porque escribir en dos filas a la vez obligaría a usar atómicas. En el código, con el softening, el término j = i no hace falta excluirlo: r_ii = 0 y aporta cero sin dividir por cero. -->

---

## **Paralelización del cálculo**

- El algoritmo calcula cada elemento de una matriz $\mathbf{F}_{ij}$ de $N \times N$ de fuerzas entre pares de partículas.
- La fuerza total en la partícula $i$ es la suma de la fila $i$ de la matriz.
- Se puede calcular cada elemento independientemente (paralelismo $\mathcal{O}(N^2)$), pero requeriría memoria $\mathcal{O}(N^2)$ y estaría muy limitado por *bandwidth*.

En esta implementación se utiliza un *tile*: una sub-matriz de $p \times p$ elementos.

---

## **Paralelización del cálculo: el mapa completo**

La matriz $\mathbf{F}_{ij}$ repartida en *tiles* de $p \times p$:

![w:540px](images/aplicaciones/full_calc_nbody.png)
<p class="credit">Fuente: GPU Gems Vol. 3</p>

<!-- NOTA — esta es una vista previa: no explicar todavía threads, bloques ni las cargas desde memoria global. Basta con que reconozcan la matriz de la diapositiva anterior cortada en cuadrados de p × p, cada uno un tile. Las etiquetas (p threads, N/p bloques, p pasos entre cargas) se leen cuando la figura vuelve en "Todas las partículas". -->

---

## **Paralelización del cálculo**

![w:720px](images/aplicaciones/p_tile.png)
<p class="credit">Fuente: GPU Gems Vol. 3</p>

- En cada *tile*, usamos los datos de $2p$ partículas para calcular $p^2$ valores de la matriz $\mathbf{F}_{ij}$.
- Las filas se calculan en paralelo, las columnas en forma secuencial.


---

## **Cálculo de la fuerza entre un par de partículas**

```cuda
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
  float3 r;
  // r_ij  [3 FLOPS]
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
  // invDistCube = 1/distSqr^(3/2)  [4 FLOPS]
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f/sqrtf(distSixth);
  // s = m_j * invDistCube  [1 FLOP]
  float s = bj.w * invDistCube;
  // a_i = a_i + s * r_ij  [6 FLOPS]
  ai.x += r.x * s;  ai.y += r.y * s;  ai.z += r.z * s;
  return ai;
}
```

---

## **Cálculo de la fuerza entre un par de partículas**

Algunos comentarios sobre el *kernel* anterior:
- Se usa `float4` para los datos en la memoria del *device*, para ayudar con el acceso contiguo. El valor `w` corresponde a la masa.
- El uso de `float3` en la función es para variables locales, donde hay que reducir el espacio usado en registros.
- Los argumentos usan un `struct` que representa un cuerpo (*body*, b).

- Existen dos opciones para usar $N$ structs (cuerpos).
  - AoS: arreglos de estructuras.
  - SoA: estructuras de arreglos.

<!-- NOTA — acá se cierra el círculo con las diapositivas anteriores: float4 es un layout AoS, no SoA. Y es la elección correcta según el criterio, porque cada thread usa las tres coordenadas Y la masa del MISMO cuerpo (mira bj.x, bj.y, bj.z y bj.w en bodyBodyInteraction), y el struct está alineado a 16 bytes. Es el caso de libro para AoS. -->

<!-- float3 en cambio ocupa 12 bytes y NO está alineado a 16, así que como dato en memoria global daría accesos desalineados; por eso se usa solo para variables locales, que viven en registros y no pasan por el sistema de memoria. -->


---

## **Estructuras de datos: AoS vs. SoA**

Un `struct` agrupa campos bajo un solo nombre. Se acceden con `.`:

```cuda
struct Particula { float x; float y; };
Particula p;  p.x = 1.0f;  p.y = 2.0f;
```

Para $N$ partículas hay dos formas de organizar los mismos datos:

```cuda
Particula particulas[N];                        // AoS: particulas[i].x
struct Particulas { float x[N]; float y[N]; };  // SoA: particulas.x[i]
```

- **AoS** (arreglo de estructuras) intercala `x y x y ...`.
- **SoA** (estructura de arreglos) separa `x x ... y y ...`.

<!-- NOTA — es la misma pregunta de la clase de memoria, pero ahora con una decisión concreta que justificar: por qué el código de N-cuerpos guarda las posiciones como float4. Para el warp, el layout decide si el acceso es contiguo o no. -->

---

## **Opciones para estructuras de datos**

![w:720px](images/memoria/figure_4_22.png)
<p class="credit">Fuente: <em>Professional CUDA C Programming</em></p>

- **SoA**: cada *thread* lee **un campo** de muchos elementos → el *warp* accede a datos contiguos.
- **AoS**: cada *thread* usa **todos los campos** de su elemento; funciona bien si el *struct* está alineado ($8$ o $16$ bytes, como `float4`).

Ejemplo: [aos.cu](../code/memoria/aos.cu) y [soa.cu](../code/memoria/soa.cu): mismo cálculo, solo cambia el *layout*. **No imprimen nada**; hay que medirlos con `ncu`.

<!-- Las métricas a pedir son las mismas de la clase de memoria: smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct para la eficiencia por instrucción, y dram__bytes_read.sum para el tráfico real. En Colab sirven ncu y nvprof, nunca nsys. Los programas son 2^20 elementos de 8 campos int, 256 threads por bloque. -->

<!-- NOTA — la aritmética, que es lo que hay que contar: sizeof(Datos_AOS) = 8 campos int = 32 bytes, o sea EXACTAMENTE un sector. En AoS el warp pide datos[i].r en direcciones separadas por 32 bytes, así que toca 32 sectores distintos: 1024 bytes movidos por 128 útiles, y la métrica marca 12.5%. En SoA cada campo es un arreglo aparte, el warp lee 32 int contiguos = 128 bytes = 4 sectores, y la métrica marca 100%. -->

<!-- PERO — y este es el punto — esos 32 sectores de AoS contienen los 8 campos de esos 32 elementos, así que las 6 cargas siguientes pegan en L1. El tráfico real a DRAM es 1024 bytes por warp en los dos casos: el mismo. O sea, este kernel es justamente el caso FAVORABLE a AoS, porque usa todos los campos y la línea que trae se aprovecha entera. La métrica de eficiencia y el tráfico real no coinciden, y la diferencia la explica el cache: es la misma lección de la clase de memoria global, aplicada a estructuras. -->

<!-- Variante para proponer: si el kernel usara UN SOLO campo, AoS movería ~8 veces más datos que SoA y ahí sí la diferencia saltaría en el tiempo, sin necesidad de profiler. -->

<!-- Aviso práctico: los dos programas lanzan el kernel sobre memoria sin inicializar, así que los resultados numéricos son basura (puede haber división por cero, que en la GPU no aborta). No importa: lo único que se mide acá es el tráfico de memoria. -->

---

## **Alineamiento de estructuras**

- La organización de los elementos en una estructura tiene consecuencias para el uso de la memoria.
- Los mismos campos, en distinto orden, ocupan distinto espacio.

Ejemplo: [alineamiento_datos.c](../code/memoria/alineamiento_datos.c). Es C puro: se compila con `gcc` y corre en el *host*, sin GPU. 
<!-- **Predecir los dos tamaños antes de ejecutarlo.** -->

- En CUDA los tipos vectoriales (`float2`, `float4`) ya vienen alineados a $8$ y $16$ bytes.

<!-- NOTA — las respuestas: prueba1 {short s; int i; char c} ocupa 12 bytes (2 del short + 2 de relleno para que el int caiga en múltiplo de 4, + 4 del int, + 1 del char, + 3 de relleno final); prueba2 {int i; char c; short s} ocupa 8 (4 + 1 + 1 de relleno + 2). Los mismos tres campos, la mitad más de memoria por el orden. -->

<!-- La regla: cada campo va en una dirección múltiplo de su tamaño, y el struct completo se redondea al tamaño de su campo más grande. De ahí que float4 (16 bytes) quede alineado a 16 y un warp que lee float4 consecutivos pida sectores completos. -->

---

## **Cálculo de los tiles**

- Los datos de $p$ partículas se cargan de la memoria global a la memoria compartida para un bloque.
- Cada *thread* en el bloque calcula $p$ interacciones.
- El resultado del cálculo de un *tile* son $p$ aceleraciones parcialmente actualizadas.

![w:720px](images/aplicaciones/p_tile.png)
<p class="credit">Fuente: GPU Gems Vol. 3</p>

---

## **Cálculo de los tiles**

```cuda
__device__ float3 tile_calculation(float4 myPosition, float3 accel)
{
  int i;
  extern __shared__ float4[] shPosition;
  for (i = 0; i < blockDim.x; i++){
    accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
  }
  return accel;
}
```

- Cada *thread* ejecuta esta función para las mismas $p$ partículas. 
- El acceso a memoria compartida es de tipo *broadcast*: no hay conflictos de bancos.

---

## **Varios tiles en un bloque**

- Cada bloque tiene $p$ *threads* y calcula $N/p$ *tiles* en forma secuencial. 
- Antes de calcular un *tile*, cada *thread* copia los datos de una partícula a la memoria compartida
  - Al terminar, hay sincronización a nivel de bloque.

![w:460px](images/aplicaciones/multiple_tiles.png)

<p class="credit">Fuente: GPU Gems Vol. 3</p>

---

## **Cálculo de todas las fuerzas**

```cuda
__global__ void calculate_forces(void *devX, void *devA)
{
  extern __shared__ float4[] shPosition;
  float4 *globalX = (float4 *)devX;
  float4 *globalA = (float4 *)devA;
  float4 myPosition;
  int i, tile;
  float3 acc = {0.0f, 0.0f, 0.0f};
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  myPosition = globalX[gtid];
  for (i = 0, tile = 0; i < N; i += p, tile++) {
    int idx = tile * blockDim.x + threadIdx.x;
    shPosition[threadIdx.x] = globalX[idx];
    __syncthreads();
    acc = tile_calculation(myPosition, acc);
    __syncthreads();
  }
  float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
  globalA[gtid] = acc4;  // guardar para la integración temporal
}
```

---

## **Todas las partículas**

- Con $p$ *threads* por bloque y $N$ partículas, necesitamos $N/p$ bloques en la grilla.
- Esto resulta en $N$ *threads* calculando $N$ interacciones ($N^2$ en total).

![w:520px](images/aplicaciones/full_calc_nbody.png)
<p class="credit">Fuente: GPU Gems Vol. 3</p>

---

## **Todas las partículas: el ejemplo**

[nbody.cu](../code/aplicaciones/nbody.cu) + [nbody.h](../code/aplicaciones/nbody.h) + [tipsy.h](../code/aplicaciones/tipsy.h) (basado en el *CUDA Toolkit*):

- `computeBodyAccel` es `calculate_forces`; `integrateBodies` le agrega el paso de tiempo.
- Buscar $p$, el número de *tiles* y el tercer argumento de `<<< >>>`.
- Abre una ventana OpenGL: **no corre en Colab**. En un PC con GPU y `freeglut`:

```bash
nvcc nbody.cu -o nbody.x -lGL -lGLU -lglut
./nbody.x    # necesita galaxy_20K.bin en la carpeta
```

<!-- NOTA — integrateBodies hace v += a dt y luego x += v_nueva dt: es el leapfrog de la diapositiva 8 si se lee v como v_{i+1/2} (sin el medio paso inicial). blockSize = 256 está fijo en main, numTiles = N/256 y sharedMemSize = 256 * 4 floats (un float4 por cuerpo del tile). galaxy_20K.bin (20224 cuerpos, formato tipsy) NO está en el repositorio: viene con el ejemplo nbody de cuda-samples. Sin pantalla el programa muere en glutInit. Ratón: arrastrar rota, Shift + arrastrar desplaza, Ctrl + arrastrar acerca. -->

---

## **N-cuerpos en Colab**

[nbody-colab.cu](../code/aplicaciones/nbody-colab.cu): el mismo *kernel*, sin OpenGL. Genera $N$ cuerpos al azar, verifica un paso contra el *host* e imprime el tiempo por paso, las interacciones por segundo y los GFLOP/s.

```bash
nvcc -arch=sm_75 nbody-colab.cu -o nbody-colab.x
./nbody-colab.x 16384 256      # N  p
```

1. Con $N = 16384$, probar $p = 64, 128, 256, 512$. ¿Cuál es más rápido? ¿Cuántos bloques hay en cada caso?
2. Duplicar $N$ dos veces. ¿Crece el tiempo por paso como $N^2$?
3. ¿Qué fracción del *peak* de la T4 ($\approx 8.1$ TFLOP/s) se alcanza?

<!-- RESPUESTAS esperadas, SIN MEDIR (el programa no se ha compilado ni corrido en la T4; TODO: correrlo y poner los números acá). 1: la T4 tiene 40 SM; con N = 16384 y p = 512 hay solo 32 bloques, así que quedan SM sin trabajo, y con p = 256 hay 64 bloques, menos de dos por SM. Bloques más chicos reparten mejor el trabajo, pero cada tile es más corto y hay más __syncthreads por interacción: lo esperable es que p = 128 o 256 gane, y que 512 pierda. 2: con N chico el GPU no está lleno (16384 threads contra 40 x 1024 = 40960 que caben), así que al duplicar N el tiempo crece menos que 4 veces; cuando el GPU se llena, crece como N^2. 3: GPU Gems cuenta 20 FLOP por interacción, pero rsqrtf es una sola instrucción de la SFU, así que la cifra es una convención; fracciones del orden de 20-50 % del peak serían razonables. -->

---

# Fin Capítulo 5

<!-- ## Próximo capítulo: librerías de CUDA y Python -->
