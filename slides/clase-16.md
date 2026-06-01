---
marp: true
paginate: true
math: katex
html: true
style: |
  section {
    font-size: 2.2em !important;
    font-family: 'Arial', sans-serif;
    overflow: hidden;
  }
  img {
    display: block;
    margin: auto;
    width: 60%;
    max-width: 100%;
  }
  .video-container {
    position: relative;
    padding-bottom: 56.25%; /* 16:9 aspect ratio */
    height: 0;
    overflow: hidden;
    max-width: 100%;
    background: black;
  }
  .video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }
---

# **Foliación del espaciotiempo y el formalismo 3+1**
## Clase 16

---

## **Objetivos de la Clase**
- El concepto de foliaciones del espaciotiempo.
- La descomposición 3+1 y su uso en relatividad numérica.
- Ecuaciones de restricción y evolución en la formulación 3+1.

---

# **Motivación para el formalismo 3+1**
- La RG unifica el concepto de espacio y tiempo para describir la geometría del **espacio-tiempo** mediante su métrica $g_{\mu\nu}$.
- Sin embargo, en simulaciones numéricas buscamos evolucionar **un sistema en el tiempo**.
  - Por ejemplo, como orbitan dos agujeros negros en el tiempo.
- En cierto sentido, esto requiere 'dar un paso atrás' y plantear RG en términos de una descripción de espacio y tiempo 'separados'.

---

# **Motivación para el formalismo 3+1**

- Notar que esto no quiere decir volver al régimen Newtoniano del tiempo absoluto.
  - Debemos hacerlo dentro de la RG.
- Esto puede hacerse mediante el **formalismo 3+1** para reescribir las EdE en una forma conveniente.
  - Dividiremos el espacio-tiempo en **rebanadas** (hiper-superficies espaciales) parametrizadas por un tiempo $t$.
  - Describiremos como se conectan dichas rebanadas (slices).

---

# **Motivación para el formalismo 3+1**

**Una analogía:**
- En mecánica newtoniana, el estado de un sistema se describe completamente mediante **posiciones y velocidades** $\forall t$.
- En RG, podemos definir el "estado" del espacio-tiempo de manera similar.
  - Las variables básicas serán una **métrica espacial** y su respectiva 'velocidad' (la **curvatura extrínseca**).
- Esto equivale a una **descripción Hamiltoniana** de la RG.


---

# **Foliación del Espaciotiempo**

![spacetime-foliation](images/spacetime-foliation.png)

- Un espaciotiempo se puede dividir en **hipersuperficies espaciales** $\Sigma_t$ etiquetadas por un parámetro temporal $t$.
- Definimos un vector normal a estas hipersuperficies, $n^a$.
  - En una esfera, este vector sería análogo a $\hat{r}$ ($R$=const.).

---

# **Conectando dos slices vecinas**

![spacetime-foliation](images/adm-pitagoras.png)


---
# **Descomposición 3+1**
- La métrica del espaciotiempo $g_{ab}$ se descompone en:
  - **Métrica espacial inducida**: $\gamma_{ij} = g_{ij} + n_i n_j$.
    - Esta mide distancias en cada hipersuperficie **espacial**.
  - **Lapso** ($\alpha$) y **shift** ($\beta^i$):

    $$
    ds^2 = -\alpha^2 dt^2 + \gamma_{ij} (dx^i + \beta^i dt)(dx^j + \beta^j dt)
    $$

- Notablemente, esta descomposición permite reescribir las ecuaciones de Einstein en términos de dos tipos de ecuaciones: **evolución y restricción**.

---
# **El lapso y el vector de shift**

- El lapso ($\alpha$) determina qué tan rápido transcurre el tiempo en cada punto del espacio.
  - $\alpha=1$: el tiempo transcurre al mismo ritmo en que Minkowski.
  - $\alpha<1$: el tiempo transcurre más lento que en Minkowski.
- El vector de shift ($\beta$) re-etiqueta las coordenadas entre dos instantes de tiempo.
  - Típicamente (pero no siempre) se escoge un sistema de coordenadas donde $\beta^i=0$.

---

# **La Curvatura Extrínseca**

- La **métrica inducida** $\gamma_{ij}$ juega el papel de la 'posición' en
la descripción Hamiltoniana.
- La **curvatura extrínseca** $K_{ij}$ será el análogo de la variable 'velocidad' en este formalismo. Se define como:

    $$
    K_{ij} = -\frac{1}{2} \mathcal{L}_n \gamma_{ij}
    $$

  donde $\mathcal{L}_n$ es la *derivada de Lie* en la dirección normal.
- Esta mide cómo la hipersuperficie $\Sigma_t$ se curva dentro del espacio-tiempo (como una esfera dentro del espacio 3D).


---

# **La Curvatura Extrínseca**

- Para un espacio-tiempo plano, $K_{ij} = 0$
- Para un espacio-tiempo curvo, $K_{ij} \neq 0$

---

# **Ecuaciones de RG en 3+1**
- Con este formalismo, las EdE se dividen en dos categorías: ecuaciones de restricción, y ecuaciones de evolución:
  - **Restricción**: no contienen derivadas temporales de $\gamma_{ij}$ o $K_{ij}$.
    - Son análogas a la Ec. de Poisson de la gravedad Newtoniana. Se satisfacen en cada instante de tiempo.
  - **Evolución**: sí contienen derivadas temporales de $\gamma_{ij}$ o $K_{ij}$.
    - Son análogas a la Ec. de onda para las ondas gravitacionales.

---

# **Ecuaciones de Restricción**
  - **Restricción Hamiltoniana**:

    $$
    \mathcal{H} \equiv R + K^2 - K_{ij}K^{ij} - 16\pi\rho = 0
    $$

  - **Restricción de Momento**:

    $$
    \mathcal{M}^i \equiv D_j (K^{ij} - \gamma^{ij} K) - 8\pi S^i = 0
    $$

  donde $R$ y $D_i$ son el escalar de curvatura de Ricci y la derivada
  covariante asociadas a la métrica $\gamma_{ij}$, $\rho$ la densidad, y $S^i$ el momentum (provienen de $T_{\mu\nu}$).

---

# **Ecuaciones de Evolución**
- Por otra parte, la evolución de $\gamma_{ij}$ y $K_{ij}$ está dada por:
  - **Evolución de la métrica espacial**:

    $$
    \partial_t \gamma_{ij} = -2\alpha K_{ij} + D_i\beta_j + D_j\beta_i
    $$

  - **Evolución de la curvatura extrínseca**:

    $$
    \partial_t K_{ij} = -D_i D_j \alpha + \alpha (R_{ij} + K K_{ij} - 2 K_{ik} K^k_j) + \mathcal{L}_\beta K_{ij} - 8\pi \alpha (S_{ij} - \frac{1}{2} \gamma_{ij} (S - \rho))
    $$

---

# **Relatividad Numérica en 3+1**

- En RN se suelen resolver las ecuaciones en la forma 3+1.
- Ecuaciones de evolución se modifican un poco para tener mejores propiedades numéricas.
  - Ejemplo: las formulaciones de BSSN, Z4, CCZ4, ...
- Las Ecuaciones de Restricción **no suelen resolverse durante una simulación**. Más bien:
  - Se resuelven en $t=0 \implies$ una condición inicial válida.
  - En $t>0$, se reemplazan en estas las soluciones obtenidas y se evalua su precisión (numéricamente, $\mathcal{H}\neq0$ y $\mathcal{M}^i\neq0$).


<!-- --- -->
<!---->
<!-- # **Comparación con Otras Teorías de Campos** -->
<!-- | Teoría               | Variables Dinámicas         | Variables de Calibre    | Ecuaciones de Evolución | Ecuaciones de Restricción | -->
<!-- |----------------------|---------------------------|------------------------|-------------------------|--------------------------| -->
<!-- | **Campo Escalar**    | \( \psi, \kappa \)        | —                      | 2                       | 0                        | -->
<!-- | **Electrodinámica**  | \( A_i, E_i \)            | \( \phi \)             | 6                       | 1                        | -->
<!-- | **Relatividad General** | \( \gamma_{ij}, K_{ij} \) | \( \alpha, \beta^i \) | 12                      | 4                        | -->
<!---->
<!-- > **Interpretación:** -->
<!-- > - La Relatividad General tiene mayor número de ecuaciones y grados de libertad debido a la curvatura del espacio-tiempo. -->
<!-- > - A diferencia de un campo escalar, donde las condiciones iniciales pueden elegirse libremente, en Relatividad General deben satisfacer **restricciones**. -->
<!---->
---

# **Ejercicios Propuestos**
- Mostrar que, en el límite Newtoniano ($v/c\ll1$, $\Phi/c^2\ll1$), la restricción Hamiltoniana $\mathcal{H}=0$ se reduce a la ecuación de Poisson (puede usar cálculos simbólicos).
  - *Hint*: es necesario *linealizar* las expresiones, dada las condiciones del límite Newtoniano.

