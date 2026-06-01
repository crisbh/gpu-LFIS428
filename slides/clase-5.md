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

# **Notación Tensorial**
## Clase 5

---

## **Plan de la Clase**
- Formalismo tensorial.
- El tensor métrico.

---

## **¿Por qué usar tensores?**
- Los tensores son fundamentales en la Relatividad General (RG) y en diversas ramas de la física teórica.
- **Pricipio de Relatividad de Einstein**: las leyes de la Física deben ser las mismas en todos los sistemas de coordenadas.
     - Las coordenadas son simplemente etiquetas para eventos. No tienen sentido físico.
     - Los tensores permiten escribir ecuaciones en forma **covariante** (independiente de las coordenadas), e.g.,:
  $$
  G_{\mu\nu} = 8\pi G T_{\mu\nu}
  $$

---

## **Los Tensores**

- Un **tensor** es un objeto matemático que generaliza **vectores y matrices**.
- Formalmente, se caracterizan por su comportamiento bajo *transformaciones de coordenadas*.
- Se describen usando índices, por ejemplo:
  - Escalar: sin índice, $\Phi$
  - Vector: un índice, $v^a$ o $v_a$
  - Matriz o tensor de rango-2: dos índices, $T^{ab}$, $T_{ab}$, $T^a_{b}$, $\dots$

---

## **Tipos de Tensores**

Los tensores pueden clasificarse de acuerdo a su *rango*:
- **Escalar ($T^0$):** Un número que no cambia con las coordenadas.
- **Vector ($T^1$):** Un objeto con una única fila o columna de componentes.
- **Tensor de rango 2 ($T^2$):** Una matriz que obedece transformaciones tensoriales.
- **Tensor de rango superior ($T^n$):** Generalización a múltiples índices.

---

## **Índices Covariantes y Contravariantes**

- **Índices superiores** $v^a$: denominados **contravariantes**. Representan componentes de un vector.
- **Índices inferiores** $v_a$: denominados **covariantes**. Representan componentes de un covector.

El tipo de índice determina la forma en que los objetos se transforman bajo cambios de coordenadas.

---

## **Notación de Einstein (Suma Implícita)**

- En Física, es muy común utilizar la notación de Einstein:
  - **Índice repetido implica suma sobre ese índice.**
- Por ejemplo, para un vector $v^a$ y un *covector* $w_a$:

$$
v^a w_a \equiv \sum_{a=1}^{n} v^a w_a
$$

- Ejemplo en 3 dimensiones:

$$
v^a w_a = v^1 w_1 + v^2 w_2 + v^3 w_3
$$

<!-- Esta notación simplifica expresiones tensoriales y evita símbolos de suma explícitos. -->


---

## **Operaciones Tensoriales**

**1. Contracción**:

- Reduce el rango del tensor en 2, sumando índices superior e inferior iguales:

$$
T^a_{\ a} = \sum_a T^a_{\ a}
$$

Ejemplo:

$$
R^a_{\ bad} = R_{bd} \quad\text{(Tensor de Riemann $\to$ Tensor de Ricci)}
$$

---

## **Operaciones Tensoriales**

**2. Producto tensorial (producto exterior)**:

- Combina tensores para aumentar su rango:

$$
A^a B^b = T^{ab}
$$

**3. Producto escalar (producto interno)**:

- Producto de un vector y covector:

$$
v^a w_a
$$

---

## **Ejemplo: Contracción y Producto Escalar**

- Dados $v^a = (1,2,3)$, $w_a = (4,5,6)$:

Producto escalar:

$$
v^a w_a = (1)(4) + (2)(5) + (3)(6) = 32
$$

Contracción del tensor $T^a_{\ b}$:

$$
T^a_{\ b} = 
\begin{pmatrix}
1 & 2 & 3\\
0 & 4 & 5\\
7 & 8 & 9
\end{pmatrix}
$$

---

## **Ejemplo: Contracción y Producto Escalar**

Contracción $T^a_{\ a}$:

$$
T^a_{\ a} = T^1_{\ 1} + T^2_{\ 2} + T^3_{\ 3} = 1 + 4 + 9 = 14
$$

Notar que este tipo de contracción equivale a tomar **la traza** del tensor.

---

## **Ejemplo: Producto Tensorial**
Dado los vectores:

$$ A^\mu = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad B^\nu = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} $$

El producto tensorial es:

$$ C^{\mu\nu} = A^\mu B^\nu $$

---
## **Ejemplo: Producto Tensorial**

$$
C^{\mu\nu} = \begin{bmatrix} 1 \times 4 & 1 \times 5 & 1 \times 6 \\
2 \times 4 & 2 \times 5 & 2 \times 6 \\
3 \times 4 & 3 \times 5 & 3 \times 6 \end{bmatrix}
= \begin{bmatrix} 4 & 5 & 6 \\
8 & 10 & 12 \\
12 & 15 & 18 \end{bmatrix}
$$

---

## **Simetría y Antisimetría**

- Existen ciertos tensores que satisfacen propiedades de simetría:

**Tensor simétrico**:

$$
S_{ab} = S_{ba}
$$

**Tensor antisimétrico**:

$$
A_{ab} = -A_{ba}, \quad A_{aa} = 0
$$

---

## **Los tensores y la geometría**

- Los tensores (escalares, campos vectoriales, campos tensoriales) son en general funciones de las coordenadas:
   - Sus valores pueden variar de punto a punto en el espacio y tiempo, e.g., $\Phi$, $\vec{g}$, $\vec{E}$, $\vec{x}$, $\vec{v}$ ...
- Por lo tanto, los tensores no son objetos abstractos que "viven en el vacío": están definidos sobre un *espacio(-tiempo)*.

   Además de tensores que representan "cantidades físicas", la geometría del espacio(-tiempo) tiene sus propios tensores.

---

## **El tensor métrico**

- Existe un tensor muy especial, el **tensor métrico**: $g_{ab}$.
- Dado un tensor métrico $g_{ab}$, podemos bajar índices:

$$
v_a = g_{ab} v^b
$$

- Con el tensor métrico inverso $g^{ab}$, podemos subir índices:

$$
w^a = g^{ab} w_b
$$

El tensor métrico es **simétrico** , y actúa como un **mapa entre vectores y covectores**.

---

## **Resumen: Operaciones Tensoriales**

| Operación | Notación | Resultado |
|-----------|----------|---------|
| Producto tensorial | $A^a B^b$  | aumenta rango |
| Contracción |  $T^a_{\ a}$ | reduce rango en 2 |
| Producto escalar | $v^a w_a$ | resultado escalar |
| Subir/Bajar índices | $v_a = g_{ab} v^b$ | índice cambia posición |

