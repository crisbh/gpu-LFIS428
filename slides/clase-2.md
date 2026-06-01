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

# **La Gravedad Newtoniana**
## Clase 2

---

## **Plan de la Clase**
- Fuerza y potencial gravitacional.
- Algunas analogías con la electroestática.
- La ecuación de Poisson para el potencial gravitacional y algunas soluciones.
<!-- - Gravedad vs aceleración. -->
<!-- - La ecuación de marea y conexión con Relatividad General. -->

---

## **La fuerza gravitacional de Newton**

Según Newton, la **fuerza** de gravedad hace que los objetos sigan trayectorias curvas. Esta fuerza está dada por:

$$ F_N = m_G g $$

donde $m_G$ es la masa del cuerpo sometido a la fuerza, y $g$ el **campo gravitacional** creado por todas las demás masas.

Es posible describir esta fuerza $g$ mediante un **campo escalar**.

---

## **El Potencial Gravitacional**

El campo gravitacional es un campo conservativo, por lo que se puede escribir como el gradiente de un potencial escalar:

$$ g = -\nabla \Phi $$

O en notación de índices:

$$ g_i = -\partial_i \Phi $$

Así, el **potencial gravitacional** debido una masa puntual $M$ es $\Phi(r) = -\frac{GM}{r}$.

---

## **Masa gravitacional vs. Masa inercial**

La ecuación de movimiento en mecánica newtoniana es:

$$ m_I a = F_N = -m_G \nabla \Phi $$

Si la masa gravitacional y la masa inercial son iguales ($m_G = m_I$), la aceleración de caída libre es:

$$ a = -\nabla \Phi $$

Esto explica la observación de Galileo de que todos los cuerpos caen con la misma aceleración (en ausencia del roce del aire).

---

## **Ecuación de Poisson**

A partir de la 2da ley de Newton es posible derivar la **ecuación de Poisson** para el campo gravitacional:

$$ \nabla^2 \Phi = 4\pi G \rho $$

donde $\rho(x)$ es la densidad de masa. 

Esta es la **ecuación de campo** para la gravedad en la teoría de Newton. En este sentido, es el equivalente Newtoniano a las Ecuaciones de Einstein de la Relatividad General.

---

## **Ecuación de Poisson**

- Notar que la ecuación anterior ya la ha encontrado en el curso de electromagnetismo.
- En tal caso, $\Phi$ juega el rol del **potencial eléctrico**:
$$
V = \frac{kQ}{r}
$$
- Por lo tanto, en el caso gravitacional también se cumple la **Ley de Gauss**!
  - *La integral del campo eléctrico sobre una superficie cerrada (flujo) es proporcional a la carga encerrada* 

---

## **Gravedad vs Electromagnetismo**
|  | **Gravedad** | **Electrostática** |
|-------------|-------------|--------------------|
| **Fuerza** | $F = \frac{GMm}{r^2}$ | $F = \frac{k q_1 q_2}{r^2}$|
| **Campo** | $g = \frac{GM}{r^2}$ | $E = \frac{kQ}{r^2}$ |
| **Potencial** | $\Phi = -\frac{GM}{r}$ | $V = \frac{kQ}{r}$ |
  | **Ecuación de Poisson** | $\nabla^2 \Phi = 4\pi G \rho$ | $\nabla^2 V = -\frac{\rho}{\epsilon_0}$ |
  | **Ley de Gauss** | $\oint_S \mathbf{g} \cdot dS = -4\pi G M_{\text{enc}}$ | $\oint_S \mathbf{E} \cdot dS = \frac{Q_{\text{enc}}}{\epsilon_0}$ |

---

## **Principio de Superposición**

- Notar que la ecuación de Poisson es una ecuación en derivadas parciales (EDP) de tipo **lineal**.
- Luego, para un sistema de múltiples masas, el potencial total es la suma de los potenciales individuales:

$$ \Phi(\mathbf{r}) = -G \sum_i \frac{m_i}{|\mathbf{r} - \mathbf{r}_i|} $$

Qué $\rho$ se asocia a una masa puntual, y a un conjunto de ellas?

---

## **Ejemplo: Resolver la ecuación de Poisson para una masa puntual**

Resuelva la ecuación de Poisson 

$$
\nabla^2 \Phi = 4\pi G \rho
$$

para encontrar el campo producido por una masa puntual.

---

## **Ejemplo: Resolver la ecuación de Poisson para una masa puntual**

Para una masa puntual $M$ ubicada en el origen, la densidad de masa está dada por una función Delta de Dirac
$$
\rho(\mathbf{r}) = M \delta^3(\mathbf{r})
$$

Debido a la simetría esférica del problema, la ecuación de Poisson en coordinadas esféricas toma la forma:
$$
\nabla^2 \Phi = \frac{1}{r^2} \frac{d}{dr} \left( r^2 \frac{d\Phi}{dr} \right) = 4\pi G M \delta^3(\mathbf{r})
$$

---

## **Ejemplo: Resolver la ecuación de Poisson para una masa puntual**

Para $r \neq 0$ (cualquier punto fuera de la masa), tenemos:
$$
\frac{1}{r^2} \frac{d}{dr} \left( r^2 \frac{d\Phi}{dr} \right) = 0
$$

Integrando dos veces, encontramos:

$$
\Phi(r) = -\frac{C_1}{r} + C_2
$$

---

## **Ejemplo: Resolver la ecuación de Poisson para una masa puntual**



Ahora aplicamos las condiciones de borde sobre la solución.
1. Físicamente, esperamos una solución *asintóticamente plana*:
   $$
   \Phi(r\to\infty)\to0  \implies C_2=0
   $$

2. Para determinar $C_1$, podemos aplicar la *Ley de Gauss*:
$$
\oint_S \mathbf{g} \cdot dS =-\oint_S \mathbf{\nabla}\Phi \cdot dS = -4\pi G M_{\text{enc}}
$$
---

## **Ejemplo: Resolver la ecuación de Poisson para una masa puntual**

Integrando sobre una esfera de radio $\epsilon$, con $dS = r^2 d\Omega$, encontramos que:
$$
\left. \frac{d\Phi}{dr} \right|_{r=\epsilon} 4\pi r^2 = 4\pi G M
$$

Por lo que $C_1 = G M$. La solución final es entonces la esperada:

$$
\Phi(r) = -\frac{GM}{r}.
$$

