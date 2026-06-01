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

# **Las Ecuaciones de Einstein**
## Clase 8

---

## **Plan de la Clase**
- Repaso última clase.
- El tensor de Riemann.
- Las Ecuaciones de Einstein.
- La solución de Schwarzschild.

---

## **Resumen de la última clase**

- **Geodesicas**: Trajectorias de partículas "libres" en una espaciotiempo curvo.
  - Son la generalización del concepto de **línea recta**, i.e. la distancia más corta entre dos puntos.
- **Derivada covariante** $\nabla_\mu$: Generaliza el concepto de derivada parcial para tensores.
  - Se reduce a la derivada parcial en el caso de escalares.

---

## **La derivada covariante**

  - En espacios curvos, la noción de derivada tradicional (parcial) se generaliza a la **derivada covariante**:

$$
\nabla_\mu v^\alpha = \partial_\mu v^\alpha + \Gamma^\alpha_{\mu\nu} v^\nu
$$

$$
\nabla_\mu v_\alpha = \partial_\mu v_\alpha - \Gamma^\nu_{\mu\alpha} v_\nu
$$

---

## **La conexión: símbolos de Christoffel**

- Los **símbolos de Christoffel**, o **conexión**, están dados por:

$$
\Gamma^\alpha_{\mu\nu} = \frac{1}{2} g^{\alpha\beta} 
\left(\partial_\mu g_{\nu\beta} + \partial_\nu g_{\mu\beta} - \partial_\beta g_{\mu\nu}\right)
$$

- Es un objeto simétrico con dos índices, pero realmente **no es un tensor**.

---

## **La ecuación geodésica**

- En RG, la gravedad **ya no es una fuerza** , i.e. las partículas son "libres" (a menos que incluyamos una fuerza **real**).
$$
a^\mu=\frac{Dv^\mu}{D\tau}=u^\mu \nabla_\nu u^\nu = 0
$$ 

- Explícitamente, la ecuación geodésica toma la forma:

$$
\boxed{
\frac{d^2 x^\alpha}{d\tau^2} + \Gamma^\alpha_{\mu\nu} \frac{dx^\mu}{d\tau}\frac{dx^\nu}{d\tau} = 0
}
$$

---

## **La ecuación de marea en RG**

- La ecuación anterior es análoga a la **2da ley de Newton**.
- Qué pasa si intentamos buscar el equivalente a la ecuación de marea de la gravedad Newtoniana?

La idea es aplicar la ecuación geodésica a dos partículas cercanas, expandir en Taylor, y escribir una ecuación para el vector de separación entre ambas.

---

## **La ecuación de marea en RG**

- Al repetir el razonamiento que nos llevó a la ecuación de marea: 
  - En vez de $F=-\nabla{\Phi}$, ahora aparece $\Gamma\sim\partial g_{\mu\nu}$.
  - Luego, donde antes apareció $\Phi$ ahora aparece $g_{\mu\nu}$.
  - Los términos $\sim\partial\partial\Phi$ ahora aparece $\sim\partial\partial g_{\mu\nu}$.

---

## **La ecuación de marea en RG**

- La versión de RG de la ecuación de marea es la **ecuación de desviación geodésica**:

$$
\boxed{
\frac{d^2\Delta x^\alpha}{d\tau^2} = - R^\alpha_{\;\beta\gamma\delta}u^\beta u^\gamma \Delta x^\delta
}
$$

donde el **tensor de Riemann** está dado por:
$$
R^\alpha_{\;\beta\gamma\delta} = 
\partial_\gamma \Gamma^\alpha_{\;\delta\beta} - 
\partial_\delta \Gamma^\alpha_{\;\gamma\beta} + 
\Gamma^\alpha_{\;\gamma\mu}\Gamma^\mu_{\;\delta\beta} -
\Gamma^\alpha_{\;\delta\mu}\Gamma^\mu_{\;\gamma\beta}
$$

---

## **El Tensor de Riemann**

- El tensor de Riemann contiene información sobre la **curvatura** del espaciotiempo.
  - Un espaciotiempo es plano si y solo si el tensor de Riemann es cero en todas partes.

- Este tensor nos permite identificar un espaciotiempo curvo **independientemente de que coordenadas utilizamos**.


---

## **El Tensor de Riemann: ejemplo**

- Imaginemos dos personas en la superficie de la Tierra caminando desde el ecuador hacia el norte: al inicio sus caminos son paralelos.
- Sin embargo, al acercarse al polo norte, sus caminos convergen.
  - Se juntan aunque no haya fuerzas actuando lateralmente!
  - La superficie curva (una esfera) produce esta convergencia.

La curvatura (tensor de Riemann) genera esta convergencia.

---

## **Tensor y escalar de Ricci**

Existen dos contracciones importantes del tensor de Riemann:

- **Tensor de Ricci** : $R_{\mu\nu}=R^\alpha_{\;\mu\alpha\nu}$.
- **Escalar de Ricci** : $R = g^{\mu\nu}R_{\mu\nu} = R^{\alpha}_{\;\alpha}$ 
  - (i.e., la traza del tensor de Ricci).

- Así como el tensor de Riemann, ambos objetos también contienen información sobre la curvatura del espaciotiempo.

---

## **Las ecuaciones de campo de Einstein** 

Las **Ecuaciones de Campo de Einstein** (EdE) conectan la geometría del espaciotiempo al contenido de materia y energía:

$$
\boxed{
G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R\,g_{\mu\nu} = 8\pi G\,T_{\mu\nu}
}
$$

- **Geometría**: El tensor de Einstein $G_{\mu\nu}$ (curvatura).
- **Materia/Energía**: Tensor de energía-momento $T_{\mu\nu}$ (distribución de masa y energía).

---

## **RG y teoría de Newton**

| Cantidad | Gravedad Newtoniana | Relatividad General |
|---|---|---|
| Campo fundamental | Potencial $\Phi$ | Métrica $g_{\mu\nu}$ |
| Ecuación de movimiento | $F^i=m a^i$ | $a^\mu=0$ (geodésica)|
| Tensor de marea | $R_{ij}=\partial_i \partial_j \Phi$ | Riemann $R^\rho_{\sigma\mu\nu}$ |
| Ecuación de campo | $\nabla^2 \Phi = 4\pi G \rho$ | $G_{\mu\nu} = 8\pi G T_{\mu\nu}$ |


---

## **Las ecuaciones de campo de Einstein** 

En palabras del físico John Archibald Wheeler:

> **El espaciotiempo le dice a la materia como moverse; la materia le dice al espaciotiempo como curvarse.**  

- La geometría rige el movimiento a través de las geodésicas.
- La materia y energía infuencian la curvatura a través de las EdE.

---

## **Las ecuaciones de campo de Einstein** 

- Las EdE son **EDP no lineales**, donde buscamos como solución las componentes de $g_{\mu\nu}$.
- Existen pocas soluciones analíticas, que típicamente requieren simplificaciones:
  - Simetrías (e.g. simetría esférica)
  - Vacío ($T_{\mu\nu}=0$)

**La solución de Schwarzschild** es una de las soluciones principales, la cual se obtiene con la simplificaciones de: vacío, esférica, estacionaria.

---

## **La solución de Schwarzschild (1916)** 

**Karl Schwarzschild** encontró la primera solución exacta de la RG tras su publicación en 1915 por Einstein.

- Simplificaciones:
  - Simetría esférica
  - Espaciotiempo estacionario ($\partial_t g_{\mu\nu}=0$)
  - Espaciotiempo vacío fuera de masa puntual ($T_{\mu\nu}=0$)

---

## **La solución de Schwarzschild (1916)** 

El elemento de línea (métrica) de Schwarzschild es:
$$
ds^2 = -\left(1-\frac{2GM}{r}\right)dt^2 + \left(1-\frac{2GM}{r}\right)^{-1}dr^2 + r^2d\Omega^2
$$

donde:
- $d\Omega^2=d\theta^2+\sin^2\theta\,d\phi^2$ (parte angular).
- $M$: parámetro de masa (masa-energía).

