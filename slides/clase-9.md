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

# **Ec. de Einstein y agujeros negros**
## Clase 9

---

## **Plan de la Clase**
- Repaso última clase (antes del paro).
- Propiedades de la solución de Schwarzschild.
- Ejercicios prácticos:
  - Cálculo de los símbolos de Christoffel.
  - Ecuación geodésica.
  - Tutoriales de `Einsteinpy`.

---

## **La conexión: símbolos de Christoffel**

- Los **símbolos de Christoffel**, o **conexión**, están dados por:

$$
\Gamma^\alpha_{\mu\nu} = \frac{1}{2} g^{\alpha\beta} 
\left(\partial_\mu g_{\nu\beta} + \partial_\nu g_{\mu\beta} - \partial_\beta g_{\mu\nu}\right)
$$

- Contiene primeras derivadas de la métrica.
- Permite construir piezas fundamentales como la derivada covariante, o el Tensor de Riemann.

---

## **El Tensor de Riemann**

- El tensor de Riemann contiene información sobre la **curvatura** del espaciotiempo:

$$
R^\alpha_{\;\beta\gamma\delta} = 
\partial_\gamma \Gamma^\alpha_{\;\delta\beta} - 
\partial_\delta \Gamma^\alpha_{\;\gamma\beta} + 
\Gamma^\alpha_{\;\gamma\mu}\Gamma^\mu_{\;\delta\beta} -
\Gamma^\alpha_{\;\delta\mu}\Gamma^\mu_{\;\gamma\beta}
$$

- Este tensor nos permite identificar un espaciotiempo curvo **independientemente de que coordenadas utilizamos**.

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

## **La solución de Schwarzschild** 

La métrica de Schwarzschild es una solución a las ecuaciones de Einstein para una partícula puntual.
- Vacío (masa-energía solo en el origen)
- Simetría esférica
- Métrica estacionaria

---

## **La solución de Schwarzschild** 

El elemento de línea (métrica) de Schwarzschild es:
$$
ds^2 = -\left(1-\frac{2GM}{r}\right)dt^2 + \left(1-\frac{2GM}{r}\right)^{-1}dr^2 + r^2d\Omega^2
$$

donde:
- $d\Omega^2=d\theta^2+\sin^2\theta\,d\phi^2$ (parte angular).
- $M$: parámetro de masa.

---

## **La solución de Schwarzschild** 

- Notar que se recupera la métrica de Minkowski al alejarse mucho de la partícula  ($r\gg 2GM$):
  $$
  ds^2 \approx -dt^2+dr^2+r^2 d\Omega^2
  $$

- Por otra parte: qué pasará cuando la métrica se indetermina?

---

## **El horizonte de eventos** 

- **Radio de Schwarzschild**:
$$
r_s = \frac{2GM}{c^2}
$$
- **Horizonte de eventos** en $r=r_s$: la luz no puede escapar.
- La métrica diverge en $r=r_s$. Sin embargo, esta es una **singularidad de coordenadas**, i.e. desaparece si elegimos otro sistema de coordenadas conveniente.
  - La divergencia en $r=0$ **sí es física**: la curvatura diverge.

---

<!-- ## **El horizonte de eventos**  -->
<!---->
<!-- - El área de una esfera de radio $r=r_s$ es: -->
<!-- $$ -->
<!-- A=4\pi r_s^2=16\pi G^2M^2 -->
<!-- $$ -->
<!-- - Esto define el "tamaño" del agujero negro. -->
<!--   - Es decir, la región de "no retorno". -->
<!---->
<!-- --- -->

## **El horizonte de eventos** 

- Fuera del horizonte ($r>r_s$): los observadores pueden permanecer estáticos.
- En el horizonte ($r=r_s$): las trayectorias de la luz no puede escapar.
- Dentro del horizonte ($r<r_s$): cualquier trayectoria lleva a la singularidad ($r=0$).

---

## **Taller (próxima clase)** 

Para el taller utilizaremos la métrica de Schwarzschild. En particular, nos interesará

1. Calcular los símbolos de Christoffel.
2. Resolver la ecuación geodésica.
3. Explorar la librería `EinsteinPy`

