---
marp: true
paginate: true
math: katex
html: true
theme: curso
---

# **Programación en GPUs**
## Proyecto final

Temas

---

## **El proyecto**

- Tomar un código de física que corre en la CPU, **portarlo a CUDA** y **optimizarlo**.
- Cada tema trae su código de referencia en C.

<!-- NOTA — los números de tema son los de la página del proyecto; los temas 3 (Gray-Scott) y 8 (Lattice Boltzmann) no se presentan en esta clase. -->

---

## **Tema 1: Ecuación del calor 2D**

<div class="lado-a-lado">
<div style="flex:0 0 38%">

![w:420px](images/proyectos/calor.png)

</div>
<div>

$$u_t = \alpha\,\nabla^2 u$$

- **Difusión**: el calor fluye de lo caliente a lo frío y la placa se enfría.
- La temperatura de cada punto depende de la de sus vecinos.

</div>
</div>

---

## **Tema 2: Ecuación de onda 2D**

<div class="lado-a-lado">
<div style="flex:0 0 38%">

![w:420px](images/proyectos/onda.png)

</div>
<div>

$$u_{tt} = c^2\,\nabla^2 u$$

- Una **membrana** que vibra, fija en el borde (un tambor).
- Las perturbaciones se propagan con velocidad $c$.

</div>
</div>

---

## **Tema 4: Ecuación de Poisson 2D**

<div class="lado-a-lado">
<div style="flex:0 0 38%">

![w:420px](images/proyectos/poisson.png)

</div>
<div>

$$\nabla^2 u = f$$

- El **potencial electrostático** $u$ creado por una densidad de carga $f$.
- No hay evolución en el tiempo: se busca el equilibrio, iterando.

</div>
</div>

---

## **Tema 5: Modelo de Ising 2D**

<div class="lado-a-lado">
<div style="flex:0 0 38%">

![w:420px](images/proyectos/ising.png)

</div>
<div>

$$E = -J \sum_{\langle i,j\rangle} s_i\, s_j, \quad s_i = \pm 1$$

- Un **imán**: espines que prefieren alinearse con sus vecinos.
- A la temperatura crítica $T_c \approx 2.269$ el orden se pierde: una **transición de fase**.

</div>
</div>

---

## **Tema 6: Ecuación de Schrödinger 2D**

<div class="lado-a-lado">
<div style="flex:0 0 38%">

![w:420px](images/proyectos/schrodinger.png)

</div>
<div>

$$i\,\partial_t\psi = -\tfrac{1}{2}\nabla^2\psi + V\psi$$

- Un **paquete de ondas** cuántico en un oscilador armónico.
- Su centro oscila como una partícula clásica.

</div>
</div>

---

## **Tema 7: Ray tracing**

<div class="lado-a-lado">
<div style="flex:0 0 38%">

![w:420px](images/proyectos/ray-tracing.png)

</div>
<div>

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

- **Óptica geométrica**: se siguen rayos de luz que se reflejan y refractan en esferas.
- Cada píxel promedia muchos rayos al azar.

</div>
</div>

---

## **Tema 9: Lennard-Jones**

<div class="lado-a-lado">
<div style="flex:0 0 38%">

![w:420px](images/proyectos/lennard-jones.png)

</div>
<div>

$$V(r) = 4\varepsilon\left[\left(\tfrac{\sigma}{r}\right)^{12} - \left(\tfrac{\sigma}{r}\right)^{6}\right]$$

- **Dinámica molecular**: átomos de un gas o líquido que se repelen de cerca y se atraen de lejos.
- Como N-cuerpos, pero la interacción es de corto alcance.

</div>
</div>

---

## **¿Cómo se asignan?**

- Se trabaja en **parejas** (excepcionalmente solo).
- Cada grupo indica **tres temas** en orden de preferencia; no se repiten temas entre grupos.
- La asignación busca que nadie quede fuera de sus tres preferencias; los empates se sortean.

Todos los detalles: [página del proyecto](../homeworks/).
