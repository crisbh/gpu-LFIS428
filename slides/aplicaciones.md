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

Los códigos de esta clase están disponibles para descargar:

- N-cuerpos: [nbody.cu](../code/aplicaciones/nbody.cu)
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

Hay varios métodos para obtener la fuerza gravitacional. El más directo es sumar todas las fuerzas entre cada par de partículas (**sumación directa**):

- Requiere una expresión para la fuerza entre un par de partículas (ley universal de la gravedad de Newton).
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

La fuerza crece arbitrariamente cuando dos partículas están muy cercanas. Para galaxias (sistemas no colisionales) no es necesario resolver la dinámica entre dos partículas cercanas:

$$\mathbf{F}_i \approx G m_i \sum_{j}^N \frac{m_j\, \mathbf{r}_{ij}}{(r_{ij}^2 + \epsilon^2)^{3/2}}$$

El factor $\epsilon$ se llama *softening length*. En la práctica necesitamos la aceleración: $\mathbf{a}_i = \mathbf{F}_i / m_i$.

---

## **Simulaciones: N-cuerpos**

Para actualizar las velocidades y posiciones usamos un integrador *leapfrog-Verlet*:

$$v_{i+1/2} = v_{i-1/2} + a_i \, \Delta t$$
$$x_{i+1} = x_i + v_{i+1/2} \, \Delta t$$

---

## **Paralelización del cálculo**

- El algoritmo calcula cada elemento de una matriz $\mathbf{F}_{ij}$ de $N \times N$ de fuerzas entre pares de partículas.
- La fuerza total en la partícula $i$ es la suma de la fila $i$ de la matriz.
- Se puede calcular cada elemento independientemente (paralelismo $\mathcal{O}(N^2)$), pero requeriría memoria $\mathcal{O}(N^2)$ y estaría muy limitado por *bandwidth*.
- En esta implementación se utiliza un *tile*: una sub-matriz de $p \times p$ elementos.

---

## **Paralelización del cálculo**

![w:520px](images/aplicaciones/p_tile.png)

Usamos los datos de $2p$ partículas para calcular $p^2$ valores de la matriz $\mathbf{F}_{ij}$. Las filas se calculan en paralelo, las columnas en forma secuencial.

<p class="credit">Fuente: GPU Gems Vol. 3</p>

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

Ejemplo: [alineamiento_datos.c](../code/memoria/alineamiento_datos.c). Es C puro: se compila con `gcc` y corre en el *host*, sin GPU. **Predecir los dos tamaños antes de ejecutarlo.**

- En CUDA los tipos vectoriales (`float2`, `float4`) ya vienen alineados a $8$ y $16$ bytes.

<!-- NOTA — las respuestas: prueba1 {short s; int i; char c} ocupa 12 bytes (2 del short + 2 de relleno para que el int caiga en múltiplo de 4, + 4 del int, + 1 del char, + 3 de relleno final); prueba2 {int i; char c; short s} ocupa 8 (4 + 1 + 1 de relleno + 2). Los mismos tres campos, la mitad más de memoria por el orden. -->

<!-- La regla: cada campo va en una dirección múltiplo de su tamaño, y el struct completo se redondea al tamaño de su campo más grande. De ahí que float4 (16 bytes) quede alineado a 16 y un warp que lee float4 consecutivos pida sectores completos. -->

---

## **Cálculo de la fuerza entre un par de partículas**

- Se usa `float4` para los datos en la memoria del *device*, para ayudar con el acceso contiguo. El valor `w` corresponde a la masa.
- El uso de `float3` en la función es para variables locales, donde hay que reducir el espacio usado en registros.

<!-- NOTA — acá se cierra el círculo con las diapositivas anteriores: float4 es un layout AoS, no SoA. Y es la elección correcta según el criterio, porque cada thread usa las tres coordenadas Y la masa del MISMO cuerpo (mira bj.x, bj.y, bj.z y bj.w en bodyBodyInteraction), y el struct está alineado a 16 bytes. Es el caso de libro para AoS. -->

<!-- float3 en cambio ocupa 12 bytes y NO está alineado a 16, así que como dato en memoria global daría accesos desalineados; por eso se usa solo para variables locales, que viven en registros y no pasan por el sistema de memoria. -->

---

## **Cálculo de los tiles**

- Los datos de $p$ partículas se cargan de la memoria global a la memoria compartida para un bloque.
- Cada *thread* en el bloque calcula $p$ interacciones.
- El resultado del cálculo de un *tile* son $p$ aceleraciones parcialmente actualizadas.

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

Cada *thread* ejecuta esta función para las mismas $p$ partículas. El acceso a memoria compartida es de tipo *broadcast*: no hay conflictos de bancos.

---

## **Varios tiles en un bloque**

Cada bloque tiene $p$ *threads* y calcula $N/p$ *tiles* en forma secuencial. Antes de calcular un *tile*, cada *thread* copia los datos de una partícula a la memoria compartida; al terminar, hay sincronización.

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

---

## **Todas las partículas**

![w:520px](images/aplicaciones/full_calc_nbody.png)

Ejemplo: [nbody.cu](../code/aplicaciones/nbody.cu) (basado en un ejemplo del *CUDA Toolkit*).

<p class="credit">Fuente: GPU Gems Vol. 3</p>

---

# Fin Capítulo 5

## Próximo capítulo: librerías de CUDA y Python
