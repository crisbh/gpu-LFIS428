# Roofline Model — Guion del Experimento en Vivo

## Objetivo

Una simulación física de la ejecución de un kernel para introducir el modelo roofline. Dos estudiantes representan un "GPU" y su subsistema de memoria. Variando la operación que hace el procesador, la clase experimenta los tres regímenes: **memory-bound**, **ridge point**, y **compute-bound**. Después se construye el diagrama roofline en el pizarrón a partir de lo que la clase acaba de observar.

**Duración total:** ~35 min (15–20 min demo + 15 min construcción del roofline).

**Ubicación en el curso:** al inicio de la Sesión 6 (Modelo Roofline), **antes** de presentar el modelo formalmente.

---

## Materiales

- **3 hojas A4 impresas** (`escenario_1_vector_add.pdf`, `escenario_2_compute_bound.pdf`, `escenario_3_ridge_point.pdf`), recortadas en tarjetas individuales por las líneas punteadas antes de la clase.
- **Pizarrón + marcadores** (2 colores, idealmente).
- **Cronómetro** (cualquier celular sirve).
- **Papel y lápiz** para el procesador.
- **Este documento** impreso, para el verificador (con la clave de respuestas al final).

**Preparación previa (5 min):** recortar las 60 tarjetas (20 por escenario). Separar en 6 pilas: A₁, B₁, A₂, B₂, A₃, B₃.

---

## Roles (5 personas)

- **Procesador (1 estudiante)** — sentado en un escritorio al frente. Recibe pares de tarjetas y realiza la operación.
- **Runner (1 estudiante)** — parado junto a las pilas de tarjetas, a **4–5 metros** del procesador. Camina hasta el procesador con cada par.
- **Timer (1 estudiante)** — cronometra cada escenario y anota el tiempo total en el pizarrón.
- **Verificador (instructor)** — confirma cada resultado que el procesador anuncia; tiene la clave de respuestas.
- **Audiencia** — observa activamente **quién está inactivo en cada momento**.

---

## Configuración del aula

```
   [ Audiencia ]

   [Pilas de     ]           [ Procesador  ]
   [ tarjetas    ] ← 4–5 m → [ (escritorio)]
        ↑                          ↑
     Runner                    procesa
```

La distancia importa: con 4–5 m, caminar toma visiblemente más tiempo que sumar dos dígitos. En un aula normal, ubica el runner y el procesador en esquinas opuestas.

---

## Antes de empezar (2 min)

1. Anuncia: *"Vamos a simular físicamente un procesador y su memoria. Al final construiremos un modelo a partir de lo que observamos."*
2. Escribe en el pizarrón la operación del Escenario 1.
3. **Ronda de calentamiento**: 2–3 pares con la suma simple, para que runner y procesador encuentren su ritmo. No cronometrar.

---

## Escenario 1 — Memory-bound (Vector Add)

**Operación en el pizarrón:** `c = a + b`

**Procedimiento:**
1. Runner en las pilas A₁ y B₁.
2. Runner toma a[0] y b[0], camina al procesador, entrega.
3. Procesador dice el resultado en voz alta. Verificador confirma.
4. Runner ya está caminando de vuelta mientras el procesador termina.
5. Repetir con los 10 pares.
6. Timer anota el tiempo total.

**Duración esperada:** 30–45 segundos para 10 pares.

**A observar (dirigir la atención de la audiencia):**
- El **procesador pasa la mayor parte del tiempo esperando**.
- La suma es trivial (~1 s); caminar toma 3–4 s en cada dirección.
- Instrucción a la audiencia: *"¿Cuántos segundos totales estuvo inactivo el procesador?"*

**Pregunta al terminar el escenario:**
> *"Si reemplazamos al procesador por alguien que suma el doble de rápido, ¿el tiempo total se reduciría notoriamente?"*

Respuesta esperada: **No.** El runner es el cuello de botella. Un procesador más rápido solo significa más espera.

---

## Escenario 2 — Compute-bound

**Operación en el pizarrón:** `c = (a + b)² + a × b`

Ejemplo: para a=2, b=3 → `(2+3)² + 2×3 = 25 + 6 = 31`.

**Procedimiento:**
1. Runner en las pilas A₂ y B₂.
2. Mismo flujo, pero ahora el procesador hace un cálculo pesado.
3. 10 pares, cronometrado.

**Duración esperada:** 90–150 segundos para 10 pares (~3× el escenario 1).

**A observar:**
- El **runner pasa la mayor parte del tiempo esperando** que el procesador termine.
- El cómputo es ahora el cuello de botella.
- Instrucción a la audiencia: *"Cuenten los segundos que el runner está inactivo."*

**Pregunta al terminar el escenario:**
> *"Si tuviéramos dos runners entregando al doble de velocidad, ¿esto iría más rápido?"*

Respuesta esperada: **No.** El procesador es el cuello de botella. Más runners solo significa tarjetas acumulándose en el escritorio.

---

## Escenario 3 — Ridge point (balanceado)

**Operación en el pizarrón:** `c = a + 2b`

Ejemplo: para a=1, b=5 → `1 + 10 = 11`.

**Procedimiento:**
1. Runner en las pilas A₃ y B₃.
2. Probar con esta operación.
3. **Si el procesador termina antes de que llegue el siguiente par → aumentar la complejidad** (por ejemplo, `c = a² + b + 1`).
4. **Si el runner llega mientras el procesador todavía calcula → simplificar** (por ejemplo, `c = a + b + 1`).
5. Objetivo: **ninguno de los dos está inactivo**.

**Duración esperada:** 50–70 segundos para 10 pares.

**A observar:**
- Procesador y runner **ambos ocupados casi todo el tiempo**.
- Nadie está inactivo.
- **Este es el punto óptimo**: no se puede acelerar el sistema mejorando solo la memoria o solo el cómputo — hay que mejorar los dos.

---

## Debrief y construcción del roofline (~15 min)

Ahora, en el pizarrón, construir el diagrama **preguntando a los estudiantes dónde ubicar cada escenario**.

### Paso 1: dibujar los ejes

- **Eje X:** *"operaciones por par de datos entregado"* (intensidad aritmética, FLOP/byte)
- **Eje Y:** *"resultados por segundo"* (rendimiento alcanzado, FLOP/s)

### Paso 2: preguntar por el Escenario 1

> *"En el escenario 1, ¿el procesador estaba a máxima velocidad?"* — No.
> *"¿Y el runner?"* — Sí.

- Dibujar el **techo de ancho de banda** (diagonal desde el origen).
- Ubicar el punto: baja intensidad, bajo rendimiento, **sobre el techo diagonal**.

### Paso 3: preguntar por el Escenario 2

> *"¿Y en el escenario 2?"* — Procesador al máximo, runner esperando.

- Dibujar el **techo de cómputo** (línea horizontal a máxima performance).
- Ubicar el punto: alta intensidad, alto rendimiento, **sobre el techo horizontal**.

### Paso 4: preguntar por el Escenario 3

> *"¿Y el escenario 3?"* — Ambos al máximo.

- El punto está **en la intersección** de los dos techos: el **ridge point**.

### Paso 5: nombrar los regímenes

- Izquierda del ridge point → **memory-bound**: mejorar ancho de banda ayuda; más cómputo no.
- Derecha del ridge point → **compute-bound**: mejorar velocidad de cómputo ayuda; más ancho de banda no.
- En el ridge point → equilibrado.

### Paso 6: puente al perfilado

> *"¿Dónde cae un kernel real de CUDA en este diagrama? ¿Cómo lo sabemos?"*

Respuesta: **medimos ambas coordenadas con `ncu`**. La intensidad aritmética se calcula del algoritmo; el rendimiento alcanzado se mide con el profiler. En la próxima sesión veremos exactamente esto.

---

## Clave de respuestas (para el verificador)

### Escenario 1 — `c = a + b`

| i | a[i] | b[i] | **c[i]** |
|:-:|:----:|:----:|:--------:|
| 0 | 3 | 4 | **7** |
| 1 | 7 | 2 | **9** |
| 2 | 2 | 8 | **10** |
| 3 | 8 | 3 | **11** |
| 4 | 5 | 7 | **12** |
| 5 | 1 | 6 | **7** |
| 6 | 9 | 1 | **10** |
| 7 | 4 | 9 | **13** |
| 8 | 6 | 2 | **8** |
| 9 | 5 | 5 | **10** |

### Escenario 2 — `c = (a + b)² + a × b`

| i | a[i] | b[i] | (a+b)² | a·b | **c[i]** |
|:-:|:----:|:----:|:------:|:---:|:--------:|
| 0 | 2 | 3 | 25 | 6  | **31** |
| 1 | 3 | 5 | 64 | 15 | **79** |
| 2 | 4 | 1 | 25 | 4  | **29** |
| 3 | 1 | 4 | 25 | 4  | **29** |
| 4 | 5 | 2 | 49 | 10 | **59** |
| 5 | 3 | 6 | 81 | 18 | **99** |
| 6 | 6 | 2 | 64 | 12 | **76** |
| 7 | 2 | 5 | 49 | 10 | **59** |
| 8 | 4 | 3 | 49 | 12 | **61** |
| 9 | 5 | 1 | 36 | 5  | **41** |

### Escenario 3 — `c = a + 2b` (o la variante que uses)

| i | a[i] | b[i] | **c[i]** (a+2b) |
|:-:|:----:|:----:|:---------------:|
| 0 | 1 | 5 | **11** |
| 1 | 4 | 2 | **8**  |
| 2 | 2 | 6 | **14** |
| 3 | 5 | 1 | **7**  |
| 4 | 3 | 4 | **11** |
| 5 | 2 | 3 | **8**  |
| 6 | 4 | 2 | **8**  |
| 7 | 1 | 6 | **13** |
| 8 | 5 | 1 | **7**  |
| 9 | 3 | 5 | **13** |

Si cambias la operación durante el escenario 3 para balancear, calcula la clave sobre la marcha.

---

## Tips y contingencias

- **Voluntariado opcional.** Algunos estudiantes de último año se sienten incómodos con dinámicas físicas. Enmarca ligero: *"Voluntarios bienvenidos, sin problema mirar."* Si nadie se ofrece, hazlo tú como procesador o como runner.

- **El runner deja caer una tarjeta.** ¡Perfecta analogía de un cache miss! Úsalo: *"Error en la transferencia — hay que reintentar. En un GPU esto también cuesta ciclos."*

- **El procesador se equivoca.** Otra oportunidad: *"Los errores introducen serialización. Por eso nos preocupa la corrección incluso en código rápido."*

- **Los tiempos exactos no importan.** Lo importante es la **razón** entre los escenarios y el **tiempo inactivo visible**. Números aproximados comunican el punto igual.

- **Asigna observadores.** Divide la audiencia: mitad observa al procesador, mitad observa al runner. Al final de cada escenario, pídeles reportar cuánto tiempo estuvo inactiva su persona asignada.

- **Ajustar el escenario 3 en vivo.** Es normal que no salga balanceado al primer intento — modelar el ajuste explícitamente es parte de la lección: *"Estamos calibrando la intensidad aritmética hasta encontrar el ridge point. Un profiler hace lo mismo, pero con números."*

- **Transición al siguiente contenido.** Termina con: *"En la sesión 6 formalizaremos esto como el modelo roofline. Pero ya tienen la intuición completa — el resto es notación."*
