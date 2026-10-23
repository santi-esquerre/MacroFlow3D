# Resolución necesaria frente a la varianza de ln K: por qué λ/h crece con σ² (análisis post-hoc de SF-37)

- Fecha: 2026-10-23
- Estado: `complete` — **post-hoc, NO pre-registrado**. No hay corridas nuevas. No cambia ningún veredicto de SF-37 ni
  ninguna decisión. No hace ninguna afirmación sobre α_L ni α_T (Regla 1).
- Para qué sirve: insumo para que el dueño fije la resolución y el dominio de SF-38/SF-39. Para qué **no** sirve: no es
  evidencia de aceptación ni reemplaza la escalera de grilla pre-registrada de SF-39.
- Datos: los CSV versionados de SF-37 (`artifacts/2026-10-19-sf37-lagrangian-cube-verification/analysis/v1v2/`; sumas
  sha256 en la salida del script) y la revisión post-hoc
  [`2026-10-22-sf37-posthoc-review-and-mecom-feasibility.md`](2026-10-22-sf37-posthoc-review-and-mecom-feasibility.md)
  (`posthoc/lh10.txt`).
- Script y salida: [`artifacts/2026-10-23-resolution-vs-variance/`](artifacts/2026-10-23-resolution-vs-variance/)
  (`scripts/resolution_vs_variance.py`, `analysis/resolution_vs_variance.txt`).
- Teoría: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) §5 y Reglas 1-3;
  [`beaudoin-de-dreuzy-2013-key-claims.md`](../theory/beaudoin-de-dreuzy-2013-key-claims.md) ("spatial resolution must be
  sufficient relative to correlation length").
- Contexto: nota de SF-37 [`2026-10-19-sf37-lagrangian-cube-verification.md`](2026-10-19-sf37-lagrangian-cube-verification.md)
  §3 (λ/h de producción 8 / 16 / 32); decisión
  [`2026-10-22-lean-lagrangian-stage.md`](../decisions/2026-10-22-lean-lagrangian-stage.md) ítem 4; nota de SF-40
  [`2026-10-22-sf40-lean-lagrangian-stage.md`](2026-10-22-sf40-lean-lagrangian-stage.md) (tabla de factibilidad).

## Pregunta

¿Por qué el consumidor lagrangiano necesita más celdas por longitud de correlación (λ/h) cuanto mayor es σ² = Var(ln K)?
¿Con qué ley crece esa necesidad, cuánto de eso está establecido y qué implica para SF-38/SF-39?

## Respuesta corta

1. **Lo establecido.** A resolución fija, el error de las estadísticas de la pluma crece con σ².
   - El peor error de cada semilla crece con σ² en las tres semillas.
   - Celda por celda (misma semilla, estadística, plano y λ/h), el error a σ² = 4 supera al de σ² = 1 en 114 de 126
     celdas de σ_T² y en 116 de 126 de var τ.
   - A λ/h = 16, el error típico (la mediana) es 4,8 veces mayor a σ² = 4 que a σ² = 1, y el peor caso es 13,5 veces
     mayor.
2. **El mecanismo derivado.** σ² no cambia la escala de Y = ln K, que sigue siendo λ. Lo que cambia es la del
   coeficiente de la ecuación de Darcy: K = e^Y y su gradiente varían en distancias del orden de λ/σ. Un esquema de
   segundo orden comete entonces un error que escala con (σh/λ)². Para mantener fijo ese error, λ/h tiene que crecer
   **al menos** como σ: λ/h ∝ σ^β con β ≥ 1. Es un argumento de escala, no una cota.
3. **El error típico sigue esa ley.** La mediana de los errores da β ≈ 1,1-1,2.
4. **El peor caso crece más rápido**, con β ≈ 1,6-2,0. Ese peor caso es siempre la misma realización (semilla 3102)
   en el plano más lejano (64 λ). La lectura es que una realización canalizada amplifica el error a lo largo del
   recorrido. Esto es una **hipótesis**, compatible con los datos pero no aislada.
5. **El error crece con la distancia recorrida:** la mediana va como x1^0,66-0,80 entre 8 λ y 64 λ. Por eso los λ/h
   de SF-37, medidos hasta 64 λ, no se pueden trasladar sin verificación a dominios de 104 a 416 λ.
6. **El signo del error es sistemático:** la grilla gruesa **subestima** la dispersión transversal (17-19 de 21
   celdas por σ²). Una corrida con resolución insuficiente quedaría sesgada hacia α_T chicos, justo en la dirección
   de la afirmación de Lester. Ese riesgo amenaza directamente el objetivo del proyecto.
7. **Los estimadores de α piden más resolución que σ_T²:** el estimador longitudinal pide λ/h 16 / 32 / 64 a
   σ² = 1 / 2,25 / 4.

## 1. Qué se midió y qué significa "resolución necesaria"

SF-37 hizo las corridas en el cubo periódico.

**Configuración:**
- λ = ℓ = 1/4 en una caja de lado 1, es decir, un período de 4 λ;
- seis grillas candidatas, N = 32 … 256 (λ/h = 8, 10, 16, 20, 32, 64), y la de referencia N = 512 (λ/h = 128), todas
  sobre el mismo continuo del campo;
- tres semillas (3101-3103) y σ² ∈ {1, 2,25, 4};
- 256² partículas, con paso máximo h/8.

**Qué se compara:** dos estadísticas Q de la pluma, σ_T²(x1) y var τ(x1), en 7 planos x1 ∈ {0,25 … 16}. El plano
x1 = 16 equivale a 64 λ. Las celdas son 3 semillas × 2 Q × 7 planos = 42 por candidato y por σ².

**La regla de SF-37 (RR5):** un candidato pasa si |Q(N) − Q(512)| ≤ max(0,05 |Q(512)|, 2 SE_D) en **todas** las
celdas. La resolución de producción es la menor que pasa junto con todas las más finas.

Tres propiedades de esta definición condicionan todo lo que sigue:
- **Es auto-convergencia:** la referencia es el mismo consumidor a λ/h = 128.
- **Es un criterio de peor caso:** basta una celda que falle.
- **Depende de la estadística elegida** y de la distancia hasta el último plano.

## 2. Por qué tiene que crecer: derivaciones

### 2.1 La escala del coeficiente se achica como λ/σ (exacto)

Y es gaussiano con covarianza C(r) = σ² exp(−(r/λ)²). Su espectro no depende de σ²: solo su amplitud cambia. A
λ/h = 8, Y ya está resuelto casi exactamente. Lo que no se mantiene es la escala de K = e^Y. La identidad lognormal
da Cov[K(x), K(x+r)] ∝ e^{C(r)} − 1, y la correlación normalizada es ρ_K(r) = (e^{C(r)} − 1)/(e^{σ²} − 1).

Su microescala de Taylor relativa a la de Y vale √((1 − e^{−σ²})/σ²), que tiende a 1/σ. Además, Var(∂Y/∂x_i) =
−C''(0) = 2σ²/λ². El script verifica la identidad por Monte Carlo: a σ² = 1, r = 0,5 λ, ρ_K da 0,6872 contra
0,6861, con un error estándar de 0,0155.

| σ² | rms(∂Y/∂x_i)·λ | e-pliegue de ρ_K / λ | microescala K / microescala Y | K(+2σ)/K(−2σ) |
|---|---|---|---|---|
| 1 | 1,41 | 0,845 | 0,795 | 55 |
| 2,25 | 2,12 | 0,680 | 0,631 | 403 |
| 4 | 2,83 | 0,527 | 0,495 | 2981 |
| 6,25 | 3,54 | 0,417 | 0,400 | 22026 |

### 2.2 La ecuación de Darcy hereda esa escala (escala derivada, no cota)

∇·(e^Y ∇h) = 0 es equivalente a ∇²h + ∇Y·∇h = 0. El coeficiente ∇Y, de tamaño √2 σ/λ, actúa como un "transporte"
dentro de la ecuación de la carga. Por eso la carga y la velocidad varían en distancias del orden de λ/σ, y el error
de truncamiento de un esquema de segundo orden escala con γ² = (√2 σ h/λ)².

Un ejemplo concreto en este código: SF-19 pone en cada cara la media armónica de K
(`src/physics/flow/AffinePeriodicFlowSolver.cu:205,232,259`, `src/numerics/operators/face_coefficient.cuh:19`).

- Para dos celdas vecinas, la media armónica es exactamente e^{(Y1+Y2)/2}/cosh(ΔY/2).
- Se aparta de la media geométrica en un factor 1 − ΔY²/8 + O(ΔY⁴).
- Como E[ΔY²] = 2σ²(1 − e^{−(h/λ)²}) ≈ 2σ²h²/λ², esa diferencia vale en media σ²h²/(4λ²).

Esto ilustra que la discretización "ve" la variación de K dentro de la celda en la medida σh/λ; no prueba cuál de
las dos medias es la correcta.

**Consecuencia:** para mantener fijo el error local hay que mantener fijo γ, es decir, λ/h ∝ σ (β = 1).

Lo que **no** limita la resolución:
- el campo Y mismo, por su espectro;
- el spline cúbico de ln K que usa el tracker: su error es lineal en σ y de cuarto orden en h/λ.

El límite está en la solución del flujo y en su representación.

### 2.3 El transporte puede amplificar el error (hipótesis)

Las estadísticas de la pluma integran errores de velocidad a lo largo de recorridos largos. A mayor σ², las
trayectorias vecinas se separan más rápido: SF-30 vio que a σ² = 4 se descorrelacionan tras 4 a 8 períodos. Además,
las estadísticas dependen de los extremos (canales rápidos y zonas lentas). Si el error se acumula y se amplifica en
el recorrido, la resolución necesaria crece más rápido que σ. Esto no está derivado; §3 muestra lo que los datos
dicen al respecto.

## 3. Lo medido (todas las cifras de `analysis/resolution_vs_variance.txt`)

### 3.1 A resolución fija el error crece con σ²

El error relativo es |Q(N) − Q(512)|/|Q(512)|, sobre las 42 celdas de cada caso:
- **peor:** el máximo de las 42;
- **mediana:** la mediana de las 42;
- **media de semillas:** el máximo, sobre las 14 combinaciones de estadística y plano, del valor absoluto del
  promedio con signo de las 3 semillas. Es un indicador aproximado del sesgo de una media de ensemble.

| λ/h | σ² = 1: peor / mediana / media de semillas | σ² = 2,25 | σ² = 4 |
|---|---|---|---|
| 8 | 2,9 % / 0,50 % / 1,8 % | 14,2 % / 1,2 % / 6,4 % | 28,1 % / 1,9 % / 12,0 % |
| 16 | 0,75 % / 0,12 % / 0,46 % | 3,8 % / 0,30 % / 1,7 % | 10,1 % / 0,58 % / 4,3 % |
| 32 | 0,18 % / 0,03 % / 0,11 % | 1,1 % / 0,08 % / 0,48 % | 2,8 % / 0,15 % / 1,1 % |

Celda por celda, el orden σ² = 1 < 4 se cumple en 114/126 celdas de σ_T² y en 116/126 de var τ. El orden estricto
1 < 2,25 < 4 se cumple en 102/126 y 71/126; var τ es la estadística más ruidosa (§3.6, §3.7). El peor error de cada
semilla crece con σ² en las tres. A λ/h = 16 es:

| semilla | σ² = 1 | σ² = 2,25 | σ² = 4 |
|---|---|---|---|
| 3101 | 0,64 % | 1,46 % | 2,90 % |
| 3102 | 0,75 % | 3,76 % | 10,1 % |
| 3103 | 0,22 % | 0,46 % | 3,20 % |

### 3.2 El peor caso es siempre la misma realización, en el plano más lejano

A λ/h 16 y 32, y en las tres amplitudes, la celda peor es siempre la semilla 3102, σ_T², en x1 = 16 (64 λ). A
σ² = 4 esa semilla tiene errores de 3,2 a 4,5 veces los de las otras dos (λ/h 16 y 32). La revisión post-hoc la describe como una
realización canalizada (var x balística, flujo concentrado; P5).

### 3.3 El error crece con la distancia recorrida

Mediana del error relativo de σ_T² sobre las semillas, a λ/h = 16:

| σ² | x1 = 1 (4 λ) | x1 = 4 (16 λ) | x1 = 16 (64 λ) | pendiente d ln(err)/d ln(x1), x1 de 2 a 16 |
|---|---|---|---|---|
| 1 | 0,04 % | 0,19 % | 0,48 % | 0,80 |
| 2,25 | 0,16 % | 0,57 % | 1,46 % | 0,70 |
| 4 | 0,41 % | 1,14 % | 2,90 % | 0,66 |

El error está presente desde el primer período, y crece con la distancia de manera parecida en las tres amplitudes.

### 3.4 La resolución necesaria como variable continua

Interpolo log-log entre los candidatos el λ/h donde el peor error relativo cruza 5 %. Es solo la rama relativa de
la regla; no usa la rama 2 SE.

| σ² | λ/h necesario (interpolado) | γ = √2σh/λ ahí | valor de SF-37 (discreto) | γ en el valor de SF-37 |
|---|---|---|---|---|
| 1 | 6,1 (extrapolado bajo 8) | 0,233 | 8 | 0,177 |
| 2,25 | 13,8 (entre 10 y 16) | 0,154 | 16 | 0,133 |
| 4 | 22,5 (entre 20 y 32) | 0,126 | 32 | 0,088 |

γ baja al crecer σ: el peor caso pide más que "γ fijo", es decir, crece más rápido que λ/h ∝ σ. Los valores 8 / 16 / 32
son el primer candidato que pasa. Las necesidades continuas quedan entre 14 y 30 % más abajo, y el salto de 20 a 32
a σ² = 4 lo fija **una** celda que falla a 20.

### 3.5 El exponente depende de qué error se mire

Sin elegir un umbral: β = ln(err(σ²=4)/err(σ²=a)) / (p ln(σ_4/σ_a)), con orden p = 2 (el orden observado, §3.6).

| métrica | β a λ/h 16 (de σ² 1 a 4 / de 2,25 a 4) | β a λ/h 32 |
|---|---|---|
| peor celda | 1,88 / 1,72 | 1,97 / 1,64 |
| media de semillas | 1,61 / 1,59 | 1,69 / 1,47 |
| mediana | 1,11 / 1,16 | 1,16 / 1,13 |

La mediana sigue la escala local de §2.2 (β ≈ 1). El peor caso y la media de 3 semillas, dominada por 3102, crecen
más rápido.

### 3.6 Orden de convergencia

En la escalera V2 (N 128 → 256 → 512):
- la mediana del orden es 2,00 en las tres amplitudes y las dos estadísticas;
- las celdas con orden dentro de [1,5; 2,5] son 42/42, 38/42 y 34/39 a σ² = 1 / 2,25 / 4;
- a σ² = 4, var τ tiene un orden mínimo de −0,14.

A mayor σ² hay más celdas fuera del régimen asintótico, incluso a λ/h 32-64.

### 3.7 Signo del error

| σ² | celdas de σ_T² con Q(N) < Q(512) | celdas de var τ con Q(N) < Q(512) |
|---|---|---|
| 1 | 19/21 en todos los λ/h | 13/21 |
| 2,25 | 17-18/21 | 10/21 |
| 4 | 17-18/21 | 11-13/21 |

La grilla gruesa da sistemáticamente **menos** dispersión transversal. La grilla gruesa actúa como un medio
suavizado, menos heterogéneo. var τ no muestra un signo sistemático.

### 3.8 Las cantidades que estima SF-39 piden más

Sombra D8 de SF-37: misma regla RR5, sobre las series del estimador de B&dD, no bloqueante.

| σ² | var x1 | d var x1/dt | var x2 | var x3 | d var x2/dt | d var x3/dt | d σ_T²/d x1 |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 16 | 8 | 8 | 16 | 16 | 16 |
| 2,25 | 32 | 32 | 16 | 10 | 32 | 64 | 20 |
| 4 | 64 | 64 | 32 | 16 | 32 | 64 | 32 |

`posthoc/lh10.txt`, peor de 3 semillas: error de una tasa del tipo α_L = Δvar x/Δt entre t = 8 y 16.

| λ/h | σ² = 1 | σ² = 2,25 | σ² = 4 |
|---|---|---|---|
| 10 | 12,0 % | 44,2 % | 73,9 % |
| 20 | 2,9 % | 11,7 % | 26,7 % |
| 32 | 1,1 % | 4,4 % | 10,4 % |

Hay dos razones para que pidan más:
- una derivada de un momento amplifica el error relativo del momento;
- la dispersión longitudinal la dominan los canales rápidos, que son los extremos peor resueltos.

## 4. Lectura crítica

### 4.1 Qué queda establecido

La columna de certeza es una adaptación cualitativa de los dominios de GRADE: riesgo de sesgo, inconsistencia,
indirección, imprecisión. No es una evaluación GRADE formal ni un puntaje.

| afirmación | evidencia | certeza | razones |
|---|---|---|---|
| A resolución fija, el error de σ_T² y var τ crece con σ² | §3.1; derivación §2 | alta, **en el cubo** | consistente en semillas, estadísticas y planos; mecanismo derivado; indirección para el dominio largo |
| La resolución necesaria crece al menos como σ (β ≥ 1) | §2.2 (escala); mediana β ≈ 1,1 | moderada | escala, no cota; 3 valores de σ² |
| El peor caso crece con β ≈ 1,6-2,0 | §3.4-§3.5 | baja | 3 semillas y una domina (imprecisión, inconsistencia); métrica de valor extremo |
| El exceso sobre β = 1 viene de realizaciones canalizadas y de la acumulación en el recorrido | §3.2-§3.3 | baja (hipótesis) | compatible, pero las alternativas de §4.2 no están excluidas |
| La grilla gruesa subestima σ_T² | §3.7 | moderada | signo consistente; referencia del mismo pipeline; var τ sin signo |
| Los estimadores de α piden más que σ_T² | §3.8 | moderada | criterio sombra; 3 semillas; d var x3/dt pide 64 ya a σ² = 2,25 y d var x2/dt solo 32, porque una componente domina por realización (P3 de la revisión post-hoc) |
| Los λ/h del cubo valen en el dominio largo | — | **no evaluada** | indirección: cubo periódico de 4 λ, recorrido de 64 λ, error que crece con la distancia |

### 4.2 Explicaciones alternativas y confusores

- **Cubo periódico.** En 64 λ las partículas cruzan 16 veces el mismo medio de 4 λ, y el régimen es superdifusivo
  (P3 de la revisión post-hoc). La acumulación del error puede ser distinta en un medio aleatorio no periódico, y no se
  sabe en qué sentido.
- **Valores extremos.** El "peor de 42 celdas en 3 semillas" crece con la cantidad de realizaciones que se miran. En
  un ensemble de cientos habrá realizaciones más canalizadas que 3102. Un requisito definido sobre la peor
  realización no es estable al crecer N_R. Lo que importa para α, que es una media de ensemble, es el sesgo de la
  media.
- **Régimen pre-asintótico.** A σ² = 4 el orden observado se dispersa (§3.6). Parte del "exceso" puede ser
  pre-asintótico y no una ley de amplificación.
- **Extremos locales del campo.** Cerca de un máximo de Y de altura u, K decae en una distancia del orden de λ/√u
  (escala, no derivada aquí). Las realizaciones con máximos más altos tienen estructuras más finas. Esto compite con
  la explicación por amplificación en el recorrido.
- **Referencia compartida.** Q(512) sale del mismo SF-18/SF-19/SF-28. Un error común a todas las grillas no se ve.
  Con orden 2, el error del candidato 64 se subestima en ~25 % (despreciable en los candidatos gruesos).
- **El paso se refina con h** (h_max = h/8). La escalera de grilla también refina la integración. V1 mostró que la
  integración es despreciable a h/8 (h/8 contra h/32: MS(dx_T)/σ_T² ≤ 1,27e-13).
- **Específico del método.** Pollock queda más cerca de la referencia fina a igual N (P2 de la revisión post-hoc).
  La necesidad de resolución es una propiedad de esta representación (splines del potencial), no de cualquier método.
  En particular, esta nota no evalúa la resolución de B&dD (λ/Δ = 10).

### 4.3 Qué no se puede concluir

- Ninguna ley cuantitativa de λ/h(σ²) con incertidumbre: hay 3 amplitudes, 3 semillas y un exponente que depende de
  la métrica.
- Ningún valor para σ² = 6,25. Extrapolar el 22,5 interpolado con β entre 1,1 y 1,9 da 29 a 34: entre los candidatos
  32 y 64, sin medición.
- Nada sobre α_L ni α_T.

## 5. Consecuencias para SF-38 / SF-39

Son propuestas para decisión del dueño; ninguna cambia una especificación por sí misma.

1. **Definir el requisito sobre la cantidad de interés.** SF-39 estima medias de ensemble de α_L y α_T, que son tasas
   de momentos. El requisito debería expresarse como sesgo de esas medias frente a la semi-amplitud del intervalo de
   confianza objetivo, no como "toda realización dentro de 5 %". Medido como sesgo de la media de semillas, la
   necesidad es menor que la del peor caso: a σ² = 4, 4,3 % a λ/h 16 frente a 10,1 % del peor. Pero con 3 semillas esa
   media está dominada por una sola.
2. **Medir a la longitud de producción.** La escalera en dominio reducido de SF-39 (32 λ de largo) es más corta que
   el recorrido del cubo (64 λ) y que los dominios de producción (104 a 416 λ). Si la pendiente de §3.3 siguiera más
   allá de 64 λ, el error relativo se multiplicaría (estimación, no medición):
   - por ~1,4 a 104 λ y por ~2,2 a 208 λ, con pendiente 0,66;
   - por ~4,5 a 416 λ, con pendiente 0,80.

   Ese último caso pondría el peor error de σ² = 1 a λ/h 8 (2,9 % a 64 λ) por encima de 5 %. La escalera tiene que
   cubrir la longitud del estudio, o medir el error en función de la distancia hasta ahí. Las comparaciones pareadas
   (mismos campos) tienen poco ruido.
3. **Medir la cola de realizaciones en el piloto.** Hace falta la distribución del error por realización con 10 a 20
   semillas a σ² = 4, y una métrica de canalización por realización para ver si predice el error.
4. **El estimador de α_L a σ² = 4 pediría λ/h 64 (D8).** A 4,9e8 celdas eso deja 52 × 6 × 6 λ, que no sirve como
   dominio. Opciones a decidir:
   - reportar α_L con el sesgo de grilla medido;
   - extrapolación de Richardson pre-registrada sobre pares pareados, ya que el orden mediano es 2 (riesgo: las
     celdas pre-asintóticas de §3.6);
   - limitar la clasificación a σ² ≤ 2,25 para α_L.
5. **No suponer que "más grueso es conservador".** Una grilla gruesa da α_T más chicos (§3.7). Un α_T ≈ 0 sin la
   escalera convergida no es evidencia a favor de Lester (Regla 1). La regla de clasificación de SF-39 debería decir
   explícitamente qué hacer si α_T **crece** al refinar.
6. **Costo.** Para un dominio fijo en unidades de λ, las celdas escalan como (λ/h)³ ∝ σ^{3β}. De σ² = 1 a 4 eso es
   un factor de 10 (β = 1,1) a 52 (β = 1,9); los valores discretos de SF-37 (8 → 32) dan 64. Esto, y no la memoria
   del tracker, es lo que achica el dominio a σ² = 4 (104 × 12 × 12 λ en la caja de SF-40).
7. **σ² = 6,25** necesita su propia escalera antes de cualquier corrida exploratoria.

## 6. Reproducción

```bash
python3 -I docs/experiments/artifacts/2026-10-23-resolution-vs-variance/scripts/resolution_vs_variance.py \
    docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification \
    > docs/experiments/artifacts/2026-10-23-resolution-vs-variance/analysis/resolution_vs_variance.txt
```

- Solo biblioteca estándar de Python.
- Lee `lambda_h_cells.csv`, `v2_cells.csv` y `d8_shadow_lambda_h.csv`, e imprime sus sha256.
- Auto-chequeos:
  - el script reproduce los 756 veredictos de SF-37 a partir de |D| ≤ max(thr_rel, thr_2se);
  - verifica la identidad lognormal por Monte Carlo (semilla fija).
- `lh10.txt` se cita tal como está versionado (`artifacts/2026-10-19-sf37-lagrangian-cube-verification/analysis/posthoc/`).

## 7. Límites

- Todo es post-hoc y exploratorio. Las métricas "media de semillas" y "mediana", la interpolación continua y el
  exponente β se definieron después de ver los datos. Sirven para orientar, no para aceptar.
- Hay un solo modelo de campo (gaussiano, ℓ = 1/4), una geometría (cubo periódico), tres semillas, dos estadísticas
  y un solo método (el consumidor lagrangiano sobre splines).
- §2.2 es un argumento de escala del error de truncamiento, no una cota.
- §2.3 y la atribución de §3.2 son hipótesis.
- No se corrió nada nuevo, ni se miró ningún dato fuera de los CSV y el `lh10.txt` citados.

## Herramientas

La estructura de la lectura crítica (§4) siguió las guías de revisión de la librería de skills:

> Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of
> Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
