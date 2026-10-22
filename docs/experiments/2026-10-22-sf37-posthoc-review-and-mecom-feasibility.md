# SF-37: revisión científica post-hoc, explicación del tracker lagrangiano y factibilidad en el dominio de MECOM 2025

- Fecha: 2026-10-22
- Estado: `post-hoc, NO pre-registrado; ningún veredicto de SF-37 cambia; informe para la revisión humana del PR #56`
- Incremento: [`SF-37-lagrangian-cube-verification.md`](../plans/active/lester-eq14/increments/SF-37-lagrangian-cube-verification.md)
  (PR https://github.com/santi-esquerre/MacroFlow3D/pull/56; cabeza con código auditada `d866c40`)
- Nota experimental de la campaña: [`2026-10-19-sf37-lagrangian-cube-verification.md`](2026-10-19-sf37-lagrangian-cube-verification.md)
- Teoría: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) §4-6 y reglas 1-3;
  [`beaudoin-de-dreuzy-2013-key-claims.md`](../theory/beaudoin-de-dreuzy-2013-key-claims.md)
- Decisión: [`2026-10-17-lagrangian-consumer-sigma4.md`](../decisions/2026-10-17-lagrangian-consumer-sigma4.md)
- Antecedentes: [`2026-10-17-sf36-lagrangian-tracker-core.md`](2026-10-17-sf36-lagrangian-tracker-core.md),
  [`2026-10-17-sf35-sigma4-feasibility.md`](2026-10-17-sf35-sigma4-feasibility.md); presentación MECOM 2025
  (`docs/references/presentation.pdf`, no versionada) y código `legacy/` (MECOM).
- Scripts y salidas (`A` = `artifacts/2026-10-19-sf37-lagrangian-cube-verification/`): `A/scripts/posthoc/*.py`,
  `A/analysis/posthoc/*.txt`, sumas en `A/analysis/posthoc/SHA256SUMS`.

**Qué es este documento.** Dos informes entregados al dueño el 2026-10-22 durante la revisión del PR #56:

- Parte I: auditoría científica e interpretación de los resultados de SF-37.
- Parte II: explicación del tracker lagrangiano, de los experimentos que lo validan (SF-36, SF-37) y de su
  factibilidad en el dominio del trabajo MECOM 2025.

**Procedencia de los números.**

- Los marcados **[post-hoc]** salen de análisis no pre-registrados. Esos análisis leen los CSV ya versionados del
  artefacto o, en dos casos, los `.npy` por partícula que quedaron en el host V100 (`~/sf37_campaign`). Esos dos se
  leyeron con `scripts/remote --increment SF-37 exec`, sin escribir nada allá.
- No se hizo ninguna corrida nueva.
- El resto de los números está copiado de la nota de SF-37, de sus digests o de la nota de SF-36.

Reproducción:

```bash
A=docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification
for s in growth bracket bracket2 alpha_scale exponents stretch lh10; do python3 $A/scripts/posthoc/$s.py $A; done
# los dos tests de pesos de V3b leen ~/sf37_campaign en el host V100 (solo lectura):
B=$(base64 -w0 $A/scripts/posthoc/v3b_weight_test2.py)
scripts/remote --increment SF-37 exec -- "echo $B | base64 -d | python3 - \$HOME/sf37_campaign"
```

---

# Parte I. Auditoría científica e interpretación de SF-37

## I.1 Resumen

**Lo que queda establecido.** En el cubo periódico, el consumidor lagrangiano produce estadísticas de pluma que
convergen de forma limpia con la grilla (orden ≈ 2). El error de integración es despreciable y la divergencia de la
representación es O(h²).

**Lo que no queda establecido:**

- el control por tolerancia (V1 vacío, RR15);
- la ley de estiramiento (V5);
- cualquier valor de α.

**Dos resultados post-hoc que cambian la lectura del PR:**

1. El `discordant` de V3b es un artefacto de cómo el comparador de Pollock asigna los pesos de inyección. El
   mecanismo está confirmado, pero el que figuraba en el PR y en la nota era inexacto en un detalle (§I.3, P1).
2. En el cubo, a igual grilla, Pollock queda más cerca de la referencia fina que el propio consumidor (P2).

**La tensión principal para el rumbo:** λ/h = 32 a σ² = 4 no entra en memoria con el dominio largo planeado.

## I.2 Qué se midió y por qué

El proyecto quiere medir α_L y α_T hasta σ² = 4. Lester 2023 (§5 de la nota de teoría) nombra dos fuentes de
dispersión transversal espuria:

- **(A) La representación de la velocidad.** Una interpolación puede ser solenoidal y aun así romper la estructura de
  helicidad nula del flujo de Darcy.
- **(B) La integración.** El integrador puede derivar fuera de las superficies invariantes.

El consumidor elegido el 2026-10-17 traza partículas hacia adelante sobre v_s = k_s ∇Φ_s:

- **Frente a (A):** v_s tiene helicidad nula por construcción, pero su divergencia es O(h²), no cero.
- **Frente a (B):** no hay garantía estructural, así que la deriva hay que medirla (Regla 3).

SF-37 hace esas mediciones en el cubo periódico, con 3 semillas nuevas y umbrales pre-registrados. Las estadísticas
del cubo son de un medio periódico: no se reporta ningún α.

Notación:

- **σ_T²(x1):** varianza transversal de la pluma en cada plano.
- **var τ:** varianza del tiempo de arribo.
- **λ/h:** longitudes de correlación por celda. Acá λ = ℓ = 1/4, así que λ/h = N/4. El plano x1 = 16 equivale a 64λ.

| experimento | qué es / por qué | resultado (hechos) |
|---|---|---|
| V1, tolerancia (fuente B) | tol 1e-8 contra 1e-10, mismas partículas | 126/126 con D = 0 exacto. Las corridas son idénticas bit a bit de 1e-6 a 1e-10, porque el paso está topado por h_max = h/8. Comparar h/8 contra h/32 da MS(dx_T)/σ_T² ≤ 1,27e-13 (RMS ≤ 3,6e-7 σ_T a 64λ). |
| V2 y λ/h, grilla | el mismo continuo a N = 32…512 | V2 126/126 por la rama relativa (orden 1,86-2,16 a σ² = 1). λ/h 8 / 16 / 32, estudio 32 (N = 128). 11 celdas fallan en los candidatos chicos, 8 de ellas de la semilla 3102; en las 10 fallas sobre σ_T² la grilla gruesa da menos dispersión. **[post-hoc]** Sobre el crecimiento σ_T²(x1) − σ_T²(0,25), el error a λ/h = 32 es ≤ 3,3 %. |
| V4, divergencia de v_s | e_v = 1 − exp(−∫div v_s dt) | 45/45, orden 1,996-2,007. **[post-hoc]** A N = 128, x1 = 16: e_v 0,0065-0,035, crece como ~x1^0,6. |
| V3a, etiquetas en grilla (σ² = 1) | etiqueta en grilla interpolada contra la etiqueta de la partícula | C acepta las 3 semillas; 12/12 concordantes; RMS 6,3e-6 → 2,8e-7 → 1,9e-8. Usa el mismo integrador y la misma v_s, así que verifica consistencia de implementación, no el flujo. |
| V3b, Pollock (RT0 sobre caras de SF-19) | representación independiente del mismo tipo que usa B&dD | 54 discordantes / 9 concordantes; \|D_P\| ≤ 1,3e-3; lo que falla es la monotonía en 256 → 512. Ver P1 y P2. |
| V5, estiramiento | ajuste exponencial contra algebraico | `insufficient` (pares idénticos). **[post-hoc]** Ver P4. |
| O3, etiquetas a σ² 2,25 y 4, a ciegas | métrica robusta pre-registrada | 5 accept, 1 unresolved (σ² 4, semilla 3102). Sin el enmascarado, el error de esa semilla crece con N (0,23 → 0,44 → 0,92) y d_max se triplica por duplicación. |
| R, regresión | | 214/214 idénticas byte a byte; ctest V100 38/38. |
| D8 (estimador de SF-39) | | var x1 y d var x1/dt piden λ/h = 64 a σ² = 4. **[post-hoc]** Tasa tipo α_L con error de hasta +10,4 % a λ/h = 32. |

## I.3 Resultados post-hoc

**P1. El mecanismo de V3b está confirmado, con una corrección** (`v3b_weight_test.py`, `v3b_weight_test2.py`).

- A N = 2M = 512 cada semilla cae exactamente sobre una cara transversal de celda.
- `pollock_init_state` asigna la partícula a la celda inferior cuando la velocidad en esa cara es ≤ 0. Entonces el
  desplazamiento del centro de celda respecto de la semilla es **±h/2 según el signo de la velocidad transversal
  local**. El PR y la nota decían "+h/2 fijo por la regla de floor", y eso es inexacto.
- Con el modelo correcto, la diferencia de pesos predicha coincide con la medida: pendiente 0,91-1,00 y correlación
  0,97-0,999 en las 9 corridas.
- La parte de peso de D_P se reproduce en cada plano (por ejemplo, eps 2, semilla 3101, x1 = 16: −1,16e-3 medido
  contra −1,15e-3 predicho).
- A N = 64 y 128 el mismo modelo da pendiente 1,000. A N = 256 la diferencia residual es de segundo orden.
- Con el +h/2 fijo, a N = 512 la pendiente da entre −0,54 y +0,15: esa forma de la hipótesis queda refutada.
- Lectura: el v1 de RT0 es discontinuo a través de las caras transversales, y con N = 2M todas las semillas caen
  sobre una. Es un defecto de cómo el comparador asigna los pesos, no del consumidor ni de la convergencia. El
  veredicto sigue siendo `discordant` (RR7).

**P2. Pollock está más cerca de la referencia fina que el consumidor a igual N** (`bracket*.py`). La referencia es
σ_T² del consumidor a N = 512.

- Pollock queda más cerca que el consumidor en 55/63 celdas a N = 64 y en 56/63 a N = 128.
- Ejemplo, σ² = 4, semilla 3102, x1 = 16, error relativo a la referencia:

| N | consumidor | Pollock |
|---|---|---|
| 64 | −10,1 % | +0,7 % |
| 128 | −2,8 % | −0,3 % |

- El "exceso de Pollock" es casi todo déficit de grilla del consumidor.
- Los dos comparten SF-18/SF-19, así que esta comparación no ve el error de SF-19.

**P3. Escala absoluta, en unidades de α** (`growth.py`, `alpha_scale.py`, `exponents.py`).

- Error de grilla del consumidor en la tasa tipo α_T, a λ/h = 32:

| σ² | error |
|---|---|
| 1 | ≤ 6e-5 λ |
| 2,25 | ≤ 5e-4 λ |
| 4 | ≤ 1,5e-3 λ |

  Los tres quedan debajo del piso de 2,5e-3 que SF-39 fija para el intervalo de confianza de α_T. Pollock a
  λ/h = 32: ≤ 1,9e-4 λ.
- Exponentes de crecimiento por duplicación a N = 512: var x(t) entre 1,33 y 2,05 (≈ 2,0, balístico, en la semilla
  3102), var τ entre 1,42 y 1,97, crecimiento transversal entre 1,03 y 1,92. El cubo es superdifusivo y no fickiano
  hasta 64λ.
- La dispersión transversal la domina una sola componente por realización (σ² 4, semilla 3102: var x2/var x3 =
  0,585/0,071).

**P4. Ley de estiramiento** (`stretch.py`).

- La tasa mediana de estiramiento finito decrece con el recorrido: a σ² = 4 va de 0,24-0,28 en x1 = 2 a 0,08-0,15
  en x1 = 16.
- Las medianas son iguales a N 256 y 512; las colas no convergen (|a| máximo 5e7).
- No aparece una tasa positiva constante, pero tampoco se puede descartar. La pregunta de V5 sigue abierta.

**P5. La semilla 3102 es extrema en casi todo:**

- 8 de las 11 fallas de λ/h;
- las únicas celdas de V2 con separación > 0,1 σ_T;
- las colas de covector más grandes;
- O3 sin resolver;
- var x(t) balístico;
- flujo canalizado.

Hipótesis: es una realización muy canalizada. El λ/h salió de 3 semillas, así que es un valor típico, no una cota
para un ensemble de cientos.

## I.4 Interpretación a la luz de la teoría

- **Fuente B.** Con el tope h/8 es despreciable hasta 64λ. En SF-30, a σ² = 4, las trayectorias se decorrelacionaban
  tras 4-8 períodos, pero ahí los pasos los limitaba la tolerancia. En ~200λ la amplificación terminará
  decorrelacionándolas, así que SF-39 necesita criterios estadísticos, que ya prevé.
- **Fuente A.** El consumidor no tiene fuga helicoidal por construcción. Su error dominante es de grilla: subestima la
  dispersión con grilla gruesa y converge a O(h²).
- **Pollock contra Lester.** A esta escala (64λ, cubo periódico) no se ve una fuga (A) relevante en un método de tipo
  B&dD. Eso **no refuta** a Lester, porque su afirmación es asintótica: si el α_T físico fuera ≈ 0, incluso una fuga
  chica dominaría. Lo que cambia es el balance práctico: la ventaja del consumidor es estructural, no de precisión
  medida en σ_T².
- **Cubo periódico.** El régimen superdifusivo y anisótropo es coherente con que las líneas de corriente no se
  cierran en el toro. El cubo es un banco de verificación, no un lugar donde medir α.
- **Etiquetas en grilla a σ² = 4.** Tienen un defecto localizado que se afila con N, compatible con la hipótesis de
  una capa delgada sin resolver. Afecta a la ruta pseudo-simpléctica futura, no al consumidor.

## I.5 Rumbo propuesto (decisiones del dueño)

1. **SF-39, criterio A.** La escalera de tolerancia {1e-7, 1e-8, 1e-9} a h/8 saldría vacía igual que V1. Hay que
   reemplazarla por una escalera de h_max (h/8, h/16, h/32), o relajar el tope para que la tolerancia limite.
2. **Resolución contra dominio.** El techo es de unas 1,6e8 celdas. A λ/h = 32, la grilla 2048×256×256 cubre
   64λ × 8λ × 8λ, frente a los 204,8λ × 25,6λ × 25,6λ de B&dD. Opciones:
   - λ/h por amplitud, con dominio reducido solo a σ² = 4;
   - Pollock como estimador co-primario, clasificando α_T como físico solo si ambos coinciden;
   - autorizar la segunda GPU (más rendimiento de ensemble, no más memoria por stage).
3. **SF-38.** Corregir los pesos de las semillas de Pollock (semillas fuera de las caras, o promedio de ambos lados)
   antes de usarlo para elegir el medio A/B.
4. **SF-39.** Medir la convergencia en h del estimador longitudinal en su propio medio (por la señal de D8).
5. **Piloto de SF-39.** Medir la sensibilidad a λ/h por realización, para detectar realizaciones canalizadas.

---

# Parte II. El tracker lagrangiano, su validación y la factibilidad en el dominio de MECOM 2025

## II.1 Qué es, comparado con Par2

En MECOM cada partícula avanzaba con Par2 así:

- velocidades en las caras, a partir de la carga resuelta con volúmenes finitos y PCG con multigrilla;
- interpolación de cada componente **lineal en su propio eje**, del tipo RT0/Pollock (`legacy/par2/Geometry/FaceField.cuh:179-181`; no es trilineal, aunque la diapositiva 25 lo diga);
- un paso de Euler con dt global = h/(2|v|max), más el salto aleatorio cuando había dispersión.

El tracker cambia cuatro cosas.

1. **Flujo.** Usa el mismo tipo de solver, pero en forma afín-periódica (SF-19):
   - el medio es periódico en las tres direcciones;
   - el potencial es Φ = G·x + φ̃, con φ̃ periódico;
   - hay tres correctores, y se impone que el flujo medio sea (1, 0, 0).
2. **Representación.** Splines cúbicos (C²) de φ̃ y de ln K, y la velocidad
   **v_s = exp(s_lnK)·(G + ∇s_φ̃) = k_s∇Φ_s**.
   - El rotor vale ∇k_s × ∇Φ_s, que es perpendicular a v_s, así que **v·(∇×v) = 0 en cada punto**: helicidad nula
     exacta, la condición de Lester.
   - El precio: div v_s no es exactamente cero, sino O(h²).
3. **Integrador.**
   - Usa Dormand-Prince 5(4) adaptativo en **longitud de arco** (dx/ds = v/|v|): no se frena en zonas lentas y no
     tiene un dt global.
   - El paso está topado en h/8, y en la práctica siempre manda ese tope (unas 6 evaluaciones del campo por paso).
   - Para caer exactamente en un plano x1 = X o en un tiempo t usa el truco de Hénon.
   - Hace dos pasadas: por planos (σ_T²(x1), τ) y por tiempos (var x_k(t), el estimador de B&dD).
4. **Lo que carga cada partícula:**
   - **Etiquetas:** su punto de entrada, constante por definición. "Conservar las etiquetas" es trivial; la pregunta
     es si la trayectoria es correcta.
   - **Tiempo de viaje τ.**
   - **I = ∫div v_s dt,** de donde sale e_v = 1 − e^(−I): cuánto flujo gana o pierde su tubo de corriente.
   - **Covectores a = ∇ψ1 y b = ∇ψ2:** la orientación y el estiramiento de sus dos superficies, con a×b = v_s e^(−I).
   - La inyección es una grilla regular de 256×256 semillas, con peso igual al flujo v_s1 en cada semilla.
   - Es **advección pura**: no tiene salto aleatorio.

| | Par2 (MECOM) | Tracker lagrangiano |
|---|---|---|
| velocidad | caras, lineal por componente (tipo RT0) | k_s∇Φ_s con splines C² |
| solenoidal | sí (hasta la tolerancia del PCG) | no, O(h²), medido con e_v |
| helicidad nula | no | sí, exacta |
| integración | Euler, dt global | DP5(4) en arco, paso ≤ h/8 |
| dispersión local | sí | no |

**Qué garantiza:** que no hay fuga por helicidad y etiquetas exactas.
**Qué no garantiza, y por eso se mide:** que v_s sea el flujo de Darcy verdadero, que la divergencia sea cero y que
la integración no derive.

## II.2 Cómo se validó

**SF-36: ¿el código integra las ecuaciones que dice?** Se usaron controles con respuesta exacta (nota de SF-36).

| control | qué prueba | resultado |
|---|---|---|
| A1, k = 1 | líneas rectas | error 0 exacto |
| A2, capas k(x1) | solución analítica | posiciones 1e-16, τ 6e-11 rel.; I 7e-4 **no cumplió**, quedó como registro (enmienda del dueño) |
| A3 | estados de falla sintéticos | conteos exactos |
| B, campo de Lester 2021 | líneas cerradas vuelven al origen | 4e-11 |
| C | GPU contra integrador de host | idénticos bit a bit |
| D | covectores contra diferencias finitas del mapa de flujo | orden ≈ 2 con la escalera de δ enmendada (la pre-registrada falló en un caso pre-asintótico) |
| E | identidad a×b = v_s e^(−I) | 5e-11 en un campo analítico; sobre splines 7e-6 a 1,5e-3, **no cumplió** el pre-registro |
| F, G, I | SF-35 intacto, memoria, determinismo | ok |

**Conclusión de SF-36:** las trayectorias están bien implementadas. Los covectores e I tienen un error O(h_max²) que
no baja con la tolerancia, porque el Hessiano de un spline C² es apenas C⁰. Esa limitación es conocida y quedó en la
enmienda del dueño.

**SF-37: ¿las estadísticas de la pluma convergen, y a qué resolución?** Ver la Parte I. En resumen:

- convergen en h con orden ≈ 2;
- λ/h de producción 8 / 16 / 32 para σ² = 1 / 2,25 / 4;
- el error de integración es despreciable;
- e_v converge con orden 2;
- V1 quedó vacío;
- Pollock, el mismo tipo de representación que usa Par2, converge a las mismas estadísticas y queda más cerca a
  igual N.

## II.3 Factibilidad en el dominio de MECOM

**El dominio, leído del código legacy** (`legacy/input/*.json`, `legacy/main_transport_JSON_input.cu`,
`legacy/random_field_generation.cu:113`):

- celdas de h = 5;
- λ_legacy = 50 en la convención exp(−πr²/4λ²), que equivale a **ℓ = 56,4 y λ/h ≈ 11,3** en la convención actual y
  la de B&dD;
- transversal: 256×256 celdas = 22,7 ℓ;
- Dirichlet en x, periódico en y, z;
- 10⁴ partículas y 500 realizaciones.

La diapositiva dice L_T = 20λ y L_L = 104-512λ, pero el código da 22,7 ℓ y 91-726 ℓ.

| σ² | grilla | memoria del stage (202 B/celda) | λ/h necesario (SF-37) | error a λ/h = 10, peor semilla, cubo [post-hoc, `lh10.py`] | veredicto |
|---|---|---|---|---|---|
| 0,25 | 1024×256² | 13 GB ✓ | ≤ 8 (inferido) | — | factible después de SF-38 |
| 1 | 2048×256² | 25-27 GB ✓ (es la cantidad de celdas de 512³, medida) | 8 | σ_T² 1,9 %; tasa tipo α_L 12 % | factible, con reserva sobre α_L |
| 2,25 | 4096×256² | 51 GB ✗ (1,7× el límite) | 16 | σ_T² 9 %; α_L 44 % | no |
| 4 | 8192×256² | 101 GB ✗ (3,5×) | 32 | σ_T² 19 %; α_L 74 % | no |
| 6,25 | 8192×256² | 101 GB ✗ | no medido (> 32) | — | no |

- Con el mismo dominio físico al λ/h requerido, σ² = 2,25 pide 1,1e9 celdas (7× el límite) y σ² = 4 pide 1,8e10
  (113×).
- Los errores a λ/h = 10 vienen del cubo periódico, superdifusivo y limitado a 64λ. Probablemente son pesimistas para
  α_L en un medio aleatorio; la escalera en dominio reducido de SF-39 mediría el valor real.

**Otras restricciones:**

- **Condiciones de borde.** El tracker necesita un medio totalmente periódico (opción "a" de SF-38). Para el
  Dirichlet en x de MECOM hace falta un spline no periódico en x (SF-40, no especificado) y una condición de salida.
- **Física.** Solo advección pura y campos gaussianos. Los casos de MECOM con advección-dispersión (α = 0,5) y
  advección-difusión (Pe = 100) quedan fuera; para esos sigue Par2.
- **Costo.** Lo corregí con una revisión adversarial.
  - El stage tarda unos 2 min a 1,34e8 celdas; eso está medido, pero en el cubo.
  - La pasada por planos para 10⁴ partículas sobre 200λ tarda ~8-20 s (extrapolado).
  - Hasta ~20 000 partículas cuestan lo mismo, porque caben en una sola ola de hilos.
  - El riesgo está en la pasada por tiempos: a σ² = 4, algunas partículas recorren ~7 veces el arco medio y el kernel
    las espera.
  - Total estimado: unas 17-33 h para 500 realizaciones por σ², si la pasada por tiempos no se dispara.

**Vía para ampliar la factibilidad:**

- El tracker en sí ocupa apenas **16 B/celda** (coeficientes de los dos splines). Medido a 512³: suma 2,1 GB sobre
  una meseta de 11,1 GB.
- El pico de ~200 B/celda es de la resolución del flujo SF-19 en GPU.
- Además, el driver mantiene vivos unos 64 B/celda de arreglos que el tracker no lee.
- Si se separa la resolución del flujo del tracking, una V100 podría rastrear sobre hasta ~2e9 celdas. Opciones para
  el flujo:
  - un solo corrector con gradiente medio impuesto, como hace MECOM;
  - precisión mixta en la multigrilla;
  - resolverlo en el host;
  - usar las dos GPU, que necesita autorización.
- Con eso σ² = 2,25 a λ/h = 16 en el dominio de MECOM entra (17,6 GB de coeficientes). σ² = 4 a λ/h = 32 sigue
  necesitando un dominio más chico; por ejemplo, 64λ × 8λ × 8λ.

## II.4 Problemas en la configuración de MECOM (verificados en el código legacy)

1. **La inyección quedó uniforme, no ponderada por flujo.** La caja de inyección está en x = 10h = 50, pero el filtro
   `50 ≤ cx ≤ 50` no encuentra ningún centro de celda (47,5 y 52,5) y el código cae al caso uniforme
   (`legacy/par2/Particles/PParticles.cu:148, 358-369`). Además está a 1 λ_legacy de la entrada; B&dD inyectan
   ponderado por flujo y a ≥ 5λ.
2. **La salida en x está cerrada:** las partículas que llegan se quedan quietas (`MoveParticle.cuh`, `validX`). Como
   t_max = 1000 λ/vm supera de lejos el tiempo de llegada, la varianza a tiempos largos queda contaminada.
3. **La convención de λ difiere de la de B&dD** (ℓ = 1,13 λ_legacy) y la diapositiva no coincide con el código.
4. **α se normaliza con la velocidad nominal** vm = K_G·J y no con la media medida (`legacy/plot_script/plot.py:384-385`).
   No se cuantificó el efecto.

## II.5 Recomendación

1. Usar el tracker en σ² = 0,25 y 1, sobre 2048×256×256 con λ/h ≈ 10-11, en el medio afín-periódico, una vez hecho
   SF-38. Es la prueba directa del α_T positivo de MECOM a σ² = 1: el mismo campo con Par2 corregido, Pollock y el
   tracker.
2. Agregar a SF-39 la escalera λ/h 10 contra 20, para α_L.
3. Para σ² ≥ 2,25 hay que decidir entre:
   - separar la resolución del flujo del tracking;
   - usar dominios más chicos;
   - autorizar el uso de las dos GPU.

---

## Límites de este informe

- Todo es post-hoc y no pre-registrado; no cambia ningún veredicto de SF-37.
- P2 se basa en un solo estadístico (σ_T²) y en el cubo periódico.
- En P1 se usó el signo de v_s en la semilla como aproximación del signo de la velocidad en la cara.
- No se recalcularon V5, D1-D4 ni D8 (sus números vienen del digest).
- El costo del dominio largo y el de la pasada por tiempos con 10⁴ partículas están extrapolados.
- La configuración del esquema euleriano de MECOM no está en el repo.
- No se verificó que el campo legacy no sea periódico en y, z.
- No se buscó en B&dD el valor de α_T a σ² = 4.
- No se hace ninguna afirmación sobre α_L ni α_T, y α_T no se presupone (Regla 1).
