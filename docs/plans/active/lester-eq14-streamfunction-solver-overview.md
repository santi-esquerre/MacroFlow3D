# Solver de streamfunctions de Lester — resumen científico y numérico

## Propósito y autoridad

Este documento explica **qué problema estamos resolviendo**, **qué caso del
paper de Lester et al. (2023) queremos reproducir** y **por qué la estrategia
numérica avanza desde Picard hasta Newton–Krylov**.

Usarlo como mapa conceptual antes de trabajar en cualquier incremento del
solver. No contiene el estado de ejecución ni habilita trabajo por sí mismo:

- el estado `NEXT`, la checklist y las decisiones operativas bloqueadas viven
  en el [dashboard](lester-eq14-streamfunction-solver-plan.md);
- el fundamento científico ampliado vive en la
  [nota de teoría](../../theory/lester-2023-key-claims.md);
- los criterios de aceptación viven en
  [acceptance-gates.md](../../validation/acceptance-gates.md);
- la fuente primaria es el
  [paper de Lester et al. (2023)](../../references/Lester-2023-WRR.pdf),
  DOI [10.1029/2022WR033059](https://doi.org/10.1029/2022WR033059).

Si este resumen y el dashboard difieren en un parámetro operativo, gobierna el
dashboard. Si una interpretación científica difiere de la fuente primaria, se
debe corregir este resumen y registrar la decisión.


### Semántica de ejecución de los incrementos

La secuencia científica `SF-00 -> ... -> SF-28` sigue siendo estrictamente
secuencial **entre incrementos**. Para el único incremento habilitado por
`NEXT`, Claude Code puede descomponer el Goal en un DAG de subtareas
autocontenidas y ejecutar en paralelo solamente nodos independientes y con
alcances de escritura compatibles.

La ejecución autónoma de un incremento termina en una **pull request auditada**:
workers Sonnet implementan/corrigen en worktrees aislados, un integrador Sonnet
combina únicamente commits aprobados y el orchestrator Fable realiza las
auditorías de aceptación antes de publicar la PR. El siguiente incremento no se
habilita hasta que esa PR sea mergeada y el nuevo estado sea visible en la rama
por defecto.

## Qué queremos obtener

MacroFlow3D ya genera una conductividad escalar heterogénea, resuelve el flujo
de Darcy y transporta partículas. El nuevo componente debe construir dos
campos escalares globales `psi1`, `psi2` para un flujo estacionario, suave,
localmente isotrópico y libre de puntos de estancamiento:

```math
\mathbf v_D=-K\nabla\phi,
\qquad
\nabla\cdot\mathbf v_D=0,
```

```math
\mathbf v_\psi=\nabla\psi_1\times\nabla\psi_2\approx\mathbf v_D,
```

```math
\mathbf v_D\cdot\nabla\psi_1\approx0,
\qquad
\mathbf v_D\cdot\nabla\psi_2\approx0.
```

Las superficies de nivel de cada `psi_i` son streamsurfaces. Su intersección
define una línea de corriente, por lo que ambos campos actúan como invariantes
lagrangianos. En el régimen cubierto por Lester, esta estructura impide el
alejamiento transversal ilimitado de líneas de corriente y, en consecuencia,
la macrodispersión transversal puramente advectiva debe tender a cero.

El objetivo inmediato no es modificar el transporte ni demostrar por sí solo
la macrodispersión nula. Primero debemos construir invariantes confiables y
demostrar simultáneamente que:

1. satisfacen el sistema elíptico no lineal;
2. reconstruyen el flujo Darcy;
3. son invariantes a lo largo de ese flujo;
4. permanecen independientes fuera de zonas Darcy genuinamente lentas;
5. convergen bajo refinamiento de malla.

Un residuo algebraico pequeño sin estas comprobaciones no es una solución
aceptada.

## Matemática del problema

### De flujo Darcy a dos invariantes

Para conductividad escalar suave `K(x)>0`, el flujo Darcy es helicidad-cero:

```math
\mathbf v_D\cdot(\nabla\times\mathbf v_D)=0.
```

En ausencia de estancamiento, el régimen considerado admite dos potenciales de
Euler o streamfunctions. Lester et al. expresan sus ecuaciones acopladas como

```math
\nabla^2\psi_1-\nabla\ln K\cdot\nabla\psi_1=S_2,
\qquad
\nabla^2\psi_2-\nabla\ln K\cdot\nabla\psi_2=S_1,
```

con

```math
S_i=
\frac{
  (\mathbf B\times\nabla\psi_i)\cdot
  (\nabla\psi_1\times\nabla\psi_2)
}{
  |\nabla\psi_1\times\nabla\psi_2|^2
},
```

```math
\mathbf B=
(\nabla\psi_1\cdot\nabla)\nabla\psi_2
-(\nabla\psi_2\cdot\nabla)\nabla\psi_1.
```

La implementación evaluará la última expresión mediante productos
Hessiano-vector,

```math
\mathbf B=H(\psi_2)\nabla\psi_1-H(\psi_1)\nabla\psi_2,
```

sin almacenar los dos Hessianos completos.

### Forma divergente que resolverá MacroFlow3D

No se discretizará `grad(log K)` de forma explícita. Se usa la identidad

```math
\nabla^2\psi-\nabla\ln K\cdot\nabla\psi
=K\nabla\cdot\left(\frac{1}{K}\nabla\psi\right).
```

Definiendo

```math
q=\frac{1}{K},
\qquad
A u=-\nabla\cdot(q\nabla u),
```

la parte lineal es un operador de difusión de coeficiente variable, simétrico
y semidefinido positivo en el dominio periódico. Esto permite verificar y,
si los contratos discretos se cumplen, reutilizar PCG y multigrilla
cell-centered. La reutilización no se da por supuesta: primero se prueban signo,
coeficientes de cara, condición periódica, gauge y residuo real.

El coeficiente en una cara entre celdas `C` y `N` será

```math
q_f=\frac{2q_Cq_N}{q_C+q_N}=\frac{2}{K_C+K_N}.
```

Es la media armónica de `q`, consistente con el flujo difusivo en forma
divergente. No es el inverso de la media armónica de `K`.

### Parte afín, fluctuaciones periódicas y gauge

El benchmark tiene flujo medio en `x1`. Sólo se almacenan fluctuaciones
periódicas cell-centered:

```math
\psi_1=\bar v x_2+\widetilde\psi_1,
\qquad
\psi_2=x_3+\widetilde\psi_2,
```

```math
\langle\widetilde\psi_1\rangle=
\langle\widetilde\psi_2\rangle=0.
```

Las partes afines se incorporan como gradientes constantes

```math
\bar{\mathbf g}_1=(0,\bar v,0),
\qquad
\bar{\mathbf g}_2=(0,0,1),
```

y nunca se almacenan como campos no periódicos. Con un parámetro de homotopía
no lineal `eta`, las ecuaciones para las fluctuaciones son

```math
A\widetilde\psi_1=
\nabla\cdot(q\bar{\mathbf g}_1)-\eta qS_2,
```

```math
A\widetilde\psi_2=
\nabla\cdot(q\bar{\mathbf g}_2)-\eta qS_1.
```

Los términos afines se construyen con exactamente los mismos coeficientes de
cara que `A`. Como el operador periódico tiene un modo nulo constante, cada
lado derecho se proyecta a media cero y el gauge se mantiene proyectando
estados, iterados PCG, correcciones multigrilla, candidatos Picard/Anderson,
direcciones Newton y campos transferidos entre mallas.

### No degeneración y regularización controlada

La cantidad

```math
\mathbf c=\nabla\psi_1\times\nabla\psi_2
```

debe permanecer no nula en el régimen sin estancamiento. Durante el arranque se
regulariza únicamente el denominador de `S_i`:

```math
d_\epsilon=|\mathbf c|^2+(\epsilon v_{\mathrm{rms}})^2.
```

`epsilon` es adimensional respecto de la velocidad RMS. Se comienza en
`1e-2`, se exige llegar al menos a `1e-6` y luego se estudia `1e-8`. Cada valor
es una etapa distinta de continuación: una solución con `epsilon` fijo no se
presenta como solución de la ecuación original. Percentiles de `|c|` y su
comparación celda a celda con `|v_D|` separan una inestabilidad numérica de una
zona Darcy físicamente lenta.

## Caso de referencia del paper de Lester

La Tabla 1 y la sección 5.1 del paper definen el caso que guía el benchmark
final:

| Magnitud | Valor del paper |
|---|---:|
| Dominio | toro 3D `T^3=[0,1]^3` |
| Borde | Born–von Kármán, triplemente periódico |
| Campo | log-conductividad multi-Gaussiana, isotrópica y suave |
| Media de `ln K` | `1` |
| Varianza de `ln K` | `4` |
| Longitud de correlación `ell` | `1/16` |
| Grilla | `256^3` |
| Espaciamiento `h` | `1/256` |
| Resolución de correlación `ell/h` | `16` |
| Velocidad media | `1` en `x1` |
| Partes medias | `bar(psi1)=x2`, `bar(psi2)=x3` |
| Residuo de diferencias finitas reportado | `1e-16` |

El paper obtiene una estimación inicial resolviendo la ecuación homogénea
`S1=S2=0` con un método Krylov y luego avanza el sistema no lineal completo con
pseudo-tiempo explícito y paso variable. Finalmente usa splines cúbicos
periódicos para construir streamfunctions continuas destinadas a sus pruebas
de tracking y una integración de Stokes sobre caras para obtener velocidades
discretamente libres de divergencia.

### Qué significa “reproducir” en este repositorio

MacroFlow3D separa tres niveles de reproducción:

1. **Reproducción paramétrica:** mismo dominio, régimen periódico, estadística
   lognormal, media/varianza, `ell`, resolución y flujo medio.
2. **Reproducción del problema físico:** el flujo reconstruido por las
   streamfunctions debe coincidir con un flujo Darcy independiente, y las
   invariancias y la no degeneración deben converger con la malla.
3. **Reproducción cinemática posterior:** invariantes aceptados deben permitir
   comprobar confinamiento a streamsurfaces y ausencia de dispersión
   transversal puramente advectiva espuria.

No se promete igualdad punto a punto con la realización aleatoria publicada:
el paper no registra en su Tabla 1 una semilla ni todos los detalles necesarios
para regenerar ese campo bit a bit. El proyecto fijará una realización
periódica reproducible y usará el mismo campo continuo en `128^3` y `256^3`.

La primera aceptación del solver tampoco intenta reproducir todavía los
splines ni el tracking del paper. Compara una reconstrucción CompactMAC con el
flujo Darcy independiente; esa reconstrucción puede ser sólo aproximadamente
libre de divergencia y debe demostrar convergencia bajo refinamiento. Una
reconstrucción de caras compatible con Stokes y el consumidor de transporte se
evaluarán después de aceptar las streamfunctions.

El paper denomina al campo “multi-Gaussiano isotrópico”, pero esa tabla no fija
por sí sola una función de covarianza completa. Para el benchmark de
MacroFlow3D se adopta explícitamente la covarianza suave

```math
C_Y(r)=\sigma_Y^2\exp[-(r/\ell)^2],
\qquad Y=\ln K,
```

con generación espectral periódica y convención de media documentada. Esta es
una decisión reproducible del proyecto y no debe presentarse como una
reconstrucción bit a bit del campo del paper.

El `1e-16` del paper se registra como dato de referencia, no como tolerancia de
aceptación inicial. En doble precisión, MacroFlow3D comienza con residuo lineal
relativo `1e-10` y residuo no lineal `1e-6`, avanzando a `1e-8` sólo cuando el
error espacial y los diagnósticos físicos lo justifiquen.

## Cómo intentamos resolver la ecuación (14)

### 1. Línea base: Picard amortiguado

Picard expone el mapa no lineal de la forma más auditable. Para un estado
aceptado `Psi^n=(tilde(psi1),tilde(psi2))`:

1. reconstruir ambos gradientes incluyendo las partes afines;
2. calcular `B`, `c`, `S1` y `S2` desde el mismo estado inmutable;
3. ensamblar y proyectar los dos lados derechos;
4. resolver consecutivamente ambos bloques con PCG y la misma jerarquía MG;
5. formar un candidato pareado con relajación `omega`;
6. proyectar el candidato al gauge de media cero;
7. reevaluar el residuo acoplado y todos los guardas físicos;
8. aceptar, reducir `omega` y reintentar, o devolver un fallo estructurado.

El primer solver usa `omega=0.25` fijo. La fase siguiente reduce a la mitad una
actualización rechazada hasta `omega_min=0.01`, permite crecimiento gradual
tras aceptaciones fáciles y detecta estancamiento. El último estado aceptado no
se sobrescribe durante backtracking.

Picard es la referencia de corrección y el fallback permanente. No se elimina
cuando se incorporen métodos más rápidos.

### 2. Continuación: hacer alcanzable el problema difícil

No se supone convergencia directa en `256^3` y varianza `4`. Se usan tres ejes
de continuación y una reducción separada de la regularización, siempre con
rollback al último estado aceptado:

- **malla:** `32^3 -> 64^3 -> 128^3 -> 256^3`, prolongando sólo las
  fluctuaciones y conservando la misma realización continua;
- **heterogeneidad:** `K_lambda=exp(lambda Y)`, desde `lambda=0` hasta `1`,
  reconstruyendo `q` y la jerarquía MG por cada etapa aceptada;
- **no linealidad:** normalmente `eta=1`; si una etapa `lambda` falla, resolver
  primero las coordenadas armónicas con `eta=0` y llevar `eta` gradualmente a
  `1`;
- **regularización:** reducir `epsilon` sólo después de aceptar
  `lambda=eta=1`.

Los pasos se reducen ante fallo y crecen sólo después de etapas fáciles. Nunca
se salta un intervalo fallido ni se cambia silenciosamente la física del caso.

### 3. Anderson acceleration

Cuando Picard y la continuación ya sean robustos, Anderson acelerará el mapa
fijo acoplado. Mantendrá historia de diferencias de estado y de residuo para
`Psi=[tilde(psi1),tilde(psi2)]`, con profundidad configurable `3–8` y valor
inicial `5`.

El pequeño problema de mínimos cuadrados se resolverá con QR pivotado. Un
candidato acelerado se proyecta y atraviesa exactamente los mismos chequeos de
residuo y degeneración que Picard. Si la historia está mal condicionada o el
candidato falla, se descarta la historia y se vuelve al candidato Picard.

Anderson debe producir los mismos campos finales dentro de la tolerancia
no lineal. Si no reduce iteraciones en la suite fija, permanece deshabilitado
por defecto. Su costo principal es `4*m` campos escalares por una historia de
profundidad `m`.

### 4. Newton–Krylov matrix-free

Después de aceptar la línea base Picard/Anderson en V100, se define el residuo
acoplado de las fluctuaciones como

```math
F(\Psi)=
\begin{bmatrix}
A\widetilde\psi_1-\nabla\cdot(q\bar{\mathbf g}_1)+\eta qS_2\\
A\widetilde\psi_2-\nabla\cdot(q\bar{\mathbf g}_2)+\eta qS_1
\end{bmatrix}.
```

Newton resuelve

```math
J(\Psi)\,\delta\Psi=-F(\Psi),
```

sin ensamblar `J`. La acción sobre una dirección se aproxima reutilizando el
mismo evaluador de residuo:

```math
J(\Psi)p\approx
\frac{F(\Psi+\delta p)-F(\Psi)}{\delta}.
```

El sistema acoplado se resolverá inicialmente con GMRES reiniciado, porque el
Jacobiano no es simétrico aunque sus bloques elípticos sí lo sean. El
precondicionador es

```math
P=\operatorname{diag}(A,A),
```

aplicando dos V-cycles proyectados. Mientras esa aplicación sea fija y lineal,
GMRES estándar es correcto; FGMRES sólo se introduce si un precondicionador
variable o de precisión mixta lo requiere. El reinicio inicial será `10` para
limitar memoria en `256^3`.

Newton se activa cerca de una solución, inicialmente después de que Picard
alcance `r_F<1e-2`, y usa tolerancia lineal inexacta más line search de Armijo.
Un fallo restaura el estado, ejecuta pasos Picard de rescate, reintenta una vez
y finalmente reduce la etapa de continuación. Newton permanece opt-in hasta
demostrar equivalencia con Picard y una mejora reproducible de costo.

## Discretización y GPU previstas

La primera versión usa celdas centradas y diferencias centradas de segundo
orden. Un stencil periódico de radio uno —la unión de 19 puntos requerida por
gradientes, derivadas mixtas y productos Hessiano-vector— permite un kernel que
calcule gradientes afines, `B`, `c` y las dos fuentes sin escribir Hessianos.

La arquitectura busca:

- reutilizar el operador de difusión, PCG, MG, reducciones y buffers CUDA
  existentes sólo después de validar sus contratos;
- fusionar el cálculo de fuentes para reducir tráfico global;
- reutilizar workspaces sin allocations ni copias CPU–GPU en loops calientes;
- mantener doble precisión hasta aceptar Picard y el benchmark V100;
- medir antes de introducir precisión mixta.

En `256^3`, un campo escalar `double` ocupa 128 MiB. El objetivo Picard es
aproximadamente 3.6–4 GiB incluyendo `Y` y la velocidad Darcy. Anderson de
profundidad cinco agrega 20 campos escalares, unos 2.5 GiB. Una base GMRES
acoplada restart-10 cuesta aproximadamente 2.75 GiB antes de work vectors, por
lo que Anderson y Newton no deben mantener simultáneamente historias
innecesarias.

## Cómo sabremos que funciona

La validación avanza de operadores a física:

1. funciones trigonométricas periódicas para gradiente, difusión,
   Hessiano-vector, proyección y transferencias, con orden bajo refinamiento;
2. medio homogéneo `K=1`, donde las fluctuaciones son exactamente cero;
3. soluciones manufacturadas pequeñas para fuentes y residuo acoplado;
4. campos Gaussianos suaves de `32^3` y `64^3`, comenzando en varianzas
   `0.25`, `1`, `2.25` y `4`;
5. misma realización física en `128^3` y `256^3`, culminando en el caso de
   referencia;
6. entre 5 y 10 realizaciones seleccionadas en una malla validada para medir
   robustez.

Para cada solución aceptada se registran como mínimo:

- residuo no lineal normalizado `r_F` y residuos lineales reales;
- error `L2`, `Linf`, por componente, de magnitud y angular entre
  `grad(psi1) x grad(psi2)` y `v_D`;
- `v_D dot grad(psi_i)` normalizado;
- divergencia del flujo reconstruido y su convergencia con la malla;
- media del gauge y defectos de compatibilidad de los lados derechos;
- mínimo, media y percentiles `0.1%`, `1%`, `5%`, `50%` de `|c|`;
- población degenerada separada entre baja velocidad Darcy explicable y
  degeneración sin explicar;
- historial completo de Picard, continuación, Anderson/Newton y rechazos;
- memoria máxima, tiempo y conteos de iteraciones en V100.

La covarianza exponencial, la conductividad tensorial, `sigma_Y^2=6.25`, la
integración con transporte y la producción de macrodispersión quedan fuera del
primer benchmark aceptado. No deben usarse para decidir si la implementación
básica de la ecuación (14) es correcta.

## Estado actual y ruta de ejecución

El harness documental está instalado, pero el solver todavía no está
implementado. El dashboard indica en todo momento el único incremento que puede
comenzar. La secuencia conceptual es:

```text
contratos y tests discretos
  -> operador proyectado y reutilización PCG/MG
  -> gradientes, Hessiano-vector, fuentes y residuo
  -> diagnósticos y API
  -> control homogéneo
  -> Picard fijo y adaptativo
  -> configuración y continuación
  -> campos Gaussianos y Darcy periódico
  -> continuación de heterogeneidad y malla
  -> Anderson
  -> optimización y benchmark V100
  -> Jv matrix-free + GMRES
  -> Newton-Krylov globalizado
  -> estudio opcional de precisión mixta
```

La implementación sólo avanza cuando el incremento actual cumple su Goal,
checklist, comandos de validación, Gate 3A cuando corresponda y bitácora, pasa la
auditoría final del orchestrator y se publica como PR. El siguiente incremento
sólo queda habilitado después del merge de esa PR y cuando el estado actualizado
es visible en la rama por defecto.
