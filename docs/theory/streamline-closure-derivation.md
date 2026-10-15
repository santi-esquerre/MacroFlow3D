# Cierre de líneas de corriente en el toro y significado de la ecuación (14)

## Propósito y estatus

Este documento desarrolla en detalle la derivación que sostiene la decisión
`docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`:

1. un par de etiquetas invariantes **afín + triplemente periódico** no degenerado
   obliga a que **todas** las líneas de corriente de Darcy se cierren en el toro;
2. para un `k` gaussiano genérico no se cierran (efecto de segundo orden en la
   amplitud);
3. la ecuación (14) periódica sí tiene solución, pero es el campo de líneas
   cerradas más cercano a Darcy, no el flujo de Darcy.

Fuentes:

- `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` (sondas CPU);
- `docs/experiments/2026-10-05-sf30-streamline-closure-gate.md` (stack de producción);
- `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md` (identidad vectorial de la (14));
- `docs/decisions/2026-10-06-eq14-inlet-label-formulation.md` (etiquetas D-1, condición D-2).

Los pasos marcados **(complemento)** completan el argumento de esas fuentes. Están
derivados, no medidos, y no fueron revisados en un PR. El documento es
explicativo: no es una decisión ni un criterio de aceptación. Si contradice a las
fuentes anteriores, gobiernan las fuentes.

---

## 0. Planteo y supuestos

- Dominio: toro $T^3=[0,1]^3$.
- Medio: $k(\mathbf x)>0$ escalar, suave, triplemente periódico.
- Flujo de Darcy $\mathbf v=-k\nabla\phi$, $\nabla\cdot\mathbf v=0$, con potencial afín-periódico
  $\phi=-\mathbf G\cdot\mathbf x+\tilde\phi$ (SF-19) y flujo medio normalizado
  $\langle\mathbf v\rangle=\mathbf e_1$.
- **Supuesto clave: $v_1>0$ en todo el dominio** (sin retroflujo). Así, toda línea de
  corriente cruza cada plano $x_1=\text{cte}$ exactamente una vez y puede
  parametrizarse por $x_1$.

Objeto buscado por el paper y por el stack periódico congelado (SF-02..SF-26):

```math
\psi_1=\bar{\mathbf g}_1\cdot\mathbf x+u_1(\mathbf x),\qquad
\psi_2=\bar{\mathbf g}_2\cdot\mathbf x+u_2(\mathbf x),\qquad u_i\ \text{triplemente periódicas},
```

```math
\mathbf v\cdot\nabla\psi_i=0,\qquad
\mathbf c:=\nabla\psi_1\times\nabla\psi_2\neq 0\quad(\text{idealmente }\mathbf c=\mathbf v).
```

> **Teorema.** Si existe un par así, no degenerado, el mapa de primer retorno de la
> cara $x_1=0$ es exactamente la identidad: todas las líneas de corriente son
> cerradas.

---

## 1. Demostración

### Paso 1 — La parte afín no depende de $x_1$

Promediando la condición de invariancia sobre la celda e integrando por partes
(sin términos de borde por periodicidad, y con $\nabla\cdot\mathbf v=0$):

```math
0=\langle\mathbf v\cdot\nabla\psi_i\rangle
=\langle\mathbf v\rangle\cdot\bar{\mathbf g}_i+\langle\mathbf v\cdot\nabla u_i\rangle
=\mathbf e_1\cdot\bar{\mathbf g}_i-\langle u_i\,\nabla\cdot\mathbf v\rangle
=\mathbf e_1\cdot\bar{\mathbf g}_i .
```

Por lo tanto $\bar{\mathbf g}_i\perp\mathbf e_1$: la parte afín de cualquier etiqueta
invariante depende sólo de $(x_2,x_3)$. Además,
$\langle\mathbf c\rangle=\bar{\mathbf g}_1\times\bar{\mathbf g}_2$, porque los
términos cruzados promedian a cero y
$\nabla u_1\times\nabla u_2=\nabla\times(u_1\nabla u_2)$ también. Si
$\mathbf c=\mathbf v$, entonces $\bar{\mathbf g}_1\times\bar{\mathbf g}_2=\mathbf e_1$.

### Paso 2 — El mapa cara→etiquetas es el mismo en $x_1=0$ y $x_1=1$, y es inyectivo

Para cada plano $x_1=a$ se define

```math
F_a:(x_2,x_3)\longmapsto\big(\psi_1(a,x_2,x_3),\ \psi_2(a,x_2,x_3)\big).
```

- **Igualdad entre caras.** La parte afín no depende de $x_1$ (paso 1) y $u_i$ tiene
  período 1 en $x_1$. Entonces $F_0\equiv F_1$.
- **Difeomorfismo local.** El jacobiano es

  ```math
  \det\frac{\partial(\psi_1,\psi_2)}{\partial(x_2,x_3)}=(\nabla\psi_1\times\nabla\psi_2)\cdot\mathbf e_1=c_1 .
  ```

  Si $\mathbf c=\mathbf v$, entonces $c_1=v_1>0$. En general $\mathbf v=\mu\,\mathbf c$,
  porque $\mathbf v$ es ortogonal a $\nabla\psi_1$ y a $\nabla\psi_2$; $\mu$ tiene
  signo fijo por continuidad, y $c_1$ también.
- **Inyectividad global.** $F_a(\mathbf x)=M\mathbf x+(\text{acotado})$, donde $M$ es
  la matriz de filas $\bar{\mathbf g}_1,\bar{\mathbf g}_2$ y
  $\det M=\langle c_1\rangle\neq0$. Un difeomorfismo local de $\mathbb R^2$ que es
  una perturbación acotada de un isomorfismo lineal es propio. Por el teorema de
  Hadamard es entonces un difeomorfismo global, en particular inyectivo.
  Formulación equivalente sobre el toro: $F$ induce un mapa $T^2\to T^2$ de grado
  $\det M=1$ que es difeomorfismo local, y un recubrimiento de grado 1 es biyectivo.

### Paso 3 — La conservación de etiquetas fuerza el retorno a la identidad

Sea una línea que parte de $p=(0,x_2,x_3)$. Como $v_1\ge\min v_1>0$, llega al
plano $x_1=1$ en un punto $q=(1,y_2,y_3)$. Las etiquetas se conservan, así que

```math
F_1(\mathbf y)=\psi(q)=\psi(p)=F_0(\mathbf x)=F_1(\mathbf x)\ \Longrightarrow\ \mathbf y=\mathbf x .
```

El desplazamiento transversal tras un período es **exactamente cero**, no sólo
módulo la red. Cada línea de corriente es una curva cerrada del toro de tipo
$(1,0,0)$. $\blacksquare$

### Corolarios

1. **$D_T=0$ por construcción en el sustituto periódico.** Si el par existe, el
   desplazamiento transversal es nulo en cada período. El cero de Lester (2023) §5
   es propiedad de la clase de campos impuesta, no una medición sobre Darcy.
2. **Recíproco (complemento).** Si el retorno es la identidad, se toman etiquetas de
   entrada con el flujo correcto (D-1, §6) y se transportan a lo largo del flujo.
   Como cada línea vuelve al mismo punto, las etiquetas toman en $x_1=1$ los mismos
   valores que en $x_1=0$, es decir, son afín + periódicas. Con $v_1>0$:
   **existe el par $\iff$ el retorno es la identidad.** Por eso el mapa de retorno
   es el oráculo exacto de existencia: usa sólo el solve de Darcy y un integrador,
   y ninguna etiqueta.

---

## 2. Por qué un $k$ genérico no cierra: segundo orden en la amplitud

Con $k=e^{\varepsilon f}$, expandimos
$\mathbf v=\mathbf e_1+\varepsilon\mathbf v^{(1)}+\varepsilon^2\mathbf v^{(2)}+\dots$

**Orden 1.**

```math
\mathbf v^{(1)}=f\,\mathbf e_1-\nabla\phi_1,\qquad \Delta\phi_1=\partial_1 f .
```

En Fourier, $\hat\phi_1(\mathbf m)\propto m_1\hat f(\mathbf m)$, así que $\phi_1$
**no tiene modos con $m_1=0$** y su promedio en $x_1$ es nulo. El desplazamiento de
primer orden es el promedio en $x_1$ de $\mathbf v_\perp^{(1)}$ sobre la recta
$(x_2,x_3)$ fija:

```math
\boldsymbol\delta^{(1)}=\int_0^1\mathbf v_\perp^{(1)}\,dx_1=-\nabla_\perp\!\int_0^1\phi_1\,dx_1=0 .
```

El primer orden se anula idénticamente, para cualquier $f$.

**Orden 2 (complemento, estructura cualitativa).** $\boldsymbol\delta^{(2)}$ reúne
productos de campos de primer orden: $f\,\mathbf v^{(1)}$, la desviación de primer
orden de la trayectoria multiplicada por el gradiente de $\mathbf v^{(1)}$, el
factor $1/v_1$, y el modo $m_1=0$ de $\phi_2$. Un producto de dos modos con
$m_1$ y $m_1'=-m_1$ produce un modo con $m_1=0$, que **sobrevive al promedio en
$x_1$**. Salvo una simetría, nada cancela esas contribuciones, y el desplazamiento
escala como $\varepsilon^2$.

**Medido.** La ley $\varepsilon^2$ y la independencia de la resolución se cumplen:

| fuente | caso | resultado |
|---|---|---|
| sondas CPU (02-10) | `gauss`, $\varepsilon$ = 0.25, 0.5, 1 | RMS 4.66e-3, 1.99e-2, 8.56e-2 (cocientes 4.27, 4.30) |
| sondas CPU | `gauss`, $\varepsilon=0.5$, 16³ vs 24³ | 1.989e-2 en ambas |
| sondas CPU | `generic3d` | cocientes 3.99, 4.00 |
| SF-30 (producción) | $R(\sigma^2=1)/R(\sigma^2=0.25)$ | 3.84–4.35 |
| SF-30 | $R$ de 128³ a 256³ | cambia 0.04–0.29 % |
| SF-30 | $R$ frente al control 2D (256³) | 34× a 884× |
| SF-30 | $(4,\,1/16,\,256^3)$, semillas 3001–3003 | $R$ = 0.135, 0.161, 0.155 por período |

Que $R$ no dependa de la malla descarta un artefacto de discretización: es una
propiedad del continuo.

---

## 3. Por qué cierran los controles

### Control 2D

Si $k$ no depende de $x_3$, entonces $v_3=0$ y el flujo es plano. En 2D todo campo
de divergencia nula periódico con flujo medio $\mathbf e_1$ tiene función de
corriente $\psi_1=x_2+\text{periódica}$, y $\psi_2=x_3$. El par existe y el flujo
cierra. Es la versión exacta del argumento de Beaudoin y de Dreuzy (2013) sobre el
confinamiento en tubos en 2D. Discretamente, con el flujo interpolado por spline,
el cierre es sólo $O(h^2)$ (SF-30: órdenes 1.99 y 2.00).

### Simetría especular de Lester (2021) (complemento: demostración)

El campo de la ec. (3.1) de Lester (2021) cumple
$f(\tfrac12-x_1,x_2,x_3)=f(x_1,x_2,x_3)$. Sea $R:x_1\mapsto\tfrac12-x_1$.

1. Por unicidad del solve de Darcy con gradiente medio fijo, $-\phi\circ R=\phi+\text{cte}$.
   De ahí $\mathbf v(R\mathbf x)=\big(v_1(\mathbf x),\,-\mathbf v_\perp(\mathbf x)\big)$.
2. Si $\mathbf x(t)$ es trayectoria, también lo es $\mathbf y(t)=R\,\mathbf x(-t)$:
   el sistema es **reversible**.
3. Sea $A$ el transporte del plano $x_1=\tfrac14$ al plano $\tfrac34$, y $B$ el del
   plano $\tfrac34$ al $\tfrac54\equiv\tfrac14$. La reversibilidad, más la
   periodicidad, da $B=A^{-1}$.
4. Por lo tanto el retorno es $P=B\circ A=\mathrm{id}$.

$R$ lleva centros de celda a centros de celda, así que la simetría también es
exacta en la malla y `lester2021` cierra a nivel de redondeo (sondas 5e-15; SF-30
≤ 3.5e-11). Rompiendo la simetría con una fase (`lester_brk`) se obtiene 8.7e-3 por
período. **La única validación publicada de "velocidad por streamfunctions =
velocidad por potencial" usa un campo que cierra por simetría, y no se transfiere
a campos aleatorios.**

---

## 4. Qué resuelve la ecuación (14)

Se define

```math
\mathbf W:=\nabla\times\mathbf c-\nabla\ln k\times\mathbf c=k\,\nabla\times(\mathbf c/k).
```

$\mathbf c$ es un flujo de Darcy exactamente cuando $\mathbf W=0$: $\mathbf c/k$ es un
gradiente, dadas las condiciones de flujo medio. La identidad vectorial de la
decisión del 2026-09-30, con $L_i=\Delta\psi_i-\nabla\ln k\cdot\nabla\psi_i$ y
$\mathbf B=H(\psi_2)\nabla\psi_1-H(\psi_1)\nabla\psi_2$, es

```math
\mathbf W=\nabla\psi_1\,L_2-\nabla\psi_2\,L_1-\mathbf B .
```

Multiplicando vectorialmente por $\nabla\psi_i$ y proyectando sobre $\mathbf c$:

```math
L_i=S_i\iff \mathbf W\cdot(\nabla\psi_i\times\mathbf c)=0\qquad(i=1,2).
```

Como $\nabla\psi_1\times\mathbf c$ y $\nabla\psi_2\times\mathbf c$ generan el plano
perpendicular a $\mathbf c$:

> **(14) $\iff\mathbf W\parallel\mathbf c$.** La ecuación impone las dos componentes
> de $\nabla\times(\mathbf c/k)=0$ transversales a $\mathbf c$, pero **no** la
> componente a lo largo de $\mathbf c$.

Esa componente faltante es
$\mathbf W\cdot\mathbf c=\mathbf c\cdot\nabla\times\mathbf c=-\mathbf B\cdot\mathbf c$,
la **helicidad de $\mathbf c$**. Un campo $\nabla\psi_1\times\nabla\psi_2$ no tiene
por qué ser libre de helicidad; el flujo de Darcy sí lo es
($\mathbf v\cdot\nabla\times\mathbf v=\mathbf v\cdot(\nabla\ln k\times\mathbf v)=0$).
Medido en las sondas (`gauss`, $\varepsilon=0.25$): la parte de
$\nabla\times(\mathbf c/k)$ transversal a $\mathbf c$ vale 6.5e-4 y 2.4e-5 (baja con
la malla), y la parte a lo largo de $\mathbf c$ vale 3.7e-2 (no baja).

### Interpretación variacional

Sea la disipación $E[\psi_1,\psi_2]=\tfrac12\langle|\mathbf c|^2/k\rangle$. Para
variaciones periódicas $\delta\psi_i$:

```math
\delta E=\big\langle(\mathbf c/k)\cdot(\nabla\delta\psi_1\times\nabla\psi_2+\nabla\psi_1\times\nabla\delta\psi_2)\big\rangle
=\big\langle\delta\psi_1\,\nabla\psi_2\cdot\nabla\times(\mathbf c/k)\big\rangle-\big\langle\delta\psi_2\,\nabla\psi_1\cdot\nabla\times(\mathbf c/k)\big\rangle .
```

Las ecuaciones de Euler–Lagrange son $\nabla\psi_i\cdot\nabla\times(\mathbf c/k)=0$, es
decir $\nabla\times(\mathbf c/k)\parallel\mathbf c$: **exactamente la (14)**.

Para cualquier $\mathbf c$ de divergencia nula con el mismo flujo medio que
$\mathbf v_D$ (complemento: verificación):

```math
\langle\mathbf c\cdot\mathbf v_D/k\rangle=\langle\mathbf c\cdot(\mathbf G-\nabla\tilde\phi)\rangle=\mathbf G\cdot\langle\mathbf c\rangle=\mathbf G\cdot\langle\mathbf v_D\rangle=\langle|\mathbf v_D|^2/k\rangle
```

```math
\Longrightarrow\quad \tfrac12\big\langle|\mathbf c-\mathbf v_D|^2/k\big\rangle=E[\mathbf c]-E[\mathbf v_D].
```

Es el principio de mínima disipación (Thomson): $\mathbf v_D$ minimiza $E$ entre los
campos de divergencia nula con ese flujo medio. Combinado con §1:

- la clase admisible $\{\nabla\psi_1\times\nabla\psi_2:\ \psi\ \text{afín + periódico}\}$
  es **la clase de campos de líneas cerradas**;
- resolver la (14) periódica es buscar puntos críticos de la distancia a
  $\mathbf v_D$, en la norma de energía $1/k$, **dentro de esa clase**;
- si $\mathbf v_D$ cierra, la solución es $\mathbf v_D$; si no, es el campo de líneas
  cerradas más cercano, a distancia $O(\varepsilon^2)$, independiente de la malla y
  del solver.

Medido (sondas CPU): `generic3d` da $e_v$ = 3.536e-3 y 1.414e-2 en
$\varepsilon$ = 0.125 y 0.25 (cociente 4.0), igual a cuatro dígitos en dos mallas,
con $r_F$ hasta 1e-11. Para `gauss` en $\varepsilon=0.5$, la ley $\varepsilon^2$
predice 1.96e-2. **Ese es el plateau de $e_v\approx2.5\,\%$ de SF-21/SF-25.**

---

## 5. Helicidad nula no implica cierre global

- **Existencia local de dos invariantes.** Es trivial: vale para cualquier campo sin
  ceros (teorema de flow-box), haya o no helicidad.
- **Helicidad nula de $\mathbf v=-k\nabla\phi$.** Significa que $\mathbf v$ es normal
  a una familia de superficies, las equipotenciales (integrabilidad de Frobenius del
  plano $\perp\mathbf v$). No dice que las líneas de corriente vivan en superficies
  invariantes globales con fluctuaciones acotadas.

El paso de "helicidad nula" a "dos invariantes globales con fluctuaciones acotadas"
es el que falla en la celda periódica genérica. El §1 de
`lester-2023-key-claims.md` (helicidad nula) se mantiene; lo refutado es la parte
global.

---

## 6. Consecuencias para el transporte y para la formulación con etiquetas de entrada

**Iteración del retorno.** El retorno $P:T^2\to T^2$ preserva el flujo (la medida
$v_1\,dx_2\,dx_3$). Tras $n$ períodos, el desplazamiento transversal es

```math
\mathbf X_n(\mathbf x)=\sum_{j=0}^{n-1}\boldsymbol\delta\big(P^j\mathbf x\big).
```

- Con $P=\mathrm{id}$, $\mathbf X_n=0$ está acotado: es la hipótesis de
  "fluctuaciones acotadas" de Lester (2023) §4.
- Con $P\neq\mathrm{id}$ puede crecer sin límite. Sondas: `two_mode` crece
  linealmente en $n$; en `gauss`, $\mathrm{var}(dx_3)$ pasa de 6e-3 a 4.97 entre
  $n=1$ y $40$, mientras $\mathrm{var}(dx_2)$ se mantiene acotada. SF-30: a
  $(4,1/16)$ la varianza crece casi linealmente en $n$; a $\sigma^2\ge1$ las
  trayectorias individuales no están resueltas más allá de $n\approx4$–$32$.
- Los $D_{22}$, $D_{33}$ por re-inyección (ec. 36 del paper) son números de
  protocolo de la celda periódica, **no** coeficientes de macrodispersión de un
  medio aleatorio (decisión O3).

**Slab no periódico en $x_1$ (SF-29, SF-33, SF-35).** Las etiquetas de entrada D-1
son

```math
Q(x_3)=\int_0^1 v_1(0,s,x_3)\,ds,\qquad
\psi_2=\int_0^{x_3}Q,\qquad
\psi_1=\frac{1}{Q(x_3)}\int_0^{x_2}v_1(0,s,x_3)\,ds ,
```

con jacobiano de cara $c_1=v_1$ por construcción (el flujo coincide en la entrada).
Transportadas por el flujo, en la salida valen

```math
\psi(1,\mathbf y)=\psi_{\text{in}}\big(P^{-1}\mathbf y\big)\neq\psi_{\text{in}}(\mathbf y).
```

La diferencia entre etiquetas de salida y de entrada es el mapa de retorno. Por eso
$x_1$ no puede ser periódico, la condición de salida D-2 se impone sobre
$\mathbf c\times\mathbf e_1$ y no sobre los valores de $\psi$, y SF-34 acepta las
etiquetas de SF-35 comparando su retorno con el medido por SF-30.

---

## 7. Estatus de cada afirmación

| afirmación | estatus |
|---|---|
| Par afín + periódico no degenerado con $v_1>0$ ⇒ retorno = identidad | derivación (nota 02-10; pasos detallados aquí) |
| Recíproco (retorno = id ⇒ el par existe) | derivación (complemento) |
| Primer orden nulo; efecto $O(\varepsilon^2)$ | primer orden derivado; segundo orden medido (cocientes ≈ 4) |
| Cierre del control 2D y del campo especular | derivación y medición |
| (14) ⟺ $\nabla\times(\mathbf c/k)\parallel\mathbf c$ ⟺ Euler–Lagrange de $E$; identidad de distancia | derivación |
| Darcy gaussiano no cierra en producción, incluidos los parámetros del paper | medido (SF-30, 6/6 casos, regla prerregistrada) |
| Retroflujo a $\sigma^2=4$ (fracción de volumen ≤ 1.2e-3), donde el argumento no aplica | medido. Las líneas sin retroflujo dan el mismo $R$ a 0.8 %, así que la conclusión no depende de ellas. |
| "La solución de (14) es el campo cerrado *más cercano*" | estrictamente, un punto crítico de la distancia; que sea el mínimo no está probado |
| $\alpha_T$ en un medio aleatorio no periódico | **abierto** |
