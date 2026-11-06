# HM fit — Fase 2: pre-registro de la campaña de cobertura (etapa 2.2)

> Documento congelado antes de la corrida completa. Los umbrales y reglas de decisión
> no se cambian después de ver resultados; cualquier cambio posterior se registra como
> **desviación** en `docs/STABILITY_PHASE2_REPORT.md` (sección "Desviaciones del
> pre-registro"), con fecha y motivo.

- Fecha de registro: 2026-11-05.
- Rejilla congelada: `validation/configs/coverage/primary_grid.json`
  (SHA-256 `caa6ae6ca5dff5eded26e91c83d3674c8fa444404eda989d61b81de1eeebf518`, 326 celdas;
  generada con `python -m validation.coverage_campaign grid --write …`).
  Tras la enmienda 1 (§9) el archivo tiene 327 celdas (SHA-256
  `f0723daec52f9c084a6adadce9eb4704c786395bc58c249a24d3bf968a86d79c`); las 326 originales
  no cambian, solo se añade una celda de diagnóstico.
- Código: `validation/coverage_campaign.py` en el commit que añade este documento.
- Semilla base `BASE_SEED = 20261105`; la réplica *r* de la celda *c* usa
  `SeedSequence([20261105, crc32(seed_id(c)), r])`; los datos y cada método reciben
  semillas hijas independientes. Las pasadas `~fr` comparten `seed_id` con la pasada
  principal: **los mismos conjuntos de datos** se analizan con el full-refit.

## 1. Relación con el trabajo previo

Las campañas de `docs/STATISTICAL_VALIDATION.md` (confirm-v1) usan una capa de
validación propia (argmin escalar y re-ajuste fuera del procesador). Esta campaña es
distinta: **cada réplica pasa por los mismos puntos de entrada que la GUI**, para que
sirva de prueba de regresión de la unificación del motor de incertidumbre (Fase 3):

| Paso | Punto de entrada |
|---|---|
| Ajuste | `hmfit_core.api.run_spectroscopy_fit`, `run_nmr_fit`, `run_acid_base_fit`; `hmfit_gui_qt.tabs.kinetics_tab._run_kinetics_fit` (objetivos de `FitWorker`) |
| Contexto de errores | `SpectroscopyTab/NMRTab/KineticsTab._update_errors_context_from_result`; `errors_context` del resultado ácido-base |
| Re-ajuste (full-refit) | `SpectroscopyTab._refit_from_data`, `NMRTab._refit_from_data`, `refit_fn` de cinética, `_refit_from_context` ácido-base |
| Errores | `ModelOptPlotsWidget._compute_errors_from_context` (UV-Vis, RMN), `KineticsModelOptPlotsWidget._compute_errors_from_context`, `compute_errors_acid_base_from_context` |

## 2. Métodos

| Clave | Método GUI | Remuestreo |
|---|---|---|
| `analytic` | Analytical covariance (IC t de Student, ν residual) | — |
| `lin_row`, `lin_el` | Bootstrap (linearized + wild) | por punto (`row`) / por elemento (`element`) |
| `os_row`, `os_el` | Bootstrap (one-step LM), λ = 10⁻³ | ídem |
| `fr_row`, `fr_el` | Bootstrap (full refit, audit), `max_iter` = 30, `tol` = 10⁻⁸, fallos omitidos | ídem |

Pesos del wild bootstrap (valor por defecto de la GUI, `hmfit_core.utils.errors.wild_multipliers`):
**Rademacher**, ±1 con probabilidad 1/2 cada uno (Mammen disponible pero no evaluado).
Con `row`, un multiplicador por punto de titulación/tiempo compartido por todas las λ o
señales; con `element`, uno independiente por observación. Los residuos se reescalan por
el factor global tipo HC1 √(n/(n − p_total)). Los IC bootstrap son percentiles 2.5–97.5.

Estimadores UV-Vis (etapa 2.3): `soft_penalty` μ = 10⁻² (valor por defecto de la GUI);
`unconstrained` = `soft_penalty` con μ = 0 (alcanzable desde el campo μ de la GUI);
`nnls_hard`. **El estimador (b) del encargo — restringido con Jacobiano condicional al
conjunto activo — no existe en el código** y no se implementa en esta fase (regla 5);
se compara (a) sin restricción, el restringido con la covarianza analítica actual y (c)
el restringido con full-refit.

## 3. Rejilla

Valores verdaderos: UV-Vis/RMN 1:1 log K = 3.0; 1:2 log K₁ = 3.0, log K₂ = 2.5; H₀ = 1 mM.
Saturación (fracción de anfitrión complejado, primer punto sin huésped): `rec` 0.20–0.80,
`poor` 0.05–0.45 (máximo < 50 %). Ácido-base (convención HM fit, pKa₁ = log β₁ es el sitio
más básico, por tanto pKa₁ > pKa₂): `sep` 8.0/4.0, `close` 5.0/4.4, `fixed` 8.0/4.0 con
pKa₂ fijo; pH 2–10. Potenciometría: 5.0 (1 pKa) y 7.5/4.0 (2 pKa), C = 10 mM.
Cinética: A→B→C k = 1.0/0.7 (`consec_sim`) y 1.0/0.1 (`consec_diff`); A+B→C k = 2.0.

| Técnica | Diseños | Puntos | Ruido |
|---|---|---|---|
| UV-Vis (31 λ) | `11zero` (bandas truncadas: ε = 0 exacto en muchas λ), `11pos` (todas ε > 0), `12zero` | 8, 10, 12, 15, 20, 30 | `iid` σ = 10⁻³; `hetero` σ = 10⁻³(0.5 + 1.5·|y|/max|y|); `offset` desplazamiento por espectro N(0, 10⁻³) + iid 5·10⁻⁴ |
| RMN ¹H (2 señales) | `11`, `12` | 8, 10, 12, 15, 20 | `iid1` σ = 0.001 ppm; `iid5` σ = 0.005 ppm; `point` error por punto N(0, 0.003) + iid 0.001 |
| Ácido-base RMN (2 núcleos) y espectroscopía (8 λ) | `sep`, `close`, `fixed` | 10, 15, 25, 40 | `iid` (3·10⁻³ ppm / 10⁻³); `offset` por espectro |
| Potenciometría | 1 y 2 pKa | 30, 60 | iid en pH, σ = 0.005 |
| Cinética (12 canales) | `consec_sim`, `consec_diff`, `second` | 50, 200 tiempos | `iid` σ = 5·10⁻³; `drift` deriva lineal por canal N(0, 5·10⁻³) |

Escenario de diagnóstico (sin expectativa de cobertura): concentración del titulante con
error sistemático de ±3 % (UV-Vis 1:1 y RMN 1:1, 12 puntos, N = 200).

## 4. Métricas y umbrales (por celda × método × parámetro)

Métricas: sesgo y sesgo/SD; SD empírica; SE medio y mediano; SE/SD con IC percentil
(2000 remuestreos de réplicas); cobertura del IC95 con IC de Wilson; fallos por la
izquierda y por la derecha; ancho mediano; tasa de fallos de ajuste; réplicas bootstrap
fallidas; tiempo medio. Las réplicas cuyo ajuste falla se cuentan como fallo y se
excluyen del resto de métricas (condicionadas al éxito).

- **Calibrado**: cobertura en [0.93, 0.97] y SE/SD en [0.85, 1.15].
- **Conservador**: cobertura > 0.97 o SE/SD > 1.15 (y no anticonservador).
- **Anticonservador**: cobertura < 0.93 o SE/SD < 0.85.
- **Sesgo relevante**: |sesgo|/SD > 0.25 (se reporta aunque la cobertura sea aceptable).
- **IC asimétrico**: ≥ 10 fallos y el lado mayor ≥ 3× el menor.

Agregación por técnica (para las reglas de 2.3/2.4). *Rejilla realista* = todas las celdas
primarias de la técnica con ruido `iid`/`hetero`/`iid1`/`iid5` y ambas saturaciones
(la saturación pobre es un error frecuente). Las celdas con ruido correlacionado
(`offset`, `point`, `drift`) se reportan aparte. Un método está **calibrado en la rejilla
realista** si en ≥ 75 % de sus pares celda-parámetro es "calibrado" y en ninguno es
anticonservador con evidencia (Wilson superior < 0.95 o IC superior de SE/SD < 1.0).
Es **anticonservador en la rejilla realista** si lo es con evidencia en algún par.

## 5. Tamaños y presupuesto de cómputo

Piloto de tiempos (N = 2 por celda, B = 1000, B_refit = 500, 24 procesos en 32 núcleos):
por réplica, ajuste 1–2 s; analítico ≤ 0.3 s; linealizado ≈ 0 s; one-step 3–10 s por
variante; **full-refit 50–210 s por variante** (UV-Vis/RMN/potenciometría/cinética),
2–3 s en ácido-base. Con N = 500 y B_refit = 500 en toda la rejilla, el full-refit
costaría ≈ 4 400 h-CPU (≈ 6 días en esta máquina): no es razonable.

Tamaños pre-registrados:

- Métodos baratos (`analytic`, `lin_*`, `os_*`): **N = 500, B = 1000** en todas las celdas
  primarias (semiancho Wilson ≈ 1.9 % a cobertura 0.95).
- Full-refit: **N = 200, B_refit = 200** en una sub-rejilla pre-registrada (pasadas `~fr`:
  UV-Vis `11zero` y RMN `11` con 8/12/15 puntos, ruido iid y correlacionado, ambas
  saturaciones; 2.3; potenciometría; cinética con 50 tiempos); ácido-base con N = 500,
  B_refit = 500. Esto es una **desviación declarada** respecto del encargo (N ≥ 500,
  B_refit ≥ 500) por presupuesto; semiancho Wilson ≈ 3.0 % a N = 200.

Orden de ejecución (prioridad del encargo): tier 1 (UV-Vis y RMN 1:1, 8–15 puntos,
métodos baratos) → tier 2 (sub-rejilla full-refit) → tier 3 (2.3, ácido-base,
potenciometría, cinética) → tier 4 (resto de UV-Vis y RMN) → diagnóstico. Lo que no
termine se reporta como no ejecutado; no se extrapola.

## 6. Reglas de decisión (copiadas del encargo, sin cambios)

**2.3 absortividades.** Estimador por defecto = el de menor RMSE en log K entre los que
queden calibrados en la rejilla realista; si ninguno, el menos anticonservador, con
advertencia; si la evidencia no es clara, se deja el valor actual y se presentan las
opciones al autor.

**2.4 bootstrap por defecto.** Por técnica, el método más barato calibrado en la rejilla
realista; si ninguno en el régimen de pocos bloques, el menos anticonservador; nunca uno
anticonservador si existe una alternativa conservadora; sin respuesta inequívoca, no se
cambia el valor por defecto. Orden de coste: analítico < linealizado < one-step < full-refit.

## 7. Pilotos realizados antes del registro

Humo (N = 3, B = 30) y piloto de tiempos (N = 2, ver §5). No se examinó cobertura (con
N ≤ 3 no es informativa). El humo reveló que el generador ácido-base usaba pKa = [4, 8],
que en la convención de HM fit describe un sistema cooperativo en el que solo
pKa₁ + pKa₂ es identificable; el diseño se corrigió a pKa₁ > pKa₂ **antes** de este registro.

## 8. Comandos

```sh
export QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
G=validation/configs/coverage/primary_grid.json
python -m validation.coverage_campaign run $G runs/coverage --tier 1 --workers 30
python -m validation.coverage_campaign run $G runs/coverage --tier 2 --workers 30
python -m validation.coverage_campaign run $G runs/coverage --tier 3 --workers 30
python -m validation.coverage_campaign run $G runs/coverage --tier 4 --workers 30
python -m validation.coverage_campaign run $G runs/coverage --tier 9 --workers 30   # diagnóstico
python -m validation.coverage_campaign summarize runs/coverage --csv validation/results/coverage/summary.csv
```

## 9. Enmienda 1 (2026-11-05, antes de leer resultados)

Registrada a petición del autor, con la campaña en pausa. **Estado de los datos al
registrarla:** solo existen réplicas del tier 1 de UV-Vis (unas 23 400 de 36 000); no se ha
ejecutado ninguna celda de RMN, ácido-base, potenciometría, cinética, full-refit ni
diagnóstico. Durante la supervisión del tier 1 se vio un resumen parcial de algunas celdas
UV-Vis (SE/SD ≈ 1,8–2 con el estimador por defecto); se declara aquí. Ninguna regla de
esta enmienda depende de esos números: se aplica igual a todas las celdas.

**9.1 Verificación del orden de pKa (sin desviación).** En el commit del registro
(`05ca856`) todas las celdas ácido-base usan pKa₁ > pKa₂ en la convención de HM fit
(pKa₁ = log β₁, sitio más básico): `sep` = [8,0; 4,0], `close` = [5,0; 4,4],
`fixed` = [8,0; 4,0] con pKa₂ fijo, potenciometría [5,0] y [7,5; 4,0]. No hace falta
corregir ninguna celda. Se añade un caso degenerado **explícito de diagnóstico**,
`diag-ab_nmr-inverted-25-iid` (RMN ácido-base, pKa = [4,0; 8,0], 25 puntos, ruido iid,
N = 200, métodos `analytic`, `lin_row`, `lin_el`, `fr_row`, tier 9). No tiene
expectativa de cobertura para cada pKa por separado (en ese régimen los datos determinan
sobre todo la suma) y no entra en ninguna agregación ni regla de decisión.

**9.2 Categoría "indeterminado".** Para cada par celda × método × parámetro se reporta,
además de la clase pre-registrada (§4, por estimación puntual, que sigue siendo la
clasificación primaria), la columna `class_ci`:

- `indeterminate` si el IC de Wilson al 95 % de la cobertura **contiene** 0,93 o 0,97 (las
  réplicas no permiten decir de qué lado del umbral está la celda);
- en otro caso, la clase pre-registrada.

Consecuencia que se declara de antemano: con cobertura real 0,95 el IC de Wilson es
[0,910; 0,973] a N = 200 y [0,927; 0,966] a N = 500; **una celda perfectamente calibrada
es casi siempre "indeterminada"** con estos tamaños. La categoría separa sobre todo las
celdas claramente anticonservadoras (Wilson superior < 0,93) o claramente conservadoras
(Wilson inferior > 0,97) de las que la campaña no puede resolver. Para confirmar
"calibrado" con el IC dentro de [0,93; 0,97] harían falta N ≳ 550 réplicas.

Agregado (§4): el veredicto pre-registrado se mantiene. Se añade `verdict_ci`, que
recalcula el veredicto en el peor caso (ningún par indeterminado cuenta como calibrado)
y en el mejor (cuentan como calibrados los indeterminados con SE/SD en [0,85; 1,15]). Si
ambos coinciden, ese es `verdict_ci`; si no, `indeterminate`. El criterio de
"anticonservador con evidencia" no cambia.

**9.3 Construcción de los intervalos.** El IC del método analítico es t de Student con
los grados de libertad residuales (θ̂ ± t₀.₉₇₅,ν · SE). En **todos los bootstrap**
(linealizado, one-step y full-refit) el IC95 es el de **percentiles 2,5 y 97,5** de las
réplicas bootstrap, y el SE es la desviación estándar de esas réplicas; **no** se usa
t con SE bootstrap. La cobertura medida es la del IC que muestra la GUI. Con
B_refit = 200 cada extremo percentil descansa en unas 5 réplicas por cola: esa
variabilidad Monte Carlo adicional forma parte de lo que se mide y no se corrige.

**9.4 Ampliación de N dirigida (solo etapa 2.3).** Si la elección del estimador de
absortividades (§6) depende de una celda cuyo `class_ci` es `indeterminate` (es decir,
si cambiar la clase de esa celda cambiaría el `verdict_ci` de un estimador o el
estimador elegido), se amplía N **solo en esa celda**: a N = 1500 para métodos baratos y
a N = 600 para el full-refit, continuando los índices de réplica con las mismas semillas
(`run … --select '^<celda>$' --n-rep <N>`). Se evalúa una sola vez; no hay más
iteraciones. Si sigue indeterminada, se aplica la cláusula "evidencia no clara" de §6
(se mantiene el valor actual y se presentan las opciones al autor).

