# Validación estadística de los métodos de incertidumbre (Etapa F)

> Documento de diseño y resultados de la comparación de métodos de
> incertidumbre sobre datos sintéticos UV-Vis 1:1. Los resultados numéricos
> provienen exclusivamente de las campañas commitadas en
> `validation/results/unc/`; el diseño se congeló antes de ejecutar cada
> campaña y queda registrado en `validation/configs/unc/`.

## 1. Métodos comparados

Todos se evalúan sobre el **mismo** ajuste (mismo dataset, mismo log K
estimado), de modo que las comparaciones son pareadas.

| Método | Definición | Supuestos |
|---|---|---|
| `analytic` | Covarianza VarPro implementada en producción: SE² = s²·[(J Jᵀ)⁻¹] con J proyectado Golub–Pereyra y dof = n_obs − rank·n_λ − p | linealización del perfilado, proyección **no restringida**, ruido i.i.d. |
| `hc1` | `analytic` × factor global HC1 √(n/(n−p_total)) | igual, corrección global de grados de libertad de los lineales |
| `lin_wild_el` | Bootstrap linealizado salvaje, multiplicadores de Rademacher por elemento del residuo, con corrección HC1 | linealización; independencia elemental |
| `lin_wild_spec` | igual, un multiplicador por espectro (bloque = punto de titulación) | tolera correlación intra-espectro |
| `onestep_spec` / `onestep_wl` | bootstrap one-step (LM amortiguado): se perturban los datos observados con multiplicadores salvajes por espectro o por longitud de onda y se recalculan J y r en el ajuste original (un paso LM) | como los linealizados, con J evaluado en datos perturbados |
| `refit_wild_el` / `refit_wild_spec` | **full-refit**: datos perturbados con signos salvajes elementales o por espectro y re-ajuste completo del log K con el **mismo proyector** del estimador | sin linealización; captura el sesgo del estimador |
| `refit_resample` | full-refit con remuestreo i.i.d. de residuos por elemento | remuestreo no paramétrico i.i.d. |

Intervalos: percentil 2.5/97.5 (`ci_low/high`) y **basic** (percentil invertido,
`ci_basic_*` = 2θ̂ − percentil). Para estimadores sesgados (el restringido lo
está, véase `docs/UVVIS_BIAS_INVESTIGATION.md`) el intervalo percentil hereda
el sesgo del bootstrap —que a su vez lo estima: `boot_bias`— mientras el basic
lo compensa en primera orden. Reportamos ambos explícitamente.

**Ampliación v3 del zoo (precriterios idénticos, campañas dedicadas).**
`hc2`/`hc3`: sandwich de residuos Ω=diag(r²/(1−h)ᵖ) con h la palanca de la
*proyección lineal* (fila i: ‖C_i‖²/Σ‖C‖², aquí h≈2·10⁻⁶ ⇒ HC2≡HC3) y factor
finito MacKinlay–White n/dof sobre Ω, con el Jacobiano proyectado; es la
definición derivable en VarPro, no un trasplante de la palanca de regresión
completa. `profile`: intervalo ΔRSS sobre la MISMA objetivo perfilada
(restaurada), umbral s²·t² con los dof de la ruta analítica. Intervalos
`bca` sobre las muestras de full-refit (z₀ percentíl + aceleración jackknife
por punto). Hallazgos numéricos en `docs/MONTE_CARLO_RESULTS.md` §10.3–10.5.
La advertencia original se mantiene en espíritu: bajo i.i.d. homoscedástico
HC0/HC1/HC2/HC3 son asintóticamente equivalentes y aquí la familia HC2/HC3 no
aporta nada medible; la distinción relevante global-vs-por-observación queda
confirmada empíricamente.

## 2. Diseños congelados

Precriterios de aceptación publicados antes de correr: cobertura 95 %
esperada en [0.90, 0.97] (error MC ±1/√n a n≥60), cobertura 68 % en
[0.62, 0.74]; razón SE/SD MC reportada con su IC; tasa de fallo de refits
< 2 %; ningún método declarado "calibrado" con n < 40.

### 2.1 Intermedia i.i.d. (ruido normal homoscedástico)

- `validation/configs/unc/unc_pen_int60.json`: escenario piloto (8 puntos,
  σ=2·10⁻⁴, log K=3, G 0–2 mM, H=1 mM), estimador penalización suave μ=10⁻²,
  60 réplicas, B=1000 (wild/one-step), B=100 (full-refit), ventana de refit
  ±0.3, base_seed 20261104 (semillas nuevas, distintas de la fase exploratoria).
- `validation/configs/unc/unc_ols_int60.json`: idéntico, estimador OLS sin
  restricción (mismas semillas).

**Enmienda de diseño v2 (documentada antes de cualquier veredicto de
cobertura).** El diseño v1 usaba ventana de re-ajuste ±0.05 en torno a θ̂. Los
pilotos diagnósticos mostraron que la distribución de remuestreo del estimador
OLS tiene colas que alcanzan ±0.2 en este régimen: la ventana recortaba la
cola del bootstrap y subestimaba su dispersión (SE de refit ≈ 0.62 × SD MC
para OLS, con cobertura 0.6–0.7). Los resultados v1 se archivaron fuera del
repositorio y no se mezclan; el diseño v2 amplía la ventana a ±0.3 y sube
B_refit de 50 a 100 para estabilizar los percentiles. Los datos de los tres
métodos de linealización y analíticos no dependen de la ventana y son
idénticos entre v1 y v2.

### 2.2 Modelos de ruido no i.i.d.

- AR(1) intra-espectro (ρ=0.6): `unc_pen_ar1.json`, `unc_ols_ar1.json`
  (40 réplicas, base_seed 20261105).
- Heterocedasticidad por punto (σ escala 1→3 a lo largo de la titulación):
  `unc_pen_hetero.json`, `unc_ols_hetero.json` (40 réplicas, base_seed
  20261106).

Para ruido correlatado intra-espectro se espera que los métodos elementales
(sub)estimen la varianza y que los de bloque por espectro la capturen; para
heterocedasticidad se espera lo mismo de los métodos que asumen residuo
exchangeable (`refit_resample`) frente a wild (que conserva la estructura de
varianza por observación).

### 2.3 Confirmatoria v1 (congelada 2026-11-03 antes de analizar)

- i.i.d.: `conf_pen_iid.json`, `conf_ols_iid.json` — 500 réplicas,
  B_wild=B_one-step=1000, B_refit=0 (los de refit van aparte), base_seed
  20261110 (ambos, pareados), mismos escenarios y semillas congeladas.
- AR(1) ρ=0.6: `conf_pen_ar1.json`, `conf_ols_ar1.json` — 150 réplicas,
  B_wild=500, base_seed 20261220/20261221.
- Full-refit: `conf_pen_refit.json`, `conf_ols_refit.json` — 150 réplicas,
  B_refit=250, B_wild=1000, base_seed 20261111 (en ejecución al redactar).
- HC2/HC3: `hc_ols_{iid,ar1,hetero}.json` — 60 réplicas OLS, B_refit=0,
  B_wild=10, base_seed 20261125–27. Perfilado: `prof_pen_{iid,ar1}.json` —
  120/60 réplicas, base_seed 20261128–29. BCa: `bca_*` — 40–60 réplicas,
  B_refit=250, B_wild=50, base_seed 20261114–17.

## 3. Resultados

Campaña intermedia v2 **completada** (60+60+40×4 = 280 réplicas, ~108 000
re-ajustes bootstrap, tasa de fallo 0 %). Tablas completas, figuras y
lectura detallada en `docs/MONTE_CARLO_RESULTS.md`; datos crudos en
`validation/results/unc/*/`. Síntesis contra los precriterios:

- iid/OLS: `analytic`, `hc1` y `lin_wild_el` cumplen cobertura 95 % y 68 %.
- iid/pen: sesgo +0.0075 ± 0.0011 (significativo); ningún método cumple
  cobertura 68 %; full-refit acierta la escala (SE/SD 0.82–0.90) y el
  sesgo bootstrap (+0.0053), pero su percentil cubre 0.08 y el "basic"
  0.75–0.77. Ningún método calibrado para el estimador restringido.
- AR(1)/OLS: ningún método alcanza cov95 ≥ 0.90 (mejor hc1 0.90; los
  elementales sub-cubren hasta 0.575). Los wild por espectro mejoran pero
  con m=8 no calibran.
- hetero/OLS: `analytic`, `hc1`, `lin_wild_el` cumplen; los métodos
  agrupados por espectro fallen (0.725).
- pen con ruido correlatado/hetero: sesgo crece a ≈+0.012; los métodos
  anchos sobre-cubren (cov95 = 1.0) y el full-refit percentil colapsa.
- Colas: la SD MC de OLS en m=8 está dominada por colas (retirar la
  peor réplica la reduce un 50 %); SE/SD debe leerse junto a la
  distribución de errores.

Confirmatoria v1: i.i.d. (n=500, B=1000) y AR(1) (n=150, B=500) **completadas**;
sus tablas están en `docs/MONTE_CARLO_RESULTS.md` §10 y §10.6-bis y
constituyen ya evidencia de calibración para OLS/i.i.d. con analítico
(0.950 ± 0.010) y de colapso elemental bajo AR(1). Las confirmatorias
full-refit (n=150, B=250; base_seed 20261111) siguen en ejecución con
supervisor reanudable; ninguna afirmación sobre intervalos de full-refit
confirmatorios es definitiva hasta completarlas.

## 4. Límites del estudio

- Bootstrap de un solo paso y linealizado comparten la linealización del
  perfilado: capturan la varianza pero no del todo el comportamiento no lineal de
  colas; el full-refit sí, a coste de B pequeño.
- Los full-refit usan el argmin de la objetivo perfilada (grid+golden, tol
  10⁻⁸), no Powell multiarranque; la equivalencia con producción está fijada
  por pruebas de regresión (≤10⁻⁷) para este escenario.
- La cobertura reportada es condicional al éxito del ajuste; la tasa de fallo
  de refit se reporta aparte y en estas campañas fue baja.
- No se incluyó bootstrap de residuos por diseño completo (resampling de
  puntos de titulación), que requeriría redesigñar la matriz de concentraciones.
