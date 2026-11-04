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

HC2/HC3 por observación ( correcciones por palanca individual) **no** se
aplican: en VarPro restringido no existe una definición canónica de la matriz
sombrero del estimador proyectado, y trasplantar fórmulas de regresión lineal
sin derivación es exactamente lo que la misión prohíbe. La corrección global
HC1 sí está justificada (factor exacto de grados de libertad del término
lineal). Bajo ruido i.i.d. homoscedástico, HC0/HC1/HC2/HC3 son asintóticamente
equivalentes; la distinción relevante aquí es global-vs-por-observación y queda
documentada como límite del estudio.

## 2. Diseños congelados

Precriterios de aceptación publicados antes de correr: cobertura 95 %
esperada en [0.90, 0.97] (error MC ±1/√n a n≥60), cobertura 68 % en
[0.62, 0.74]; razón SE/SD MC reportada con su IC; tasa de fallo de refits
< 2 %; ningún método declarado "calibrado" con n < 40.

### 2.1 Intermedia i.i.d. (ruido normal homoscedástico)

- `validation/configs/unc/unc_pen_int60.json`: escenario piloto (8 puntos,
  σ=2·10⁻⁴, log K=3, G 0–2 mM, H=1 mM), estimador penalización suave μ=10⁻²,
  60 réplicas, B=1000 (wild/one-step), B=50 (full-refit), base_seed 20261104
  (semillas nuevas, distintas de la fase exploratoria).
- `validation/configs/unc/unc_ols_int60.json`: idéntico, estimador OLS sin
  restricción (mismas semillas).

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

## 3. Resultados

(Se rellenan al completarse cada campaña; véase `docs/MONTE_CARLO_RESULTS.md`
para las tablas completas y `validation/results/unc/*/summary.json` para los
datos crudos curados.)

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
