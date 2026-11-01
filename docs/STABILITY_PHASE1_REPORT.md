# HM fit — Fase 1: Integridad matemática y corrección de defectos críticos

*Rama:* `fix/stable-phase1-mathematical-integrity` (desde `origin/fix/uncertainty-reliability`, `7124267`)
*Fecha:* noviembre 2026 · *Estado:* lista para revisión; sin push ni fusión.

Convenciones: **[Defecto comprobado]** = reproducido con un caso concreto antes de corregir;
**[Limitación metodológica]** = el método no está definido o validado para ese caso;
**[Sospecha]** = indicio no demostrado.

---

## 1. Resumen de los problemas encontrados

| # | Tipo | Módulo | Problema | Estado |
|---|---|---|---|---|
| 1 | Defecto comprobado | Cinética | Límites invertidos (`lower > upper`) se convertían en parámetro fijo sin aviso | Corregido |
| 2 | Defecto comprobado | Cinética | Un parámetro fijado **por máscara** con límites finitos se movía a su **límite inferior** (k2 = 0.4 → 0.01) | Corregido |
| 3 | Defecto comprobado | Cinética | El multistart aleatorizaba también los parámetros fijos | Corregido |
| 4 | Defecto comprobado | Cinética, Espectroscopía | Una columna nula del Jacobiano (parámetro **no identificable**) se interpretaba como "fijo" → SE = 0 | Corregido (NaN + nota) |
| 5 | Defecto comprobado | Todos | Bootstrap linealizado/one-step: parámetros colineales o sin sensibilidad salían con SD finita engañosa o 0 | Corregido (NaN) |
| 6 | Defecto comprobado | Espectroscopía | El error analítico y el bootstrap one-step **no aceptaban `fixed_mask`**: un log β fijado recibía SE ≠ 0 e inflaba los SE libres; el procesador no trasladaba los fijos de la GUI (min = max) | Corregido |
| 7 | Defecto comprobado | Ácido-base (RMN y espectroscopía) | `pka_fit_mask` y `pka_bounds` **se ignoraban**: un pKa marcado como fijo se reajustaba (11 → 6.68) | Corregido |
| 8 | Defecto comprobado | Ácido-base | `x or []` con arreglos NumPy → `ValueError: truth value…`; además `_linear_structure` tragaba la excepción y contaba 0 parámetros lineales en silencio | Corregido |
| 9 | Defecto comprobado | RMN | Si el análisis de errores fallaba, el SE quedaba en **0** y se reportaba "± 0.00e+00" | Corregido (NaN) |
| 10 | Defecto comprobado | Espectroscopía, RMN, Cinética | g.l. ≤ 0 se truncaban a 1 sin aviso; en ácido-base `scale = 1` | Corregido (NaN + nota) |
| 11 | Defecto comprobado | Espectroscopía, RMN | Observaciones con peso 0 contaban como datos | Corregido |
| 12 | Defecto comprobado | Ácido-base | Coeficientes lineales contados como "entradas finitas" en lugar de rango efectivo | Corregido |
| 13 | Defecto comprobado | Pruebas | `test_nmr_regression` pasaba sin ejecutar nada (archivo de datos ausente) | Ahora `skip` explícito |
| 14 | Terminología | Bootstrap | "Corrección de leverage" era un factor global tipo HC1 | Renombrado y documentado |
| 15 | Limitación metodológica | Espectroscopía | Con ε ≥ 0, el error analítico/linealizado/one-step no está derivado para el estimador restringido; 20–26 % de coeficientes en la restricción en diseños típicos | Diagnóstico + advertencia; **Fase 2** |
| 16 | Limitación metodológica | Espectroscopía | Sesgo residual del estimador restringido cuando ε verdadera = 0 en muchas λ: +0.012 en log K (diseño 1:1 de prueba) | Documentado; **Fase 2** |
| 17 | Limitación metodológica | Bootstrap por bloques | Con < 20 puntos de titulación el bootstrap por bloques es impreciso y puede subestimar | Advertencia; decisión de valor por defecto en **Fase 2** |

## 2. Explicación matemática de cada corrección

### 2.1 Límites y parámetros fijos (pasos 2–3)
Reglas implementadas en `resolve_fixed_parameters` (cinética) y equivalentes en ácido-base
(`_least_squares_with_fixed`, `_normalize_bounds`) y espectroscopía:

* `l < u` → libre; `l = u` finitos → fijo en ese valor; `l > u` o NaN → `ValueError`.
* Fijo por máscara → conserva **exactamente** su valor definido; si además `l = u`, ambos deben coincidir.
* Máscara ausente o vacía = nada fijo; longitud incorrecta → `ValueError`.
* El vector de optimización contiene solo los libres; el Jacobiano devuelto tiene columnas nulas
  para los fijos, y la covarianza se calcula **solo sobre las columnas libres**:
  \(\mathrm{Cov}_{FF} = s^2 (J_F^\top J_F)^{-1}\), \(\mathrm{Cov}_{\text{fijo},\cdot} = 0\) por definición.

### 2.2 Covarianza linealizada única (`hmfit_core.utils.errors.linearized_covariance`)
\[
\nu = n_{\text{obs}} - p_{\text{libres}} - p_{\text{lineales}},\qquad s^2=\frac{\sum_i w_i r_i^2}{\nu},\qquad
\mathrm{Cov}=s^2\,(J^\top J)^{+}.
\]
* \(n_{\text{obs}}\): observaciones finitas, no enmascaradas y con peso > 0.
* \(p_{\text{lineales}}\): rango efectivo de la matriz de diseño de proyección variable por canal
  (espectroscopía: \(\mathrm{rank}(C_{\text{eff}})\times\) nº de λ con peso; RMN: rango de cada
  bloque por señal; ácido-base: rango de las fracciones observadas por canal; cinética:
  \(\mathrm{rank}(C)\times\) nº de canales por conjunto de datos).
* \(\nu \le 0\) → \(s^2\) y SE = NaN con nota "grados de libertad insuficientes" (antes se usaba \(\nu = 1\)).
* Identificabilidad: SVD de \(J\) con columnas normalizadas (invariante a unidades). Un parámetro
  con componente > 1e-6 en el espacio nulo numérico (tolerancia \(\max(n,p)\,\varepsilon\,\sigma_1\)) es
  **no identificable** → fila/columna NaN. **Un SE = 0 significa únicamente "fijado".**

### 2.3 Bootstrap
* Linealizado y one-step: los parámetros fijos no se mueven (solo se actualiza el subvector libre);
  los no identificables → muestras NaN.
* Factor global **tipo HC1** \(\sqrt{n/(n-p_{\text{total}})}\) (`hc1_rescale_factor`; alias
  `leverage_rescale_factor` por compatibilidad). No es HC2/HC3 (\(r_i/\sqrt{1-h_{ii}}\), \(r_i/(1-h_{ii})\)).
* Avisos por número de bloques independientes (`resampling_notes`): < 20 bloques → impreciso;
  remuestreo por elemento → supone errores independientes entre canales.
* Correlaciones bootstrap: columnas sin dispersión (fijas) o no finitas → NaN (no 0).
* Semillas: siempre registradas; reproducibilidad probada (mismo `seed` → mismas muestras).

### 2.4 Restricciones de absortividad (paso 6) — auditoría
| Pregunta | Respuesta |
|---|---|
| Estimador del ajuste principal | `_solve_absorptivities`: `soft_penalty` (exacto, Newton semiliso), `nnls_hard` (NNLS exacto), `soft_bound`/`delta_mode` (gradiente proyectado con cota inferior), opcional penalización de suavidad |
| Estimador en el cálculo de errores | Proyección **sin restricción** \(A = C^{+}Y\) (analítico, linealizado, one-step). El full-refit usa el estimador restringido |
| Cuándo se activa | Cuando \(C^{+}Y\) tiene entradas negativas (o bajo la cota inferior) |
| Coeficientes libres | En `nnls_hard` las entradas activas valen 0 (fijas); en `soft_penalty` están penalizadas, no fijas |
| Efecto en Jacobiano y g.l. | El Jacobiano de proyección variable debería usar, por λ, solo las columnas libres de \(C\); los g.l. cambian en consecuencia |
| ¿Covarianza válida? | No está derivada para el estimador restringido. Cerca de \(\varepsilon = 0\) verdadero la distribución no es gaussiana (problema de frontera), por lo que una covarianza condicional al conjunto activo tampoco sería rigurosa sin validación |

Decisión: **no** se implementó una corrección improvisada. Se añadió `absorptivity_constraint_report`,
que cuenta los coeficientes en la restricción y emite una advertencia en el texto de resultados,
en la exportación (`eps_constraint_active/total`, `uncertainty_notes`) y en la pestaña Errors,
recomendando el bootstrap full-refit. No se afirma que el error analítico sea conservador.

## 3. Archivos y funciones modificados

| Archivo | Funciones / cambios |
|---|---|
| `hmfit/kinetics/fit/optimizer.py` | `resolve_fixed_parameters` (nueva), `fit_global` (validación, fijos exactos, x0 fuera de límites, covarianza vía helper, `fixed_mask`/`nonidentifiable`/`notes` en `GlobalFitResult`) |
| `hmfit_core/utils/errors.py` | `linearized_covariance`, `nonidentifiable_parameters`, `sample_correlation`, `hc1_rescale_factor`, `resampling_notes` (nuevas); `compute_errors_spectro_varpro` (`fixed_mask`, n_obs con peso, sin truncar g.l.); `compute_errors_nmr_varpro` (helper, NaN, notas); `bootstrap_linearized_wild`, `bootstrap_one_step_spectro` (`fixed_mask`), `bootstrap_one_step_nmr`, `bootstrap_full_refit` |
| `hmfit_core/acid_base.py` | `_least_squares_with_fixed` (nueva), `_covariance_and_correlation`, `_build_fit_result`, `_count_linear_coefficients` (rango), `fit_spectroscopy_acid_base`, `fit_nmr_acid_base` |
| `hmfit_core/acid_base_errors.py` | `_array_or_empty` (nueva), `_normalize_fixed_mask`, `_normalize_bounds`, `compute_errors_acid_base` (helper), `_linear_structure` (sin excepciones silenciosas), `_build_acid_base_output` (notas de remuestreo), contextos con arreglos |
| `hmfit_core/run_acid_base.py` | `_pka_fit_controls` (nueva); `_run_spectroscopy`, `_run_nmr` pasan máscara y límites; patrones `or` |
| `hmfit_core/processors/spectroscopy_processor.py` | `absorptivity_constraint_report` (nueva); límites invertidos → error; fijos desde min = max hacia errores y tabla; avisos |
| `hmfit_core/processors/nmr_processor.py` | SE inicial NaN en lugar de 0 |
| `hmfit_gui_qt/tabs/kinetics_tab.py` | Ruta JAX con `resolve_fixed_parameters`; `_compute_covariance`; `_randomize_params` respeta fijos; estado por parámetro (fijo / estimado / no identificable) |
| `hmfit_gui_qt/widgets/model_opt_plots.py` | `fixed_mask` en analítico y one-step espectro; notas de remuestreo y restricciones |
| `hmfit_gui_qt/widgets/kinetics_model_opt_plots.py` | Máscara explícita (no se infiere de columnas nulas); NaN para no identificables; notas HC1/bloques |
| `docs/RELIABILITY_REVIEW.md` | Terminología HC1 y afirmaciones sobre el error analítico corregidas |

## 4. Pruebas añadidas

* `tests/test_phase1_invariants.py` (21 pruebas): límites libres/fijos/invertidos/NaN; máscaras
  vacías y de tamaño incorrecto; todos fijos / ninguno / mixtos; valor fijo exacto; parámetro no
  identificable → NaN; invariantes de fijos en analítico y one-step espectroscópico; bootstrap con
  parámetros colineales; contextos ácido-base con arreglos de 1, 2 y 5 parámetros; vector vacío;
  covarianza de referencia \(s^2(J^\top J)^{-1}\) con \(\nu = n-p-p_{\text{lin}}\); \(\nu \le 0\);
  invariancia de escala del rango; pKa fijo respetado en RMN y espectroscopía con los tres métodos;
  diagnóstico de restricciones; HC1 y avisos de bloques; reproducibilidad con semilla; conteo de
  réplicas fallidas.
* `tests/test_phase1_validation.py` (21 pruebas) + `tests/phase1_cases.py` (casos sintéticos
  reproducibles): recuperación de parámetros consistente con el SE reportado; fijos exactos;
  datos sin ruido; covarianza ácido-base frente a diferencias finitas independientes; escala de
  segundo orden; Monte Carlo pequeño (RMN ácido-base, sesgo espectroscópico).
* `tests/test_nmr_regression.py`: `skip` explícito si falta el archivo de referencia.

Ninguna prueba existente fue eliminada ni desactivada. Las modificaciones de pruebas previas de la
rama base (`test_nmr_errors_repro`, `test_acid_base`, `test_runtime_optimizations`) son de esa rama.

## 5. Resultados de las pruebas antes y después

PySide6 6.10.1 con `QT_QPA_PLATFORM=offscreen`, Python 3.13, NumPy 2.5, SciPy 1.18.

| | Recolectadas | Pasan | Fallan | Omitidas | Vacías (pasan sin comprobar) |
|---|---|---|---|---|---|
| Línea base (`7124267`) | 162 | 158 | 4 | 0 | 1 (`test_nmr_regression`) |
| Fase 1 (`f36d6c8`) | 204 | 199 | 4 | 1 | 0 |

Las 4 que fallan son **las mismas** que en la línea base (preexistentes, ninguna nueva):
`test_custom_model_dimensions_start_without_forced_roles`, `test_gui_equation_editor_generates_canonical_model`,
`test_gui_potentiometry_validation_warns_for_operational_ph` (GUI ácido-base) y
`test_absgrp_shared_groups_reduce_optical_components` (convergencia de Powell). Dos ejecuciones
completas consecutivas dieron resultados idénticos. Las pruebas de extremo a extremo de la GUI
(cinética, RMN, espectroscopía) reproducen exactamente los valores previos en los casos no afectados.

## 6. Validaciones numéricas con datos sintéticos

Casos deterministas (`python tests/phase1_cases.py`). Ruido gaussiano; semilla y σ en la última
columna. "fijo" = parámetro fijado (SE = 0 por definición).

| Caso | Técnica | Verdadero | Estimado | SE | |Δ|/SE | g.l. | Semilla / ruido |
|---|---|---|---|---|---|---|---|
| UV-Vis 1:1, log K = 3 | UV-Vis | 3 | 3.028 | 0.0133 | 2.10 | — | 300 / N(0,0.0002) abs |
| UV-Vis 1:1, no noise | UV-Vis | 3 | 3 | 2.939e-14 | 3.60 | — | 300 / N(0,0) abs |
| UV-Vis 1:2, log b2 fixed | UV-Vis | 4, 7 | 4, 7 | 0.001073, 0 | 0.41, fijo | — | 300 / N(0,0.0002) abs |
| NMR 1:1, log K = 3 | 1H NMR | 3 | 2.999 | 0.009807 | 0.13 | — | 4 / N(0,0.002) ppm |
| NMR 1:2, log b2 fixed | 1H NMR | 4, 7 | 4.029, 7 | 0.02478, 0 | 1.15, fijo | — | 4 / N(0,0.002) ppm |
| Acid-base NMR, pKa 4.0/7.5 | acid-base NMR | 4, 7.5 | 4.233, 7.241 | 0.1677, 0.1684 | 1.39, 1.54 | 20 | 0 / N(0,0.003) ppm |
| Acid-base NMR, close pKa 4.0/4.4 | acid-base NMR | 4, 4.4 | 3.982, 4.353 | 0.01849, 0.0184 | 0.98, 2.54 | 20 | 0 / N(0,0.003) ppm |
| Acid-base NMR, pKa2 fixed | acid-base NMR | 4, 9 | 3.987, 9 | 0.01743, 0 | 0.74, fijo | 11 | 0 / N(0,0.003) ppm |
| Potentiometry, pKa 5.0 | potentiometry | 5 | 5 | 0.0003187 | 1.06 | 59 | 11 / N(0,0.002) pH |
| Kinetics 1st order A->B->C | kinetics | 1.2, 0.4 | 1.255, 0.4087 | 0.06258, 0.03849 | 0.87, 0.23 | 442 | 0 / N(0,0.01) |
| Kinetics 1st order, no noise | kinetics | 1.2, 0.4 | 1.2, 0.4 | 1.04e-12, 6.687e-13 | 15.72, 20.11 | 442 | 0 / N(0,0) |
| Kinetics 1st order, k2 fixed | kinetics | 1.2, 0.4 | 1.262, 0.4 | 0.05418, 0 | 1.14, fijo | 443 | 0 / N(0,0.01) |
| Kinetics 2nd order, k = 5e3 M^-1 s^-1 | kinetics | 5000 | 5002 | 7.601 | 0.22 | 695 | 0 / N(0,0.002) |

Notas: en los casos sin ruido los SE son del orden de 1e-12–1e-14 y el cociente |Δ|/SE no tiene
significado (error de redondeo); la recuperación es exacta a 1e-6. El SE de los dos pKa
"cercanos" coincide con un cálculo independiente por diferencias finitas a 1e-3 (su correlación es
0.10: con desplazamientos equiespaciados el diseño determina por separado la suma y la diferencia
de los pKa).

Monte Carlo pequeño (determinista, en las pruebas automatizadas):

| Caso | Réplicas | Media | SD empírica | SE medio | SE/SD | Cobertura IC95 |
|---|---|---|---|---|---|---|
| Ácido-base RMN, 10 puntos, pKa 5.0, σ = 5e-3 (semillas 1000–1039) | 40 | 5.0055 | 0.0169 | 0.0171 | **1.01** | **95.0 %** (t, ν = 7) |
| Cinética A→B→C, k = (1.2, 0.4), σ = 1e-2 (semillas 100–139) | 40 | 1.186 / 0.416 | 0.067 / 0.046 | 0.072 / 0.047 | **1.07 / 1.01** | **95 % / 95 %** (z) |
| UV-Vis 1:1, log K = 3, σ = 2e-4 (semillas 500–511) | 12 | 3.0123 | 0.0099 | — | — | — |

El último caso muestra el **sesgo residual** del estimador restringido (ε ≥ 0) cuando las ε
verdaderas son 0 en muchas λ: +0.012 (antes de la rama base era +0.043; el estimador sin
restricción da 3.000). Se registra como limitación para la Fase 2.

## 7. Cambios esperados en los resultados científicos

* **Valores ajustados:** sin cambios, salvo (a) ácido-base RMN/espectroscopía con pKa marcados como
  fijos o con límites, que ahora se respetan, y (b) parámetros cinéticos fijados por máscara con
  límites finitos, que ya no se mueven al límite inferior.
* **SE en espectroscopía con parámetros fijos (min = max):** el fijo pasa a SE = 0 y los libres
  dejan de estar inflados por la covarianza con el fijo.
* **Parámetros no identificables:** SE = NaN ("undefined") en lugar de valores pequeños o cero.
* **Muy pocos datos (ν ≤ 0):** SE = NaN con nota, en lugar de una cifra aparentemente precisa.
* **Configuraciones inválidas** (límites invertidos, NaN, máscaras de tamaño incorrecto, valor
  inicial fuera de límites en cinética) generan un error claro en lugar de un ajuste silencioso.
* Texto de resultados: el formato de la tabla se conserva (`const` para fijos); se añaden líneas
  `>>> WARNING` cuando hay restricciones activas o notas de identificabilidad.

## 8. Problemas que permanecen abiertos

1. **[Limitación]** Covarianza con ε ≥ 0 activa (§2.4) y sesgo residual del estimador restringido (§6).
2. **[Limitación]** Bootstrap por bloques con pocos puntos: valor por defecto (`row`) a decidir con
   estudio de cobertura (avisos ya implementados). La rama base lo cambió de `element` a `row`.
3. **[Defecto]** Powell se detiene en puntos distintos según el valor inicial
   (`test_absgrp_shared_groups…`) — Fase 3 (optimizadores).
4. **[Defectos GUI]** Tres pruebas ácido-base de GUI preexistentes (roles por defecto, detección de
   archivo `pH,signal`, avisos de pH operacional) — requieren decisión de diseño (Fase 4).
5. **[Limitación]** En espectroscopía con pesos por punto (m × nλ) el estimador lineal no se pondera
   igual que el residuo; con pesos por λ es exacto (cada columna se resuelve por separado).
6. **[Limitación]** Potenciometría: los pKa fijos no aparecen en la tabla de errores (se excluyen del
   vector); el comportamiento es correcto pero la presentación es incompleta.
7. Archivo de referencia `test_2_tit_RMN3.xlsx` ausente del repositorio.

## 9. Riesgos y posibles regresiones

* Usuarios con configuraciones antiguas que contenían límites invertidos o máscaras de longitud
  incorrecta verán ahora un error en lugar de un ajuste (comportamiento deseado, pero visible).
* En cinética, un valor inicial fuera de límites ya no se recorta en silencio: se rechaza.
* Correlaciones de parámetros fijos o no identificables aparecen como NaN (antes 0); la GUI las
  muestra como "nan". Revisar presentación en la Fase 4.
* `fixed_mask` de tamaño incorrecto en el contexto de errores de espectroscopía/RMN de la GUI se
  trata como "nada fijo" (el procesador valida la máscara del ajuste); se mantiene para no romper
  proyectos guardados.
* La tolerancia de rango (\(\max(n,p)\,\varepsilon\,\sigma_1\) sobre J normalizado) es estricta: modelos
  casi singulares no se marcan como no identificables, sino que muestran SE muy grandes (correcto,
  pero conviene combinarlo con el diagnóstico de condición existente).

## 10. Recomendaciones concretas para la Fase 2

1. **Campaña de cobertura** (≥ 500 conjuntos por caso, B ≥ 1000) para los cuatro métodos y las cuatro
   técnicas, con ruido i.i.d., heterocedástico y correlacionado por espectro; decidir con datos el
   remuestreo por defecto (`row` vs `element`) y un umbral de bloques.
2. **Restricciones de absortividad:** comparar (a) estimador sin restricción para K + restricción solo
   para mostrar ε, (b) Jacobiano condicional al conjunto activo, (c) full-refit; cuantificar sesgo y
   cobertura cerca de ε = 0.
3. Evaluar HC2/HC3 (con la diagonal de la matriz sombrero de proyección variable) frente al factor
   global HC1.
4. **Perfil de verosimilitud** para parámetros con IC asimétricos o mal identificados.
5. Comparación con referencias externas (BindFit, HypNMR, Hyperquad) usando `tests/phase1_cases.py`
   como base de casos.
6. Unificar en un solo motor (`hmfit_core/uncertainty.py`) las tres implementaciones paralelas de
   errores; la Fase 1 ya centraliza covarianza, identificabilidad, HC1 y avisos.

---

## Dictamen

La rama **está preparada para integrarse a una rama de desarrollo** (no a `main` todavía): corrige
13 defectos comprobados sin introducir fallos nuevos, añade 42 pruebas y no modifica los
resultados de los casos que no debían cambiar. **No** está lista para una versión estable hasta
resolver en la Fase 2 las limitaciones 1–2 de la sección 8, que afectan a las incertidumbres en
espectroscopía con restricciones activas y al bootstrap con pocos puntos.
