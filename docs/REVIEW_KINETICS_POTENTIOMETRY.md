# HM fit — Revisión independiente: cinética, potenciometría y cálculo de errores

*Noviembre de 2026 · rama `review/kinetics-potentiometry-reliability`, desde
`test/literature-benchmark` (`277b255`) · sin push ni fusión.*

Etiquetas, como en las fases anteriores: **[Defecto comprobado]** (reproducido con un caso
concreto antes de corregir), **[Limitación metodológica]**, **[Sospecha]** (no verificado).
Entorno: Python 3.13, NumPy 2.5.3, SciPy 1.18.1, PySide6 6.10.1 (offscreen). No es el entorno de
`constraints.txt`; el CI debe repetirlo.

## 1. Lectura general

Las Fases 1–2 y el banco de literatura dejaron bien cubiertas Espectroscopía y RMN. Los dos
módulos nuevos no habían recibido esa atención: la campaña de cobertura nunca llegó a sus
celdas (tier 3) y sus pruebas partían siempre de valores iniciales cercanos al verdadero y de
mecanismos de primer orden. Al usarlos como lo haría un estudiante (valores por defecto,
mecanismo bimolecular, varios conjuntos de datos) aparecieron defectos que cambian números en
silencio. El más grave es numérico y estaba en el núcleo, no en la estadística.

## 2. Defectos comprobados y corregidos

| # | Módulo | Defecto | Evidencia (antes → después) |
|---|---|---|---|
| K1 | Cinética, núcleo | Con leyes de conservación (más especies dinámicas que reacciones independientes + 1) la matriz C es de rango deficiente; el valor singular de redondeo (~1e-13) quedaba dentro o fuera del corte de `lstsq` según los parámetros. El objetivo era ruidoso, el Jacobiano por diferencias finitas era ruido (error relativo ~2000) y el optimizador se detenía en el valor inicial con "Fit complete" | `A + B -> C -> D`, k = (30, 0.01), inicio (20, 0.02): k1 = 20.000, SSQ 22× el ruido → k1 = 29.90, SSQ 0.92× el ruido. Error del Jacobiano 2000 → 5·10⁻⁶ |
| K2 | Cinética, núcleo | `gtol` de SciPy es absoluto sobre JᵀR: el mismo experimento daba resultados distintos según las unidades de la señal | Señal ~1e-3, inicio (0.2, 0.02): k1 = 0.050571; la misma señal ×100: 0.049864. Con el inicio por defecto (1, 1): se quedaba en (1, 1), SSQ 2270× el ruido, "Fit complete" |
| K3 | Cinética | Ajuste no convergido o que no se movió del inicio se presentaba como éxito | Ahora ">>> FIT NOT RELIABLE" con la causa |
| K4 | Cinética, GUI | `_sync_selected_metadata_from_tables` escribía las concentraciones de la tabla en **todos** los conjuntos al pulsar Process | Ajuste global de `A + B -> P` con [B]₀ = 1, 2 y 4 mM: usaba 2 mM en los tres, k = 41.9 (verdadero 50) → 49.95 |
| K5 | Cinética, GUI | Concentraciones en `QDoubleSpinBox` de 6 decimales | 2.5·10⁻⁷ M → 0; 1.5·10⁻⁶ → 2·10⁻⁶. Ahora sin redondeo y con notación científica (tabla y asistente de importación) |
| K6 | Cinética | Multi-start con los límites por defecto (mín 0, sin máx) abortaba: los arranques eran x₀ + N(0, 0.2) absoluto, negativos. Con límites finitos, muestreo uniforme lineal (en [1e-5, 1e3] nunca probaba constantes pequeñas) | `ValueError: Initial value outside its bounds` → 5 arranques válidos |
| K7 | Cinética | Ambigüedad lento-rápido de pasos consecutivos: (k1, k2) y (k2, k1) ajustan igual y se devolvía una u otra según el inicio y las unidades, sin aviso | Aviso con la razón de SSQ (una evaluación por par; no cambia números). Con multi-start, los mínimos equivalentes ya no alternan |
| A1 | Ácido-base, GUI | El selector "Data type" se creaba y se ocultaba (`77aaf78`, mayo de 2026): espectroscopía y RMN frente a pH no eran accesibles desde la interfaz | Visible y deducido de las columnas. La prueba `test_gui_equation_editor_generates_canonical_model` deja de ser xfail |
| A2 | Ácido-base y cinética | Ningún diagnóstico de estructura en los residuos | Ver §3 |

Además, en la interfaz: panel izquierdo con el ancho de su contenido (antes 540 px para un
contenido de ~700: botones y columnas cortados) en Cinética y Ácido-base; tabla de resumen con
filas alternas legibles; ejemplos de mecanismo y sintaxis en el editor; concentraciones usadas
por conjunto en el informe; botones "Export PNG/CSV" sin implementar ocultos.

**Cambios esperados en resultados.** K1: cambian constantes y SE de mecanismos con leyes de
conservación (antes no eran un mínimo); los g.l. cuentan rango(C) real. K2: cambian ajustes con
señales pequeñas que terminaban antes de tiempo. K6: las mismas semillas generan otros
arranques. El resto no cambia números. Mecanismos de primer orden bien condicionados: sin
cambios (las 500 pruebas previas pasan sin tocar tolerancias).

## 3. Cálculo de errores: qué dice la evidencia nueva

Monte Carlo pequeño con el código del producto (`A -> B -> C`, 300 réplicas, B = 500; potenciometría
diprótica, 200 réplicas, B = 400). SE/SD ideal 1; cobertura ideal 95 %.

| Caso | Método | SE/SD (k1, k2) | Cobertura IC95 |
|---|---|---|---|
| Cinética, 30 tiempos × 1 canal, ruido i.i.d. | Analítico | 0.97, 0.95 | 95.3 %, 93.3 % |
| | Bootstrap linealizado / one-step | 0.75, 0.96 | **80 %**, 89 % |
| Cinética, 30 × 8, i.i.d. | Analítico | 1.03, 1.00 | 95.7 %, 96.0 % |
| | Bootstrap por tiempo | 0.72, 0.95 | **80 %**, 92 % |
| | Bootstrap por elemento | 0.92, 0.98 | 92 %, 93 % |
| Cinética, 12 × 1, i.i.d. | Analítico | 0.87, 0.94 | 92 %, 93 % |
| | Bootstrap | 0.65, 0.92 | **77 %**, 86 % |
| Cinética, 30 × 1, ruido ∝ señal | Analítico | 1.13, 0.96 | 97 %, 95 % |
| | Bootstrap | 0.83, 0.97 | 85 %, 90 % |
| Cinética, 30 × 1, ruido AR(1) ρ = 0.8 | Analítico | 0.58, 0.37 | **75 %, 55 %** |
| | Bootstrap | 0.43, 0.37 | **56 %, 51 %** |
| Potenciometría, 40 puntos, σ(pH) = 0.005 | Analítico | 1.02, 0.98 | 94.5 %, 94.5 % |
| | Bootstrap linealizado | 1.01, 0.97 | 91.5 %, 92 % |
| Ídem + σ(V) = 0.005 mL | Analítico | 1.00, 1.01 | 94.5 %, 94.5 % |

Lectura:

1. **[Limitación metodológica] En cinética el bootstrap *wild* cubre menos que el analítico**
   para la constante del paso rápido, que depende de pocos puntos iniciales de alto *leverage*
   (el *wild* linealizado equivale al estimador sándwich HC1). Un prototipo HC2 mejora poco
   (88 %) y HC3 es errático. El método por defecto de la pestaña Errors en cinética ya es el
   analítico; la evidencia apoya mantenerlo (decisión 2.4 para cinética). El bootstrap
   *full-refit* no se pudo medir aquí por coste.
2. **[Limitación metodológica] Con residuos correlacionados en serie fallan todos los métodos**
   (cobertura 50–75 %), porque todos suponen errores independientes. Es la situación real más
   frecuente en una traza cinética o una curva de titulación: deriva, mecanismo incompleto,
   error de concentración. Un prototipo HAC (Newey–West) no lo rescata con 30 puntos. Lo que sí
   es fiable es **detectarlo**: se añadió la autocorrelación de orden 1 de los residuos a lo
   largo del tiempo / volumen / pH (`hmfit_core/utils/residual_diagnostics.py`), con aviso en
   el informe, en las estadísticas y en el resumen de Errors. Calibración con ajustes del
   producto: ruido i.i.d. → 0 falsos avisos en 400 ajustes; AR(1) ρ = 0.8, 30 puntos → 64 %
   detectado.
3. **Potenciometría: los errores aleatorios están bien calibrados; el riesgo es el error
   sistemático.** Concentración de titulante introducida con un 2 % de error: pKa1 = 9.410 ±
   0.017 (verdadero 9.5; 5 SE), IC95 [9.375, 9.444]. Antes no había ninguna señal; ahora
   r1 = 0.72 y aviso.

## 4. Abiertos (no corregidos; decisión del autor)

| # | Etiqueta | Asunto |
|---|---|---|
| 1 | Defecto comprobado por inspección | Backend JAX de cinética: RK4 de **un paso por intervalo de muestreo**, sin control de error; la exactitud depende de la rejilla de tiempos (inestable si k·Δt ≳ 2.8). Además usa `pinv` con el corte por defecto (mismo problema que K1). No se pudo ejecutar aquí (JAX no instalado). Recomendación: subpasos o marcarlo "experimental"/ocultarlo en la versión estable |
| 2 | Limitación | Mínimos locales en cinética desde inicios lejanos (p. ej. k2 → ∞ con SSQ 200× el ruido, estado "convergido"). El aviso de residuos lo señala, pero conviene valores iniciales automáticos (~1/t_max) y Log10 por defecto |
| 3 | Limitación | El bootstrap de parámetros ajustados en log10 se muestrea en escala lineal (puede dar percentiles negativos); el analítico usa IC multiplicativos |
| 4 | Limitación | Los espectros (A) se estiman por conjunto de datos; compartirlos entre conjuntos daría más poder al ajuste global |
| 5 | Limitación | t = 0 es el primer tiempo medido: no hay tiempo muerto/t₀ ajustable |
| 6 | Presentación | Potenciometría: "chi_square" con σ = 1 es la SSE en pH²; "reduced_chi_square" ~1e-5 confunde. Mostrar RMS (pH) y pKa ± SE en el informe |
| 7 | Presentación | "Warning: Initial volume is zero." se refiere al primer volumen de titulante (0 mL es lo normal) |
| 8 | Limitación | Potenciometría: residuo solo en pH; un peso tipo Hyperquad, 1/(σ_pH² + (dpH/dV)²σ_V²), haría robusto el ajuste cerca de los puntos de equivalencia. Bootstrap full-refit: 0.24 s por réplica (brentq punto a punto); vectorizarlo daría ~×50 |
| 9 | Limitación | `fit_global_acid_base` (multimodal, no usado por la GUI): bloques en unidades distintas sin pesos y g.l. sin parámetros lineales |
| 10 | UX | Idiomas mezclados (menú y parte del registro en español, resto en inglés); el mismo ajuste de ancho del panel aplicable a Espectroscopía y RMN (`fit_left_panel`) |

## 5. Pruebas

| | Aprobadas | xfail | Omitidas |
|---|---|---|---|
| Línea base (`277b255`) | 500 | 18 | 1 |
| Esta rama | 520 | 17 | 1 |

20 pruebas nuevas (`tests/test_kinetics_reliability.py`, `tests/test_residual_diagnostics.py`,
`tests/test_kinetics_gui_reliability.py`); cada una falla en `277b255` por el motivo que indica.
Ninguna prueba existente se modificó, salvo retirar el `xfail` de la que ahora pasa (A1).
