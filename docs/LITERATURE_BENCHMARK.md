# HM fit — Banco de validación externa con datos publicados

> **Estado (2026-11-07): etapas A–E completas (la E, opcional, tras A–D con el CI en verde).** Las
> secciones 1–4 son el pre-registro (commit `a39f0c0`) y no se han modificado después, salvo esta
> línea de estado; los resultados empiezan en §5. Etiquetas, como en las fases anteriores: **[Defecto comprobado]** (reproducido con un caso
> concreto), **[Limitación metodológica]**, **[Sospecha]** (no verificado).

Rama: `test/literature-benchmark`, desde `fix/stable-phase2-completion` (`8a45080`; el encargo
cita `d71fb95`, y `8a45080` solo añade el enlace del CI final de la Fase 2 al informe).

## 1. Fuentes

- **[R09a]** S. Roelens, A. Vacca, C. Venturi, "Binding of Ionic Species: A General Approach To
  Measuring Binding Constants and Assessing Affinities", *Chem. Eur. J.* **2009**, *15*,
  2635–2644. DOI: [10.1002/chem.200802298](https://doi.org/10.1002/chem.200802298).
- **[R09b]** S. Roelens, A. Vacca, O. Francesconi, C. Venturi, "Ion-Pair Binding: Is Binding Both
  Binding Better?", *Chem. Eur. J.* **2009**, *15*, 8296–8302. DOI:
  [10.1002/chem.200900342](https://doi.org/10.1002/chem.200900342).

Los datos pertenecen a sus autores y se usan solo para la validación interna de HM fit. Los
PDF no están en el repositorio (`tests/data/literature/.gitignore` excluye `*.pdf`); solo se
versionan los números extraídos. Detalle por caso, con página del SI: `tests/data/literature/SOURCES.md`.
Programa de referencia: HypNMR (versiones 16.11.2006 y 19.06.2007, según cada página de resultados).

## 2. Capacidades de HM fit (A.1)

Verificado con un sistema sintético de tres reactivos (R, Q, X; especies RQX, RX, QX; dos
titulaciones; un valor faltante) por `hmfit_core.api.run_nmr_fit`, el punto de entrada de la
pestaña RMN, con la configuración que construye `NMRTab._collect_config` y el mapa de
estequiometría de `NMRTab._inject_stoichiometry_map`. Sin ruido, HM fit recupera log β =
6.0000/3.0000/4.0000 con RMS ≈ 10⁻⁸ ppm, todo libre y con β(QX) fija.

| Capacidad | Estado | Cómo se expresa en HM fit |
|---|---|---|
| Tres reactivos con especies de par iónico | **Sí** | Matriz de modelo especies × reactivos; solver de balances genérico |
| Señales de reactivos distintos, promedio ponderado por estequiometría y normalizado por T del reactivo | **Sí** | "Parent" de cada señal → columna del mapa de estequiometría; `D = C·S_j`, fracciones `C·S_j/D` (`project_coeffs_block_onp_frac`) |
| Varias titulaciones con constantes compartidas | **Sí** | Filas apiladas en una sola hoja; la especiación se resuelve por fila. HM fit no tiene noción de "conjunto": los desplazamientos límite también se comparten, igual que en HypNMR |
| Valores faltantes por señal y por punto | **Sí** | Celdas vacías → máscara por señal |
| Constantes de formación fijas | **Sí** | `k_fixed` |
| Desplazamientos límite fijos | **No** | Todos los desplazamientos límite son parámetros lineales resueltos por mínimos cuadrados; no hay forma de fijar uno |
| Pesos unitarios en RMN | **No seleccionable** | El ajuste RMN aplica siempre IRLS con un peso por señal (1/σ_j², hasta 5 ciclos); ver §4.3 |
| Desplazamientos límite en la salida | Parcial | No están en el resultado del ajuste; se obtienen del cálculo analítico (`compute_errors_nmr_varpro`, `coef`) y de la hoja "Coefficients" de la exportación. Sin error estándar en ningún sitio |
| Correlaciones de log β | Sí | Pestaña Errors, método analítico (pesos unitarios) |

**Casos bloqueados:** solo `r09b-d-qx` (seis desplazamientos fijos de DQ). Bloquea 1 de 10
casos de ajuste (< 50 %), así que se continúa y el caso queda como **no soportado**.
Estimación de esfuerzo para soportarlo: restar la contribución conocida
`F_fijo·δ_fijo` de cada señal antes de la proyección, excluirla del Jacobiano de proyección
variable y del recuento `n_linear`, y exponerlo en la GUI (tabla de desplazamientos fijos).
Núcleo y pruebas: ≈ 1–2 días; GUI: Fase 4. No implementado (regla 7).

**[Defecto comprobado] Exportación de desplazamientos límite (corregido, `f30919d`).** La hoja
"Coefficients" de `write_results_xlsx` recalculaba los desplazamientos con la columna "padre"
deducida del nombre de la señal y sin el mapa de estequiometría del ajuste. En el sistema
sintético exportaba, por ejemplo, Me en QX = −0.103 ppm (verdadero 3.22) y NH en RX = 3.667
(verdadero 5.5), aunque las constantes eran correctas. Corregido para usar las mismas
fracciones que el ajuste; las especies sin el reactivo del núcleo quedan vacías. Prueba:
`tests/test_nmr_export_coefficients.py` (falla sin la corrección). Cambio esperado: solo
cambian los números de esa hoja exportada, en modelos con mapa de estequiometría; ningún
resultado del ajuste cambia.

## 3. Casos y verificación de la extracción (A.2–A.4)

| Caso | Fuente | Espectros | Valores | Núcleos | m/(m−p) | p (β + δ) | Soportado |
|---|---|---|---|---|---|---|---|
| `r09a-r1-11-me` | R09a, SI S4–S5 | 20 | 20 | 1 | 20/17 | 1 + 2 | sí |
| `r09a-r1-11-3n` | R09a, SI S4, S6 | 20 | 58 | 3 | 58/51 | 1 + 6 | sí |
| `r09a-r1-3sp` | R09a, SI S7–S8 | 40 | 116 | 3 | 116/104 | 3 + 9 | sí |
| `r09a-r1-5sp` | R09a, SI S7, S9; Tabla 1 | 40 | 116 | 3 | 116/98 | 5 + 13 | sí |
| `r09a-qx-dil` | R09a, SI S10–S11 | 25 | 25 | 1 | 25/22 | 1 + 2 | sí |
| `r09a-p2-058` | R09a, SI S12–S14; Tabla 2 | 39 | 257 | 7 | 257/212 | 5 + 40 | sí |
| `r09a-p2-106` | R09a, SI S17–S19 | 40 | 259 | 7 | 259/214 | 5 + 40 | sí |
| `r09b-r-qx` | R09b, SI S9–S10; Tabla 1 | 40 | 116 | 3 | 116/98 | 5 + 13 | sí |
| `r09b-b-qx` | R09b, SI S13–S14; Tabla 2 | 50 | 45 | 1 | 45/39 | 2 + 4 | sí |
| `r09b-d-qx` | R09b, SI S15–S17; Tabla 3 | 41 | 235 (236 en la tabla) | 6 | 235/202 | 4 + 29 | **no** (δ fijos) |
| `r09b-spec-r`, `-b`, `-d-x`, `-d-q` | R09b, SI S20–S23 | 50 filas c/u | — | — | — | — | etapa D |

Las páginas de resultados de r09a-p2-058/-106 continúan en S14 y S19 (correlaciones y el
último bloque de desplazamientos); las gráficas (S15–S16, S20–S21) no se usan.

**Extracción** (`python -m validation.literature_extract --pdf-dir DIR --write`):

- *Pasada A*: pdfplumber (pdfminer.six), palabras con coordenadas; los bloques de dos columnas
  de desplazamientos límite se asignan por la posición x.
- *Pasada B*: `pdftotext -layout` (poppler); los bloques se asignan con la regla química (un
  núcleo solo tiene desplazamiento en especies que contienen su reactivo); la columna de
  carácter solo resuelve el caso "valor sin error" de DQ.
- Las dos pasadas se comparan campo a campo (cadenas impresas): **0 diferencias** en los 10
  casos y en las 4 tablas de especiación.
- **Conteos**: espectros, valores y núcleos coinciden con los impresos, salvo `r09b-d-qx`: la
  tabla del SI tiene 236 valores y la página de resultados imprime 235, y la identidad se
  cumple con m = 235. **[Limitación metodológica]** un valor tabulado no se usó en el ajuste
  original y el SI no dice cuál. El caso no está soportado, así que no afecta a los resultados.
- **Identidad de grados de libertad** (σ / RMS)² = m/(m − p), con p = β refinadas +
  desplazamientos impresos con error: se cumple en los 10 casos con diferencia relativa
  ≤ 5·10⁻⁸ (σ y RMS se imprimen con 8 cifras). En `r09b-d-qx`, p = 33 = 4 β + 29
  desplazamientos: los de D2 se refinan aunque β(D2) está fija, y los seis de DQ, impresos sin
  error, están fijos en los valores impresos (`case.json`, estado `fixed`). Verificado.
- **Repeticiones**: `r09b-r-qx` y `r09a-r1-5sp` (otro SI) son idénticos en datos y resultados;
  la titulación de 0.15 mM es idéntica en `r09a-p2-058` y `r09a-p2-106`; la tabla de dos
  reactivos (S4) es idéntica a la primera titulación de la de tres reactivos (S7).
- **Tablas del artículo**: transcritas a mano y comprobadas automáticamente contra el texto
  extraído por las dos herramientas; cada log β y SD del artículo coincide con el valor del SI
  redondeado (mitad hacia arriba), con una excepción que se reporta sin juzgarla: la Tabla 2
  de R09a imprime SD(log β P2X) = 0.03 y el SI 0.0353, que redondea a 0.04. El banco usa el SI.
- **Constantes fijas**: se usa el log β impreso (0.1000, 4.0897, 0.1892), que reproduce el β
  impreso (10^4.0897 = 1.2294·10⁴; 10^0.1892 = 1.5460), es decir, el valor que se introdujo.

## 4. Criterios pre-registrados (etapa B)

Escritos y con commit **antes** de comparar cualquier resultado de HM fit con la referencia.
El hash de este commit se cita en las secciones de resultados.

### 4.1 Niveles y clasificación

| Nivel | Magnitud | Tolerancia propuesta |
|---|---|---|
| 0 (diagnóstico) | RMS de HM fit con **todas** las constantes fijas en los valores de referencia (solo se resuelven los desplazamientos) frente al RMS de referencia | Igual que el nivel 1 del caso |
| 1 | Ajuste: RMS de HM fit frente al "RMS weighted residual" de HypNMR | Diferencia relativa ≤ 3 % |
| 2 | log β | \|Δ\| ≤ máx(0.25 · SD de referencia, 0.005) |
| 3 | SE de log β | Diferencia relativa ≤ 15 % |
| 4 | Desplazamientos límite y correlaciones | Desplazamientos dentro del error reportado; correlaciones dentro de ±0.05 |

Clasificación: **reproduce** (niveles 1, 2 y 3), **parcial** (pasa el 1 y falla el 2 o el 3),
**no reproduce** (falla el 1 o el ajuste falla), **no soportado**. El nivel 4 y el σ se
reportan, pero no clasifican. El nivel 0 no clasifica: sirve para revisar las hipótesis (1)
transcripción y (2) especificación del modelo de la regla 5, sin depender del optimizador.

### 4.2 Magnitudes equivalentes

- **RMS frente a RMS.** El ajuste de HM fit no reporta σ con pesos unitarios: su `covfit` usa
  los pesos IRLS. El nivel 1 compara el RMS sin pesos del resultado (`stats_table` → `RMS`,
  √(Σr²/m)) con el "RMS weighted residual" de HypNMR, que con pesos unitarios (identidad de
  §3) es el mismo cociente. Como dato secundario se reporta σ = √(s²) del método analítico de
  la pestaña Errors (pesos unitarios), cuyo ν = m − p_libres − n_linear incluye los parámetros
  lineales; se comprueba que ν coincide con m − p de HypNMR.
- **SE.** HypNMR da la SD de log β como SD relativa de β / ln 10 (propagación lineal). El nivel 3
  compara con el **SE analítico** que muestra el resultado del ajuste (`constants[].SE_log10K`),
  no con intervalos. Ese SE usa los pesos IRLS del ajuste; como dato secundario se reporta el
  SE analítico de la pestaña Errors (pesos unitarios, en el mismo log β).
- **Desplazamientos límite y correlaciones**: `coef` y covarianza de `compute_errors_nmr_varpro`
  con los mismos argumentos que `ModelOptPlotsWidget._compute_errors_from_context` (método
  analítico). Las correlaciones son las de la pestaña Errors.
- **Pesos.** Se suponen pesos unitarios en HypNMR (consistente con la identidad de §3). HM fit
  no permite elegirlos (§2). **[Limitación metodológica]** en los casos de varios núcleos, el
  estimador de HM fit no es el de HypNMR: pondera cada señal por 1/σ_j² de sus residuos. Con un
  solo núcleo el peso es una constante común y ambos estimadores coinciden. Si el nivel 1 o el 2
  fallan de forma sistemática en los casos de varios núcleos y no en los de uno, se reportará
  como causa probable la diferencia de pesos (hipótesis (2) de la regla 5), sin cambiar el
  método.

### 4.3 Configuración del ajuste

Igual que la GUI por defecto: optimizador `powell`, especiación `Newton-Raphson`, modelo
`Free`, un arranque, sin límites. **Valores iniciales**: log β de referencia + 0.2 en todas las
constantes refinadas (no se arranca en la solución); las fijas, en su valor impreso. Esta
regla es única para todos los casos y no se ajusta a posteriori.

### 4.4 Análisis de sensibilidad al redondeo

`python -m validation.literature_benchmark sensitivity --n 50 --workers 30` →
`validation/results/literature/sensitivity.json` (semilla base 20261107). Para cada caso, 50
reajustes con perturbaciones uniformes dentro de media unidad del último dígito impreso:
concentraciones según sus cifras impresas (3 o 4 significativas; los ceros se mantienen), con
una sola perturbación por valor impreso de la sal compartida por catión y anión (son la misma
magnitud física); desplazamientos ±0.00005 ppm. El script **solo** reporta dispersiones; no
lee ni imprime valores de la referencia. Los niveles 2 usan la SD de referencia solo porque
la propia tolerancia se define con ella.

| Caso | CV(RMS) | máx. SD(log β) | máx. rango(log β) | máx. CV(SE) | Ampliaciones (3 × dispersión) |
|---|---|---|---|---|---|
| `r09a-r1-11-me` | 4.9 % | 0.0004 | 0.0017 | 4.9 % | L1 → 14.7 % |
| `r09a-r1-11-3n` | 0.06 % | 0.0005 | 0.0021 | 4.2 % | — |
| `r09a-r1-3sp` | 0.05 % | 0.0024 | 0.011 | 1.7 % | — |
| `r09a-r1-5sp` | 2.1 % | 0.0121 | 0.053 | 3.3 % | L2 R2X → 0.036 |
| `r09a-qx-dil` | 0.56 % | 0.0004 | 0.0019 | 0.56 % | — |
| `r09a-p2-058` | 1.2 % | 0.0153 | 0.070 | 3.5 % | L2 PQX → 0.023; P2X → 0.037; P2QX → 0.046 |
| `r09a-p2-106` | 1.5 % | 0.0100 | 0.050 | 2.1 % | L2 P2QX → 0.030 |
| `r09b-r-qx` | 1.6 % | 0.0116 | 0.050 | 3.2 % | L2 R2X → 0.035 |
| `r09b-b-qx` | 5.0 % | 0.0058 | 0.028 | 5.2 % | L1 → 14.8 %; L2 BQX → 0.017 |

Tolerancias resultantes por caso y constante: `tests/data/literature/<caso>/tolerances.json`
(generadas con `python -m validation.literature_benchmark tolerances … --write`). El nivel 3
no se amplía en ningún caso (CV(SE) ≤ 5.2 %).

Observaciones del análisis (solo dispersiones):

- **[Limitación metodológica]** La dispersión mezcla dos efectos que este diseño no puede
  separar: la propagación del redondeo y el punto en que se detiene Powell, que cambia con
  cualquier perturbación mínima de los datos. Las mayores dispersiones de log β (0.012–0.015)
  aparecen en los casos de 5 constantes con Powell, mientras que la del RMS es pequeña. Separar
  ambos efectos exige otro optimizador (Fase 3).
- En `r09a-r1-11-me` y `r09b-b-qx` el RMS cambia un 5 %: con un solo núcleo y RMS de 2–5·10⁻⁴
  ppm, la perturbación de ±5·10⁻⁵ ppm por punto no es despreciable frente a los residuos.
- `r09a-r1-5sp` y `r09b-r-qx` son los mismos datos con semillas distintas: sus dispersiones
  difieren poco y sus tolerancias quedan casi iguales; se mantienen las de cada uno.
- El análisis se ejecutó con el arnés de este commit antes de añadirle la generación de
  tolerancias (commit de datos `4550ee5` en el manifiesto). Python 3.14.9, NumPy 2.5.4,
  SciPy 1.18.1.

### 4.5 Línea base de regresión

Cada caso soportado guarda los resultados actuales de HM fit en `baseline.json` (log β, SE, RMS,
σ, desplazamientos, evaluaciones, commit y entorno). Una prueba aparte exige
|Δ log β| ≤ 10⁻⁴ y SE con diferencia relativa ≤ 10⁻³ frente a ella. Solo se actualiza con un
commit explícito "Update literature baseline" y su justificación en este informe.

### 4.6 Etapa D (especiación) — criterios

Concentraciones libres: diferencia relativa ≤ 0.5 %; fracciones: diferencia absoluta ≤ 0.0005;
en filas con fracciones cercanas a 0 o 0.98 basta la absoluta. Se prueban las constantes de la
página de resultados y las redondeadas del artículo, y se documenta cuál reproduce la tabla.
Una diferencia mayor que el redondeo es candidata a [Defecto comprobado] del resolvedor.

### 4.7 Etapa E (BC50) — criterios (añadidos antes de comparar, commit de `hmfit_core/bc50.py`)

Implementación: `hmfit_core/bc50.py` (`bc50_curve`), sin GUI; probada contra las expresiones
analíticas A23–A30 del apéndice de R09a (`tests/test_bc50.py`). Comparaciones:

1. **Tablas S20–S23 de R09b** (T_C = T_A): con la rejilla x exacta (0.00 … 0.98), BC50, T_B,
   concentraciones libres y fracciones frente a lo impreso. Una tabla **reproduce** si el 100 %
   de las celdas está a ≤ 1 unidad del último dígito impreso y al menos el 98 % a ≤ 0.5
   unidades (redondeo puro). Se prueban las constantes del SI y las del artículo.
2. **R09a** ([TMA] = 1 mol/L; A = receptor, B = Cl⁻, C = TMA): BC⁰₅₀ = 1.03·10⁻² (1) y
   6.7·10⁻⁴ (2) dentro de media unidad del último dígito del valor central; cociente
   BC50(1)/BC50(2) = 15.4 ± 0.05 en todo x ∈ [0, 0.98]. Con T_TMA = T_Cl: máximo del cociente
   16.5 ± 0.05 (se reporta su posición, "al 50 %") y 4.6 ± 0.05 en x = 0.
3. **R09b** (T_X = T_Q): BC⁰₅₀ de X 5.0·10⁻³ (R) y 1.05·10⁻² (D); de Q 1.95·10⁻¹ (B) y
   1.19·10⁻² (D), dentro de media unidad del último dígito del valor central.

Las incertidumbres entre paréntesis de los artículos (p. ej. 1.03(3)) no se reproducen: solo los
valores centrales.

## 5. Resultados por caso y nivel

Criterios: pre-registro del commit **`a39f0c0`** (§4), sin cambios posteriores. Resultados de
`python -m validation.literature_benchmark run --json validation/results/literature/run.json`
en el commit `f93bfcd` (con las dos correcciones de §6.4), en el entorno fijado por
`constraints.txt` (Python 3.14.9, NumPy 2.4.2, SciPy 1.17.1, pandas 3.0.1). El análisis de
sensibilidad (§4.4) se hizo antes con NumPy 2.5.4 y SciPy 1.18.1; las líneas base y las pruebas
usan el entorno fijado. Toda cifra de referencia es de la página de resultados del SI citada en
cada fila.

### 5.1 Resumen

| Nivel | Pasan | Casos que fallan |
|---|---|---|
| 0 (diagnóstico: RMS con las constantes de referencia fijas) | 8/9 (\|Δrel\| ≤ 2·10⁻⁶) | `r09a-qx-dil` (+7.9 %) |
| 1 (RMS) | 6/9 | `r09a-p2-106` (+10.1 %), `r09a-qx-dil` (+7.2 %), `r09a-r1-3sp` (+28.0 %) |
| 2 (log β, todas las constantes del caso) | 2/9 | todos los demás; pasan 15 de 28 constantes refinadas |
| 3 (SE, todas las constantes del caso) | 5/9 | `r09a-p2-058`, `r09a-p2-106`, `r09a-r1-11-3n`, `r09a-r1-3sp` |
| 4 desplazamientos dentro del error | 3 casos 13/13, 4/4, 2/2 (y 2/2 en `qx-dil`); 34/40, 32/40, 2/6, 2/9 en el resto | — |
| 4 correlaciones ±0.05 | solo `r09b-b-qx` (1/1) | ver §6.4 (c) |

**Clasificación:** reproduce **2** (`r09a-r1-11-me`, `r09b-b-qx`); parcial **4**
(`r09a-r1-5sp`, `r09b-r-qx`, `r09a-p2-058`, `r09a-r1-11-3n`); no reproduce **3**
(`r09a-p2-106`, `r09a-r1-3sp`, `r09a-qx-dil`); no soportado **1** (`r09b-d-qx`).

Lectura en una frase: **[Defecto comprobado] no hay ninguno en el modelo químico ni en el
cálculo analítico de HM fit**; los dos casos de un núcleo con datos consistentes reproducen la
referencia en todos los niveles, y los casos de varios núcleos no la reproducen porque el
ajuste RMN de HM fit **pondera cada señal por 1/σ_j²** (IRLS) y HypNMR usa pesos unitarios
(§6.1). Con pesos unitarios, el modelo directo y el cálculo analítico de HM fit reproducen la
referencia en 8 de 9 casos (§6.1, diagnóstico). El noveno, `r09a-qx-dil`, tampoco se reproduce
con un cálculo independiente (§6.2).

### 5.2 Tabla por caso

Generada por `validation.literature_benchmark` (`markdown_table`), con la fuente de cada caso.
"L2"/"L3": constantes que pasan / refinadas. "SE HM fit" es el SE analítico del resultado del
ajuste (pesos IRLS); "SE pestaña Errors" es el del método analítico de la pestaña Errors (pesos
unitarios, afectado por §6.4 (c)); σ es el de la pestaña Errors.

| Caso | Fuente (SI) | L0 RMS fijo Δrel | L1 RMS HM / ref (Δrel) | L2 log β | L3 SE | L4 desplaz. | L4 correl. | Clasificación |
|---|---|---|---|---|---|---|---|---|
| `r09a-p2-058` | R09a S12, S13, S14 | -1.6e-07 | 4.993e-04 / 4.936e-04 (+1.2%) | 3/5 | 0/5 | 34/40 | 0/10 | parcial |
| `r09a-p2-106` | R09a S17, S18, S19 | +2.0e-08 | 8.274e-04 / 7.515e-04 (+10.1%) | 1/5 | 0/5 | 32/40 | 3/10 | no reproduce |
| `r09a-qx-dil` | R09a S10, S11 | +7.9e-02 | 1.012e-03 / 9.439e-04 (+7.2%) | 0/1 | 1/1 | 2/2 | — | no reproduce |
| `r09a-r1-11-3n` | R09a S4, S6 | -2.3e-08 | 6.713e-03 / 6.570e-03 (+2.2%) | 0/1 | 0/1 | 2/6 | — | parcial |
| `r09a-r1-11-me` | R09a S4, S5 | +1.9e-06 | 1.809e-04 / 1.809e-04 (+0.0%) | 1/1 | 1/1 | 2/2 | — | reproduce |
| `r09a-r1-3sp` | R09a S7, S8 | +7.5e-08 | 9.382e-03 / 7.331e-03 (+28.0%) | 0/3 | 0/3 | 2/9 | 0/3 | no reproduce |
| `r09a-r1-5sp` | R09a S7, S9 | +1.9e-06 | 2.496e-04 / 2.489e-04 (+0.3%) | 4/5 | 5/5 | 13/13 | 2/10 | parcial |
| `r09b-b-qx` | R09b S13, S14 | -8.0e-08 | 5.135e-04 / 5.135e-04 (-0.0%) | 2/2 | 2/2 | 4/4 | 1/1 | reproduce |
| `r09b-d-qx` | R09b S15, S16, S17 | — | — | — | — | — | — | no soportado |
| `r09b-r-qx` | R09b S9, S10 | +1.9e-06 | 2.496e-04 / 2.489e-04 (+0.3%) | 4/5 | 5/5 | 13/13 | 2/10 | parcial |

#### `r09a-p2-058` — R09a, SI S12, S13, S14; R09a Table 2

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| PQX | 7.1737 | 7.1888 | -0.0151 | 0.0226 | 0.0159 | 0.0205 | -22.4% ✗ | 0.0301 |
| PX | 3.5969 | 3.5777 | +0.0192 ✗ | 0.0050 | 0.0100 | 0.0137 | -27.0% ✗ | 0.0172 |
| QX | 4.0604 | 4.0632 | -0.0028 | 0.0050 | 0.0104 | 0.0144 | -27.7% ✗ | 0.0227 |
| P2X | 6.2387 | 6.2558 | -0.0171 | 0.0370 | 0.0254 | 0.0353 | -27.9% ✗ | 0.0442 |
| P2QX | 9.5127 | 9.4656 | +0.0471 ✗ | 0.0457 | 0.0269 | 0.0376 | -28.4% ✗ | 0.0379 |

RMS: HM fit 4.99318e-04, ref. 4.93564e-04 (+1.17%). σ (pestaña Errors, ν = 210): HM fit 5.42569e-04, ref. 5.43428e-04 (-0.16%).

#### `r09a-p2-106` — R09a, SI S17, S18, S19

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| PQX | 7.1914 | 7.1419 | +0.0495 ✗ | 0.0061 | 0.0193 | 0.0244 | -21.1% ✗ | 0.0363 |
| PX | 3.6073 | 3.6100 | -0.0027 | 0.0054 | 0.0159 | 0.0216 | -26.4% ✗ | 0.0299 |
| QX | 4.0937 | 4.0724 | +0.0213 ✗ | 0.0050 | 0.0082 | 0.0173 | -52.6% ✗ | 0.0315 |
| P2X | 6.1246 | 6.0807 | +0.0439 ✗ | 0.0123 | 0.0315 | 0.0491 | -35.9% ✗ | 0.0623 |
| P2QX | 9.5335 | 9.5975 | -0.0640 ✗ | 0.0299 | 0.0317 | 0.0274 | +15.7% ✗ | 0.0357 |

RMS: HM fit 8.27416e-04, ref. 7.51522e-04 (+10.10%). σ (pestaña Errors, ν = 212): HM fit 9.13759e-04, ref. 8.26770e-04 (+10.52%).

#### `r09a-qx-dil` — R09a, SI S10, S11

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| QX | 4.0579 | 4.0508 | +0.0071 ✗ | 0.0050 | 0.0133 | 0.0123 | +8.5% | 0.0133 |

RMS: HM fit 1.01188e-03, ref. 9.43854e-04 (+7.21%). σ (pestaña Errors, ν = 22): HM fit 1.07867e-03, ref. 1.00615e-03 (+7.21%).

#### `r09a-r1-11-3n` — R09a, SI S4, S6

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| RG | 2.5699 | 2.3781 | +0.1918 ✗ | 0.0295 | 0.0031 | 0.1179 | -97.4% ✗ | 0.1683 |

RMS: HM fit 6.71307e-03, ref. 6.57020e-03 (+2.17%). σ (pestaña Errors, ν = 50): HM fit 7.22992e-03, ref. 7.00660e-03 (+3.19%).

#### `r09a-r1-11-me` — R09a, SI S4, S5

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| RG | 2.5700 | 2.5700 | -0.0000 | 0.0050 | 0.0031 | 0.0031 | +0.9% | 0.0042 |

RMS: HM fit 1.80930e-04, ref. 1.80930e-04 (+0.00%). σ (pestaña Errors, ν = 16): HM fit 1.68820e-04, ref. 1.96246e-04 (-13.98%).

#### `r09a-r1-3sp` — R09a, SI S7, S8

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| RQX | 6.1778 | 6.5296 | -0.3518 ✗ | 0.0460 | 0.0125 | 0.1839 | -93.2% ✗ | 0.2432 |
| RX | 2.9424 | 2.1429 | +0.7995 ✗ | 0.0295 | 0.0107 | 0.1180 | -90.9% ✗ | 0.2619 |
| QX | 4.1120 | 4.2560 | -0.1440 ✗ | 0.0467 | 0.0056 | 0.1869 | -97.0% ✗ | 0.1990 |

RMS: HM fit 9.38169e-03, ref. 7.33064e-03 (+27.98%). σ (pestaña Errors, ν = 102): HM fit 1.00040e-02, ref. 7.74201e-03 (+29.22%).

#### `r09a-r1-5sp` — R09a, SI S7, S9; R09a Table 1

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| RQX | 6.1225 | 6.1213 | +0.0012 | 0.0050 | 0.0105 | 0.0101 | +4.0% | 0.0098 |
| RX | 2.9463 | 2.9461 | +0.0002 | 0.0050 | 0.0088 | 0.0085 | +3.2% | 0.0144 |
| R2 | 0.7670 | 0.7825 | -0.0155 ✗ | 0.0067 | 0.0250 | 0.0267 | -6.4% | 0.0305 |
| R2X | 4.4863 | 4.5125 | -0.0262 | 0.0362 | 0.0442 | 0.0457 | -3.2% | 0.0464 |
| QX | 4.0905 | 4.0897 | +0.0008 | 0.0050 | 0.0052 | 0.0051 | +1.8% | 0.0086 |

RMS: HM fit 2.49588e-04, ref. 2.48946e-04 (+0.26%). σ (pestaña Errors, ν = 96): HM fit 2.50638e-04, ref. 2.70845e-04 (-7.46%).

#### `r09b-b-qx` — R09b, SI S13, S14; R09b Table 2

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| BQX | 4.7668 | 4.7668 | -0.0000 | 0.0174 | 0.0208 | 0.0208 | +0.2% | 0.0203 |
| BQ | 1.3733 | 1.3733 | +0.0000 | 0.0050 | 0.0071 | 0.0071 | -0.1% | 0.0069 |

RMS: HM fit 5.13535e-04, ref. 5.13535e-04 (-0.00%). σ (pestaña Errors, ν = 37): HM fit 5.14842e-04, ref. 5.51625e-04 (-6.67%).

#### `r09b-r-qx` — R09b, SI S9, S10; R09b Table 1

| Especie | log β HM fit | log β ref. | Δ | tol. L2 | SE HM fit | SD ref. | ΔSE rel. | SE pestaña Errors |
|---|---|---|---|---|---|---|---|---|
| RQX | 6.1225 | 6.1213 | +0.0012 | 0.0050 | 0.0105 | 0.0101 | +4.0% | 0.0098 |
| RX | 2.9463 | 2.9461 | +0.0002 | 0.0050 | 0.0088 | 0.0085 | +3.2% | 0.0144 |
| R2 | 0.7670 | 0.7825 | -0.0155 ✗ | 0.0067 | 0.0250 | 0.0267 | -6.4% | 0.0305 |
| R2X | 4.4863 | 4.5125 | -0.0262 | 0.0347 | 0.0442 | 0.0457 | -3.2% | 0.0464 |
| QX | 4.0905 | 4.0897 | +0.0008 | 0.0050 | 0.0052 | 0.0051 | +1.8% | 0.0086 |

RMS: HM fit 2.49588e-04, ref. 2.48946e-04 (+0.26%). σ (pestaña Errors, ν = 96): HM fit 2.50638e-04, ref. 2.70845e-04 (-7.46%).

## 6. Discrepancias, hipótesis revisadas y capacidades faltantes

Regla 5: (1) transcripción, (2) especificación del modelo (constantes fijas, asignación de
señales, pesos), (3) defecto de HM fit. Diagnósticos versionados:
`python -m validation.literature_benchmark diagnose --json validation/results/literature/diagnose.json`.

### 6.1 Casos de varios núcleos: pesos IRLS de HM fit frente a pesos unitarios

Casos: `r09a-r1-5sp`, `r09b-r-qx`, `r09a-p2-058` (parcial), `r09a-r1-11-3n` (parcial),
`r09a-p2-106`, `r09a-r1-3sp` (no reproduce).

1. **Transcripción: descartada.** Dos pasadas idénticas, conteos e identidad de grados de
   libertad (§3) y, sobre todo, el **nivel 0**: con las constantes de referencia fijas, el RMS
   de HM fit coincide con el impreso con |Δrel| ≤ 2·10⁻⁶ en los seis casos.
2. **Especificación: pesos.** El ajuste RMN de HM fit hace hasta 5 ciclos IRLS y pondera cada
   señal por 1/max(σ_j, 10⁻⁴)², sin opción para desactivarlo (§2). Evidencia:
   - El objetivo ponderado de HM fit (con sus propios pesos finales) es menor en su solución
     que en la de referencia en los seis casos (p. ej. 1.000 frente a 8.43 en `r09a-r1-11-3n`;
     0.953 frente a 0.975 en `r09a-p2-058`; 1.000 frente a 1.004 en `r09a-r1-5sp`).
   - Arrancar el ajuste **en** los valores de referencia lleva al mismo punto que el arranque
     pre-registrado (|Δ log β| ≤ 5·10⁻⁴): HM fit converge al mínimo de su objetivo. **El
     optimizador Powell no explica la discrepancia** en estos casos.
   - **Diagnóstico con pesos unitarios** (no es un resultado de HM fit): mínimos cuadrados
     (`scipy.optimize.least_squares`) sobre el modelo directo de HM fit (`NewtonRaphson` +
     `project_coeffs_block_onp_frac`). Reproduce la referencia en los seis casos: RMS idéntico a
     6 cifras, |Δ log β| ≤ 1.2·10⁻⁴ y SE/SD entre 0.92 y 1.00 (0.92–1.01 en los ocho casos con datos consistentes).
   - El SE analítico del núcleo (`compute_errors_nmr_varpro`, pesos unitarios) en las β de
     referencia, con todos los valores observados, coincide con la SD de HypNMR con ≤ 0.9 % y
     con las correlaciones con ≤ 0.001.
   - Los pesos relativos de HM fit van de 1 a 0.001 entre núcleos (`r09a-r1-3sp`,
     `r09a-r1-11-3n`: Me pesa ×1000 más que los NH) y de 1 a 0.008 en `r09a-p2-058`. En los
     modelos deliberadamente inadecuados, el ajuste sigue casi solo al núcleo mejor ajustado:
     log β(RG) de `r09a-r1-11-3n` = 2.5699, el mismo valor que el ajuste de solo Me
     (`r09a-r1-11-me`, 2.5700), frente a 2.3781 de HypNMR.
   - El SE del resultado del ajuste usa esos pesos, de ahí el nivel 3: −22 a −28 % en
     `r09a-p2-058` y −97 % en `r09a-r1-11-3n`. Con un solo núcleo el peso es una constante común
     y ambos estimadores coinciden, como muestran `r09a-r1-11-me` y `r09b-b-qx`.
3. **Defecto de HM fit: no.** **[Limitación metodológica]** El ajuste RMN de HM fit implementa
   otro estimador (mínimos cuadrados con pesos por señal estimados iterativamente), no
   seleccionable y no anunciado en el resultado. No es un error de cálculo: con los mismos pesos
   que HypNMR, el modelo y el cálculo de errores de HM fit reproducen la referencia. Pero los dos
   programas no dan el mismo número con los mismos datos, y el SE del resultado
   (pesos IRLS) no es el de la pestaña Errors (pesos unitarios). No se ha cambiado ningún método
   (regla 4); ver recomendaciones (§9).

### 6.2 `r09a-qx-dil`: el RMS publicado no es alcanzable con los datos impresos

1. **Transcripción.** Dos pasadas idénticas; 25 espectros y 25 valores como los impresos;
   identidad (σ/RMS)² = 25/22 con 6·10⁻⁹. La tabla es la de la página S10.
2. **Especificación.** Modelo de la página de resultados: Q + X ⇌ QX, Me en Q, T_Q = T_X. Un
   cálculo analítico independiente (especiación cerrada de 1:1) con la β y los desplazamientos
   publicados da RMS = 1.0203·10⁻³ frente a 9.4385·10⁻⁴ impreso (+8.1 %). El mínimo sin pesos
   para los datos impresos es RMS = 1.0119·10⁻³ en log β = 4.0579, y HM fit lo alcanza (lo
   confirma el diagnóstico con pesos unitarios). El redondeo de los datos impresos solo mueve el
   RMS un 0.56 % (§4.4).
3. **Defecto de HM fit: no** (el cálculo independiente da lo mismo).

**[Limitación metodológica]** Con los datos impresos y pesos unitarios no hay ninguna β que dé
el RMS publicado. Causas posibles, sin poder discriminarlas con el SI: el ajuste original usó
concentraciones o desplazamientos con más cifras que las impresas, pesos no unitarios, o un
conjunto de datos distinto del tabulado. Ambos valores: **HM fit log β(QX) = 4.0579 ± 0.0133;
referencia 4.0508 ± 0.0123** (R09a, SI S11); diferencia 0.0071, 0.4 SD combinadas.

### 6.3 `r09b-d-qx`: no soportado

Seis desplazamientos límite de DQ están fijos en el ajuste publicado (identidad p = 33, §3) y
HM fit no puede fijar desplazamientos (§2). La prueba es `xfail(strict=True)`. Estimación: ≈ 1–2
días para el núcleo y sus pruebas; la tabla de la GUI es de la Fase 4. Sin implementar
(regla 7). Ajustarlo con esos desplazamientos libres sería otro modelo, con otra p, y no se
compararía con la referencia.

### 6.4 Defectos encontrados con el banco

**Actualización etapa 3.6:** el defecto (c) fue corregido tras aprobación del
autor; `test_errors_tab_uses_every_observed_value` ahora pasa sin xfail. La tabla
siguiente conserva los valores históricos del diagnóstico previo a la corrección.
Véase `docs/STABILITY_PHASE3_REPORT.md`, sección «3.6 — máscara RMN».

| # | Etiqueta | Módulo | Defecto | Estado |
|---|---|---|---|---|
| a | [Defecto comprobado] | `hmfit_core/utils/errors.py` (`compute_errors_nmr_varpro`) | Con tantas señales como especies, el mapa de estequiometría (especies × señales) se transponía. El ajuste de β era correcto, pero el RMS reportado, los SE, las correlaciones y los desplazamientos usaban especies equivocadas para cada señal. Hallado con `r09a-r1-11-3n`: nivel 0 = +22 % (después −2·10⁻⁸) | Corregido (`f93bfcd`); `tests/test_nmr_square_stoichiometry.py` |
| b | [Defecto comprobado] | `hmfit_core/exports.py` (hoja "Coefficients") | Desplazamientos límite exportados sin el mapa de estequiometría del ajuste | Corregido (`f30919d`); `tests/test_nmr_export_coefficients.py` |
| c | [Defecto comprobado] | `hmfit_gui_qt/widgets/model_opt_plots.py` (pestaña Errors) | La máscara usa la columna "padre" deducida del nombre de la señal (por defecto, el receptor) y exige \|D\| > 0, en lugar de `C·S_j`. Descarta el punto T_receptor = 0 de cada titulación de la señal de la sal: ν = m − p − (nº de titulaciones). En las β de referencia, los SE del método analítico cambian hasta un 71 % (`r09a-r1-5sp`: RX +71 %, QX +69 %; `r09a-r1-11-me`: +36 %) y las correlaciones hasta 0.42 (`r09a-p2-106`) | **No corregido**: cambia números de la GUI (otro módulo). `xfail(strict=True)` en `tests/test_literature_benchmark_gui.py`. **Decisión del autor** |
| d | [Sospecha] | Pestaña Errors, bootstrap | Los métodos bootstrap de RMN parten de la misma máscara que (c) | No verificado |

**Cambios esperados** por las correcciones (a) y (b):

- (a) Cambian el RMS, `covfit`, los SE, las correlaciones y los desplazamientos de RMN **solo**
  cuando el mapa de estequiometría es cuadrado (n.º de señales = n.º de especies absorbentes).
  Las constantes ajustadas no cambian. La suite completa (434 pruebas) pasa sin cambios. Las
  celdas RMN de la campaña de la Fase 2 no pasan mapa de estequiometría
  (`validation/coverage_campaign.py`), así que no les afecta; sus réplicas guardadas no están en
  esta sesión y no se han recalculado.
- (b) Solo cambia la hoja "Coefficients" del archivo exportado en modelos con mapa de
  estequiometría; las especies sin el reactivo del núcleo quedan vacías.

### 6.5 Otras observaciones

- **Nivel 4, correlaciones.** Solo `r09b-b-qx` pasa. Hay dos causas, ambas ya descritas: los
  pesos IRLS desplazan el punto (§6.1) y la máscara de la pestaña Errors (§6.4 (c)). En las β de
  referencia y con todos los valores, HM fit reproduce las correlaciones con ≤ 0.001.
- **Nivel 4, desplazamientos.** Todos dentro del error en los casos que reproducen y en
  `r09a-r1-5sp`/`r09b-r-qx` (13/13). Los que fallan siguen a las β desplazadas por los pesos.
  HM fit no calcula errores de los desplazamientos (§2).
- **σ de la pestaña Errors**: ν coincide con m − p de HypNMR salvo por los puntos que descarta
  §6.4 (c). `n_linear` de HM fit es igual al número de desplazamientos refinados de HypNMR en
  los 9 casos.
- `r09b-d-qx`: la tabla tiene 236 valores y el ajuste original usó 235 (§3).
- R09a, Tabla 2: SD(log β P2X) = 0.03 en el artículo y 0.0353 en el SI (§3); el banco usa el SI.


## 7. Etapa D — Resolvedor de especiación (R09b, SI S20–S23)

**Significado de las columnas** (inferido del apéndice de R09a, S22–S26, y del texto de R09b;
comprobado sobre las tablas por `tests/test_literature_speciation.py`):

- El título "BC50 values (X vs R)" significa que el **reactivo analizado** A es X: su fracción
  ligada `xbound(X)` es el eje, y `BC50` = T_X es el total de A necesario para ligar el 50 % del
  **analito** B = R. El tercer reactivo C = Q cumple T_Q = T_X (electroneutralidad: la columna
  `T(Q)` es igual a `BC50` en todas las filas). Análogamente, Q vs B (A = Q, B = B, C = X), X vs D
  y Q vs D.
- `[·]` son concentraciones libres. `x(A)` = [A]/T_A y `x(S)` = a_S·[S]/T_A, con a_S el
  coeficiente de A en S: fracciones del total de A ("fraction of the total anion/cation",
  figuras S1–S2). Suman 1 en todas las filas, y `xbound(A)` es la suma sobre las especies que
  contienen A y B.
- Se comprueba además que en cada fila el 50 % de B está ligado (± 1 %, por el redondeo de los
  totales).

**Método.** Cada fila se resuelve con `NewtonRaphson` (el de la GUI por defecto) y con
`LevenbergMarquardt`, a partir de los totales impresos (T_A = BC50, T_B, T_C = T_A) y de las
constantes con toda su precisión de la página de resultados (S10, S14, S16) o redondeadas del
artículo (Tablas 1–3). `python -m validation.literature_speciation --json validation/results/literature/speciation.json`.

| Tabla | Fuente | Constantes | Filas que pasan (NR / LM) | máx. Δrel conc. | máx. \|Δ\| fracción | máx. \|Δ\| xbound | Filas fallidas explicadas por el redondeo de los totales |
|---|---|---|---|---|---|---|---|
| `r09b-spec-b` | R09b SI S21 | página de resultados | 39/50 / 39/50 | 0.87% | 0.0010 | 0.0010 | 11/11 |
| `r09b-spec-b` | R09b SI S21 | artículo (redondeadas) | 39/50 / 39/50 | 0.86% | 0.0010 | 0.0010 | 11/11 |
| `r09b-spec-d-q` | R09b SI S23 | página de resultados | 43/50 / 43/50 | 0.67% | 0.0010 | 0.0010 | 7/7 |
| `r09b-spec-d-q` | R09b SI S23 | artículo (redondeadas) | 42/50 / 42/50 | 0.71% | 0.0008 | 0.0008 | 8/8 |
| `r09b-spec-d-x` | R09b SI S22 | página de resultados | 44/50 / 44/50 | 0.67% | 0.0009 | 0.0009 | 6/6 |
| `r09b-spec-d-x` | R09b SI S22 | artículo (redondeadas) | 42/50 / 42/50 | 0.65% | 0.0009 | 0.0009 | 8/8 |
| `r09b-spec-r` | R09b SI S20 | página de resultados | 40/50 / 40/50 | 0.51% | 0.0009 | 0.0008 | 10/10 |
| `r09b-spec-r` | R09b SI S20 | artículo (redondeadas) | 37/50 / 37/50 | 0.55% | 0.0011 | 0.0011 | 13/13 |

- **Criterio pre-registrado estricto** (0.5 % y 0.0005): pasan 37–44 de 50 filas por tabla.
  Desviación máxima: 0.87 % en concentraciones libres y 0.0011 en fracciones.
- **Las filas que fallan se explican por el redondeo de las entradas.** Los totales están
  impresos con 3 cifras significativas, es decir, con hasta 0.5 % de incertidumbre, la misma
  magnitud que la tolerancia. Para cada fila que falla se prueban 2000 perturbaciones de los
  totales dentro de media unidad del último dígito, manteniendo T_C = T_A: en **todas**
  (34 de 34 con las constantes del SI, 40 de 40 con las del artículo) alguna perturbación pone
  la fila **entera** dentro de tolerancia. Ninguna fila queda sin explicar.
- **NR y LM coinciden** a 10⁻⁶ relativo en todas las filas. Solo en la fila con T_B = 0 cada
  solver deja un valor despreciable distinto (10⁻¹² a 10⁻³³) para el reactivo ausente.
- **Constantes usadas por el programa original.** Ambas colecciones reproducen las tablas
  dentro del redondeo. La fila x = 0 (BC⁰₅₀, que no depende del total de B) no discrimina:
  R: 4.988·10⁻³ (SI) y 4.994·10⁻³ (artículo), ambos impresos como 4.99E-03; B: 0.1947 y 0.1948,
  ambos 1.95E-01. Con las constantes del SI pasan algo más de filas en 3 de las 4 tablas
  (40/37, 44/42, 43/42; empate 39/39 en B), pero no basta para decidir.
- **[Defecto comprobado] del resolvedor: ninguno.** El resolvedor de balances de HM fit
  reproduce las cuatro tablas dentro de su precisión impresa.

### 7.1 Etapa E — BC50 y BC⁰₅₀ (`hmfit_core/bc50.py`)

`python -m validation.literature_bc50 --json validation/results/literature/bc50.json`; criterios
de §4.7 (commit `2aea949`, anterior a la comparación); pruebas en `tests/test_literature_bc50.py`.
La implementación reproduce exactamente las expresiones A23, A24, A25, A28 y A30 del apéndice
de R09a (`tests/test_bc50.py`).

**Tablas S20–S23 de R09b** (rejilla x exacta, T_C = T_A), celdas: BC50, T_B, T_C,
concentraciones libres y fracciones:

| Tabla | Constantes del SI: ≤ 0.5 u / ≤ 1 u | Constantes del artículo: ≤ 0.5 u / ≤ 1 u | Clasificación (SI) |
|---|---|---|---|
| `r09b-spec-r` (S20) | 530/550 (96.4 %) / 550/550 | 319/550 / 473/550 (peor 2.2 u) | no reproduce (< 98 %) |
| `r09b-spec-b` (S21) | 477/500 (95.4 %) / 500/500 | 467/500 / 500/500 | no reproduce (< 98 %) |
| `r09b-spec-d-x` (S22) | 535/550 (97.3 %) / 550/550 | 470/550 / 550/550 | no reproduce (< 98 %) |
| `r09b-spec-d-q` (S23) | 539/550 (98.0 %) / 550/550 | 458/550 / 549/550 | **reproduce** |

"u" es la unidad del último dígito impreso. Con las constantes del SI, todas las celdas están a
≤ 0.55 u. **Las tablas se calcularon con las constantes de la página de resultados (o con más
cifras), no con las redondeadas del artículo**: con estas hay celdas a más de 1 u (hasta 2.2 u en
`r09b-spec-r`), aunque los pies de figura citan las Tablas 1–3. Esto responde también a la
pregunta de §7, que la fila x = 0 no podía decidir.

**[Limitación metodológica]** El 2.7–4.6 % de las celdas de tres tablas está entre 0.50 y 0.55 u,
así que no se cumple el umbral pre-registrado del 98 % (no se ha cambiado). No se explica por la
precisión de las constantes impresas: ni log₁₀ del β impreso ni 40 perturbaciones de log β dentro
de ±5·10⁻⁵ bajan la peor celda de 0.53 u. Causas posibles, sin poder discriminarlas: la tolerancia
del resolvedor del programa original (≈ 10⁻⁴ relativo bastaría), su forma de redondear o
constantes con más cifras que las impresas.

**Valores de los artículos** (constantes del SI; con las del artículo fallan los cuatro de R09a):

| Magnitud | Fuente | Publicado | HM fit | Resultado |
|---|---|---|---|---|
| BC⁰₅₀ receptor 1, [TMA] = 1 M | R09a, texto y Fig. 4 | 1.03(3)·10⁻² | 1.0337·10⁻² | ✓ |
| BC⁰₅₀ receptor 2, [TMA] = 1 M | R09a | 6.7(3)·10⁻⁴ | 6.662·10⁻⁴ | ✓ |
| BC50(1)/BC50(2), [TMA] = 1 M | R09a | 15.4, constante en x | **15.52**, constante en x | ✗ (`xfail` estricto) |
| Máximo del cociente, T_TMA = T_Cl | R09a | 16.5, al 50 % | 16.48 en x = 0.50 | ✓ |
| Cociente en x = 0, T_TMA = T_Cl | R09a | 4.6 | 4.64 | ✓ |
| BC⁰₅₀ receptor 1 sin pares iónicos (T_TMA = T_Cl, x = 0) | R09a, texto | 1.10·10⁻³ | 1.103·10⁻³ | ✓ (adicional) |
| BC⁰₅₀ de X, receptor R | R09b, Fig. 2 | 5.0(1)·10⁻³ | 4.988·10⁻³ | ✓ |
| BC⁰₅₀ de X, receptor D | R09b | 1.05(2)·10⁻² | 1.050·10⁻² | ✓ |
| BC⁰₅₀ de Q, receptor B | R09b, Fig. 1 | 1.95(9)·10⁻¹ | 1.947·10⁻¹ | ✓ |
| BC⁰₅₀ de Q, receptor D | R09b | 1.19(3)·10⁻² | 1.187·10⁻² | ✓ |

Sobre el cociente 15.4: ambos valores se reportan. HM fit lo obtiene constante en x, como dice el
artículo, pero igual a 15.52 (15.70 con las constantes del artículo). Una causa posible: el
cociente de los BC⁰₅₀ ya redondeados, 1.03·10⁻²/6.7·10⁻⁴ = 15.37 ≈ 15.4.

**Limitación del descriptor (R09a, nota 18).** Si hay especies multinucleares en el analito B que
contienen A (p. ej. R2X en "X vs R"), `bc50_curve` siempre añade un aviso: su peso en BC50 depende
de x_A y de T_B, y BC⁰₅₀ las ignora. Si ninguna especie mononuclear en B contiene A (caso AB₂C,
ec. A28), BC⁰₅₀ es indefinido y se devuelve NaN con aviso. Nunca devuelve un valor sin aviso. Las
dos condiciones para el tercer reactivo están implementadas: [C] libre fija (1 mol/L por defecto)
y electroneutralidad (T_C = T del contraion, que puede ser A o B).

## 8. Variabilidad entre titulaciones (solo reportada)

Diferencias expresadas en SD combinadas, z = Δ / √(SD₁² + SD₂²). Sin cambios metodológicos.
**Advertencia:** `r09a-p2-058` y `r09a-p2-106` comparten la titulación de 0.15 mM, así que no
son independientes y z sobrestima la independencia. Igual con `r09a-r1-5sp`, que comparte con
el resto el mismo lote de TMAC.

**Receptor 2, 0.15 + 0.58 mM (`r09a-p2-058`) frente a 0.15 + 1.06 mM (`r09a-p2-106`)**
(R09a, SI S13 y S18):

| Especie | Referencia, 058 | Referencia, 106 | Δ (106 − 058) | z (ref.) | HM fit, 058 | HM fit, 106 | z (HM fit) |
|---|---|---|---|---|---|---|---|
| PQX | 7.1888 ± 0.0205 | 7.1419 ± 0.0244 | −0.0469 | −1.47 | 7.1737 ± 0.0159 | 7.1914 ± 0.0193 | +0.71 |
| PX | 3.5777 ± 0.0137 | 3.6100 ± 0.0216 | +0.0323 | +1.26 | 3.5969 ± 0.0100 | 3.6073 ± 0.0159 | +0.56 |
| QX | 4.0632 ± 0.0144 | 4.0724 ± 0.0173 | +0.0092 | +0.41 | 4.0604 ± 0.0104 | 4.0937 ± 0.0082 | +2.51 |
| P2X | 6.2558 ± 0.0353 | 6.0807 ± 0.0491 | −0.1751 | −2.90 | 6.2387 ± 0.0254 | 6.1246 ± 0.0315 | −2.82 |
| P2QX | 9.4656 ± 0.0376 | 9.5975 ± 0.0274 | +0.1319 | +2.84 | 9.5127 ± 0.0269 | 9.5335 ± 0.0317 | +0.50 |

**log β(QX)** por dilución (`r09a-qx-dil`, SI S11) y por ajuste global (SI S9, S13, S18):

| Comparación | A | B | Δ (B − A) | z (ref.) | z (HM fit) |
|---|---|---|---|---|---|
| `r09a-qx-dil` vs `r09a-r1-5sp` | 4.0508 ± 0.0123 | 4.0897 ± 0.0051 | +0.0389 | +2.92 | +2.28 |
| `r09a-qx-dil` vs `r09a-p2-058` | 4.0508 ± 0.0123 | 4.0632 ± 0.0144 | +0.0124 | +0.65 | +0.15 |
| `r09a-qx-dil` vs `r09a-p2-106` | 4.0508 ± 0.0123 | 4.0724 ± 0.0173 | +0.0216 | +1.02 | +2.29 |
| `r09a-r1-5sp` vs `r09a-p2-058` | 4.0897 ± 0.0051 | 4.0632 ± 0.0144 | −0.0265 | −1.73 | −2.59 |
| `r09a-r1-5sp` vs `r09a-p2-106` | 4.0897 ± 0.0051 | 4.0724 ± 0.0173 | −0.0173 | −0.96 | +0.33 |

**[Limitación metodológica]** Con los valores de referencia, las especies P2X y P2QX difieren
≈ 2.9 SD combinadas entre los dos ajustes del receptor 2, y log β(QX) por dilución y por el
ajuste global del receptor 1 difieren 2.9 SD. El texto de R09a describe el acuerdo de 4.05 con
el ajuste global como "excellent agreement"; aquí solo se reporta la cifra. Estas diferencias
son mayores que las SD de una sola titulación, y el patrón es compatible con un error
sistemático por titulación (p. ej., en la concentración de la sal), cuya SD no capturan los
ajustes. Es evidencia para la Fase 3 (factor de concentración por titulación); no se ha
cambiado nada.

## 9. Recomendaciones para la Fase 3

1. **Pesos en RMN.** Ofrecer pesos unitarios (estimador de HypNMR) y hacer explícito el IRLS
   por señal: opción visible, pesos finales y número de ciclos en el resultado. Decisión del
   autor: cuál es el valor por defecto. Con pesos unitarios, el banco predice que los seis
   casos de §6.1 pasarían a reproducir. Al cambiar, sus `xfail(strict=True)` fallarán (XPASS)
   y habrá que retirarlos y actualizar la línea base en un commit "Update literature baseline".
2. **Un solo SE.** Hoy el resultado del ajuste (pesos IRLS) y la pestaña Errors (pesos
   unitarios, con la máscara de §6.4 (c)) dan SE distintos para el mismo ajuste. El motor
   unificado debería usar los pesos del estimador y la máscara del ajuste (`C·S_j`). Con eso,
   el cálculo analítico del núcleo ya reproduce a HypNMR (≤ 0.9 % en SE, ≤ 0.001 en
   correlaciones).
3. **§6.4 (c)**: decidir si se corrige ya o dentro del motor unificado; la prueba xfail estricta
   avisará cuando se corrija.
4. **Desplazamientos límite fijos** (`r09b-d-qx`, §6.3) y **errores de los desplazamientos** en
   el resultado.
5. **Levenberg–Marquardt.** En estos casos Powell llega al mínimo de su objetivo (§6.1), pero la
   dispersión de §4.4 mezcla el redondeo con el punto de parada de Powell. Con LM, la línea base
   (|Δ log β| ≤ 10⁻⁴) dirá si converge al mismo mínimo; si no, la actualización requiere
   justificación.
6. **Factor de concentración por titulación** (fuera de alcance aquí): §8.

## 10. Reproducción y CI

```sh
python -m validation.literature_extract --pdf-dir DIR --write          # requiere los PDF y pdfplumber
python -m validation.literature_benchmark sensitivity --n 50 --workers 30 --json validation/results/literature/sensitivity.json
python -m validation.literature_benchmark tolerances validation/results/literature/sensitivity.json --write
python -m validation.literature_benchmark run --json validation/results/literature/run.json --md TABLA.md
python -m validation.literature_benchmark diagnose --json validation/results/literature/diagnose.json
python -m validation.literature_speciation --json validation/results/literature/speciation.json
python -m validation.literature_bc50 --json validation/results/literature/bc50.json
python -m validation.literature_benchmark baseline --write           # solo con "Update literature baseline"
```

Pruebas: `tests/test_literature_benchmark.py` (casos, nivel 0 y línea base),
`tests/test_literature_benchmark_gui.py` (caminos de la GUI), `tests/test_literature_speciation.py`
(etapa D), `tests/test_literature_bc50.py` y `tests/test_bc50.py` (etapa E),
`tests/test_nmr_square_stoichiometry.py`, `tests/test_nmr_export_coefficients.py`.
Tiempo total del banco ≈ 50 s: corre completo en cada push (workflow `tests`), sin subconjunto
de humo. Los PDF no están versionados (`git ls-files '*.pdf'` vacío antes de cada push).

Ejecución de CI (workflow `tests`, commit `07b8f58`, los cuatro trabajos en verde: Linux y Windows, Python 3.11 y 3.14; el banco corre en los cuatro, salvo la prueba de GUI en Windows): <https://github.com/PJGV333/GUI_HM_fit/actions/runs/42406298647>. Ejecución anterior, sin la etapa E: <https://github.com/PJGV333/GUI_HM_fit/actions/runs/42404783942>.
