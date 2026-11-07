# HM fit — Banco de validación externa con datos publicados

> **Estado:** etapas A (capacidades y extracción) y B (pre-registro) completas. Las secciones de
> resultados se añaden **después** del commit del pre-registro, sin modificar las secciones 1–5.
> Etiquetas, como en las fases anteriores: **[Defecto comprobado]** (reproducido con un caso
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
