# HM fit — Fase 2: validación estadística de las incertidumbres

> **Estado (2026-11-05): etapas 2.0, 2.1 y 2.5 completas. La campaña 2.2 queda aplazada, por
> decisión del autor, a una campaña final de validación tras las fases siguientes. Las decisiones
> 2.3 y 2.4 quedan sin decidir y se mantienen los valores por defecto actuales, con su aviso.** Etiquetas: **[Defecto comprobado]**
> (reproducido con un caso concreto), **[Limitación metodológica]**, **[Sospecha]** (no
> verificado). Todas las cifras proceden de scripts versionados con semilla; el comando
> exacto acompaña a cada una. El informe anterior de esta fase se conserva sin cambios
> como **Anexo A**.

Rama de trabajo: `fix/stable-phase2-completion`. Contiene `fix/uncertainty-reliability`
(`14a6f10`) y `main` (`5077e2e`) como ancestros: integrarla en `fix/uncertainty-reliability`
es un avance rápido (*fast-forward*), sin conflictos.

## 1. Resumen: problemas y estado

| # | Etiqueta | Ámbito | Problema | Estado |
|---|---|---|---|---|
| 1 | Defecto comprobado | Cinética (API de grafo) | `KineticReactionEdge` usaba `super()` sin argumentos en una dataclass `slots=True`: `TypeError` en Python < 3.14 | Corregido (`24e2335`) |
| 2 | Defecto comprobado | Potenciometría | Con un pKa fijo, el análisis de errores lanzaba `IndexError`: sin tabla de errores ni hojas de error en la exportación | Corregido; el pKa fijo se muestra como fijo (`95e7092`) |
| 3 | Defecto comprobado | Espectroscopía, RMN, cinética, ácido-base | Convenciones de incertidumbre incumplidas: correlaciones de fijos/no identificables = 0 o 1; SE finito con ν ≤ 0 en los bootstrap; cinética truncaba ν a 1; full-refit con SE finito para no identificables | Corregido en los tres motores (`390ae95`, `996a98e`, `88a43fb`); matriz de 105 casos |
| 4 | Defecto comprobado | RMN | `ValueError: shape mismatch` en el análisis de errores cuando una constante no es identificable | Corregido (`390ae95`) |
| 5 | Defecto comprobado | Espectroscopía y RMN (GUI) | Valor inicial no numérico → 1.0 en silencio; límite no numérico → sin límite | Rechazado con un mensaje legible (`6e62398`) |
| 6 | Defecto comprobado | Cinética (GUI) | El combo "Algorithm" ofrecía opciones que no cambiaban el método; "multistart" forzaba 10 arranques | Una sola opción real; configuraciones antiguas conservan su comportamiento (`b70fbae`) |
| 7 | Defecto comprobado | Errores (GUI) | Cabecera "log10K" también en ácido-base; celdas indefinidas vacías o "nan" sin motivo | Etiqueta real (pKa / log β / log10K), "—" con causa (`ce99054`, `ae8b6b2`) |
| 8 | Defecto comprobado | Ácido-base (GUI y docs) | Los ejemplos del editor de ecuaciones y de `ACID_BASE_POTENTIOMETRY.md` describían una escalera invertida (pKa=4.50 para HL, 8.90 para H2L); los valores iniciales por defecto eran crecientes (5, 7, 9) | Corregido; advertencia de orden invertido (`582b42e`, `88b241c`) |
| 9 | Defecto comprobado | Pruebas y validación | Los casos ácido-base de la Fase 1 (`tests/phase1_cases.py`) y la campaña G (`validation/configs/ph/`) usaban pKa₁ < pKa₂ (sistema cooperativo) | Pruebas corregidas (`0c742e1`); lectura de la campaña G corregida en §9.3 |
| 10 | Defecto comprobado (rendimiento) | Bootstrap one-step | `_as_onp` intenta `import jax` en cada llamada (≈ 25 % del tiempo) y cada réplica recalcula la especiación (≈ 50 %) | Corregido sin cambiar resultados (`1771b58`): 5.1 s → 1.7 s por variante; verificado bit a bit (§3.1) |
| 11 | Limitación metodológica | Campaña | Con N = 200 (y aun con N = 500) una celda calibrada no puede confirmarse con su IC de Wilson | Declarado (enmienda 1, §2) |
| 12 | Limitación metodológica | Ácido-base | Un pKa fuera del intervalo de pH muestreado (casi no identificable) recibe SE finito | Abierto (§10) |
| 13 | Limitación metodológica | Estimadores de ε | El estimador (b) del encargo (Jacobiano condicionado al conjunto activo) no existe en el código | Declarado en el pre-registro; no implementado |
| 14 | Defecto comprobado | Empaquetado | `scripts/build_hmfit_pyside6_appimage.sh:196` instala sin `-c constraints.txt` | Abierto; pregunta al autor |
| — | — | Campaña 2.2 | 30 754 réplicas guardadas (47 celdas UV-Vis y 15 de RMN); el resto no se ejecutó | **Aplazada a la campaña final** por decisión del autor (§2) |

## 2. Desviaciones del pre-registro

Pre-registro: `docs/PHASE2_PREREGISTRATION.md` (commit `05ca856`).

1. **Tamaños del full-refit** (declarada en el propio registro, §5): N = 200 y
   B_refit = 200, en lugar de N ≥ 500 y B ≥ 500. Motivo: coste medido de ≈ 4 400 h-CPU con
   los tamaños del encargo.
2. **Enmienda 1** (`5c38662`, §9 del pre-registro, a petición del autor y antes de leer
   resultados). Cuando se registró solo había réplicas UV-Vis del tier 1; durante la
   supervisión se vio un resumen parcial de algunas celdas, y eso queda declarado. La
   enmienda contiene:
   - la categoría **"indeterminado"**: el IC de Wilson de la cobertura contiene 0.93 o 0.97;
   - el veredicto agregado `verdict_ci` (peor y mejor caso);
   - la constancia de que todos los bootstrap usan IC de **percentiles** (no t con SE
     bootstrap) y el analítico usa t con ν residual;
   - una ampliación de N dirigida y única para la etapa 2.3 (N = 1500 en métodos baratos,
     600 en full-refit);
   - la verificación del orden de pKa de todas las celdas ácido-base, que era correcto:
     no hubo que corregir ninguna;
   - una celda degenerada de diagnóstico, `diag-ab_nmr-inverted-25-iid`, con pKa = [4, 8].

   La rejilla pasa a 327 celdas (SHA-256 `f0723dae…`); las 326 originales no cambian.
3. **Enmienda 2** (`76d4583`, §10 del pre-registro, antes de leer resultados). Asigna cada celda
   a la decisión que alimenta (`python -m validation.coverage_plan table`):
   - **tier 1:** 2.3 y 2.4 (UV-Vis y RMN, métodos baratos);
   - **tier 2:** full-refit de UV-Vis y RMN, condicional;
   - **tier 3:** las celdas de estimadores (2.3), ácido-base, potenciometría y cinética (2.4);
   - **tier 4:** la rejilla realista con 20 y 30 puntos y los diseños 1:2 (2.4 y el umbral de
     pocos bloques);
   - **diferidas:** las celdas con ruido correlacionado y el diagnóstico, que no entran en
     ninguna regla.

   R1 (118 celdas) se reanudó sin cambios de código: el código actual reproduce bit a bit las
   réplicas guardadas.
4. **Ejecución detenida y aplazada** por decisión del autor (2026-11-05, 18:42 MST): primero se
   avanza con las fases y al final se hace una campaña de validación sobre el motor
   resultante. Se detuvieron dos pasadas:
   - la del tier 1 a las 17:24 (UV-Vis: 46 celdas completas y 1 parcial);
   - R1 a las 18:42 (RMN: 15 celdas en curso; ninguna de estimadores, ácido-base,
     potenciometría ni cinética).

   **No se leyó ninguna tabla de cobertura ni ningún agregado.** Lo no ejecutado se reporta
   como no ejecutado; no se extrapola. Las 30 754 réplicas guardadas se conservan (JSONL con
   manifiesto) y la campaña final puede reanudarlas o descartarlas. Si el motor cambia en la
   Fase 3, habrá que verificarlas antes con `validation.coverage_replay_check`.

## 3. Resultados de la campaña por técnica y método

**No ejecutada en esta fase** (aplazada, §2). Las tablas se generarán, sin edición manual, con:

```sh
python -m validation.coverage_report RUN_DIR --csv validation/results/coverage/summary.csv \
    --md docs/PHASE2_COVERAGE_TABLES.md --aggregate-csv validation/results/coverage/aggregate.csv
```

Los datos crudos (JSONL por celda y manifiesto con commit y versiones) se publican como
artefacto del workflow `campaign` y no se versionan. Casos sintéticos para comparar con
BindFit o HypNMR: `validation/results/coverage/external_cases/`. Los análisis secundarios
declarados (cobertura agregada en 8–15 puntos con prueba χ² de heterogeneidad, e IC t con SE
bootstrap frente a percentiles) están implementados en `validation/coverage_secondary.py`
y se ejecutarán con la campaña final.

### 3.1 Verificación de la optimización del one-step

- **Cambio** (`1771b58`): la especiación a k̂ se resuelve una vez por análisis en los one-step
  de espectroscopía y RMN (`_FixedPointConcentrations`), y la disponibilidad de JAX se resuelve
  una sola vez. El solver de especiación no guarda estado entre llamadas.
- **Verificación:**

  ```sh
  python -m validation.coverage_replay_check RUN_DIR --per-cell 3 --workers 30 \
      --json validation/results/coverage/replay_onestep_optimisation.json
  ```

  Recalcula 3 réplicas guardadas por celda, con sus mismas semillas, y compara est, SE, lo y
  hi de cada método con lo guardado (rtol = 10⁻⁹, atol = 10⁻¹²).
- **Resultado:** 186 réplicas de 62 celdas (UV-Vis y RMN; métodos analítico, linealizado y
  one-step por fila y por elemento), 3 720 comparaciones y **diferencia máxima 0**: salida
  idéntica bit a bit.
- **Prueba unitaria:** `tests/test_onestep_speciation_cache.py`. Compara las muestras con y
  sin caché (idénticas), comprueba que la especiación se llama 1 vez en lugar de 1 + B y que
  la importación de jax se intenta una sola vez.
- **Tiempo**, una réplica UV-Vis 1:1, 10 puntos, B = 1000: 5.1 s → 1.7 s por variante de
  one-step.
- **Límites:** ácido-base, potenciometría y cinética no tenían réplicas guardadas, así que la
  verificación no las cubre. El cambio no toca sus caminos: el one-step de cinética ya
  reutilizaba el Jacobiano y el ácido-base no tiene one-step.

## 4. Decisión: estimador de absortividades (2.3)

**Sin decidir en esta fase**, porque la campaña está aplazada (§2). Por la regla del
pre-registro ("si la evidencia no es clara, se mantiene el valor actual"), **se mantiene
`soft_penalty` (μ = 10⁻²)**. La decisión se tomará con la campaña final. Ya está establecido
lo siguiente:

- los pesos por punto están implementados y probados frente a referencias independientes
  (`tests/test_phase2_weights.py`);
- el estimador (b) no existe y no se compara;
- si la evidencia no es clara, se mantiene `soft_penalty` (μ = 10⁻²) y se presentan las
  opciones al autor.

## 5. Decisión: bootstrap por defecto (2.4)

**Sin decidir en esta fase.** Por decisión del autor ("si 2.4 no puede decidirse para alguna
técnica, se mantiene su valor por defecto actual con su aviso"), **no cambia ningún método ni
bloque por defecto** y el aviso de pocos bloques sigue en 20 puntos. El umbral de advertencia por
pocos bloques (hoy 20 puntos) se justificará con la tabla de cobertura frente al número de
puntos que genera `coverage_report`. HC2/HC3 ya se evaluaron y no se adoptan (Anexo A, §6).

## 6. Defectos en versiones publicadas

Versiones publicadas con binarios: **v0.2.0**, construida desde `main` en `247c4df` el
2026-04-24, y **v0.1.0-beta.2** (`f86936a`, 2026-03-10). Ambas se construyeron con
Python **3.11** (`release.yml` de esa versión). El `main` actual (`5077e2e`, 2026-09-11)
no tiene release; ácido-base y potenciometría (`acid_base_errors`) llegaron a `main` después
de v0.2.0 (`f14ba48`, `5095c43`, mayo de 2026).

### 6.1 Método

```sh
git archive v0.2.0 | tar -x -C /tmp/v020      # ídem con main
QT_QPA_PLATFORM=offscreen python3.11 validation/regressions/repro_published_defects.py /tmp/v020 --json OUT
```

Python 3.11.15, NumPy 2.4.2, SciPy 1.17.1, PySide6 6.10.2. Resultados versionados:
`validation/results/regressions/repro_v020.json` y `repro_main.json`. El script combina dos
tipos de comprobación:

- **Sondas directas** por los widgets de la GUI o la API.
- **La prueba de regresión que fija cada corrección**, copiada de esta rama y ejecutada contra
  el código del árbol. Los imports de nivel superior van protegidos para que un módulo ausente
  solo afecte a las pruebas que lo usan.

Cada defecto queda en uno de cuatro estados:

- **reproducido**: falla la aserción o aparece el error documentado;
- **ausente**: la prueba pasa;
- **corrección ausente (API)**: la prueba no puede llamar a la función que necesita; el
  comportamiento **no** queda verificado por la prueba;
- **no aplica**: el módulo no existe en ese árbol.

En esta rama todas salen *ausente*.

### 6.2 Resultado

"Números" significa que el defecto cambia en silencio un valor, SE, IC o correlación mostrados;
"caída" significa que algo falla de forma visible sin cambiar números.

| Id | Origen | Módulo | Defecto | Efecto | v0.2.0 | `main` |
|---|---|---|---|---|---|---|
| P2-D2a | Fase 2 | Tabla de parámetros, espectroscopía/RMN | Errata en valor o límite: valor → 1.0, límite → sin límite | **números** | reproducido (`'3x'` → 1.0; `'8x'` → None) | reproducido |
| P2-D2b | Fase 2 | Ídem | **Coma decimal** `'0,5'` → 1.0; límite `'0,1'` → sin límite | **números** | reproducido | reproducido |
| P2-D2c | Fase 2 | Tabla de parámetros, cinética | Coma decimal `'0,5'` → **0.0**; límite → sin límite | **números** | reproducido | reproducido |
| RR2 | Revisión #2 | Solver de ε UV-Vis (`soft_penalty`, por defecto) | No converge a escala molar: log K sesgado y varianza reducida | **números** | reproducido (error del solver 1.1·10⁻⁶ frente a < 10⁻⁹) | reproducido |
| RR1 | Revisión #1 | Bootstrap one-step (cinética; también espectroscopía y RMN) | Amortiguamiento λ·I sin escalar: SD ~7000× menor con k en M⁻¹ s⁻¹ | **números** | corrección ausente; **reproducido por inspección**: `jac_lin.T @ jac_lin + float(lam) * np.eye(p)` en `kinetics_model_opt_plots.py:1022`, el mismo código que en `main`, donde la revisión midió el defecto | ídem |
| RR4 | Revisión #4 | IC analíticos (todas) | z = 1.96 en lugar de t(ν) | **números** | corrección ausente; **reproducido por inspección** (`z95 = 1.96` en `model_opt_plots.py:1160` y `kinetics_model_opt_plots.py:1111`) | ídem |
| RR3 | Revisión #3 | ν en cinética y ácido-base | No resta los parámetros lineales | **números** | cinética: corrección ausente (API); ácido-base no aplica | reproducido (ácido-base RMN) |
| P1-1 | Fase 1 #1 | Cinética | Límites invertidos fijan el parámetro sin aviso | **números** | reproducido | reproducido |
| P1-4 | Fase 1 #4 | Cinética / espectroscopía | Columna nula del Jacobiano → SE = 0 (parece "fijo") | **números** | reproducido | reproducido |
| P1-5 | Fase 1 #5 | Bootstrap linealizado / one-step | Colineales o insensibles con SD finita o 0 | **números** | reproducido | reproducido |
| P1-x | Fase 1 §2.1 | Cinética | Valor inicial fuera de límites aceptado; invariantes de fijos | **números** | reproducido | reproducido |
| P1-2 | Fase 1 #2 | Cinética, fijo por máscara | El fijo se mueve a su límite inferior | **números** | corrección ausente (API) | ídem |
| P1-6 | Fase 1 #6 | Errores de espectroscopía con log β fijo | `fixed_mask` ignorado: fijo con SE ≠ 0 e inflación de los libres | **números** | corrección ausente (API); la matriz D5 lo **reproduce** (SE = 0.0052 para un fijo) | ídem |
| P1-10 | Fase 1 #10 | ν ≤ 0 | Truncado a 1 sin aviso | **números** | corrección ausente (API) | reproducido (ácido-base) |
| P1-7 | Fase 1 #7 | Ácido-base RMN/espectroscopía | `pka_fit_mask` y `pka_bounds` ignorados | **números** | no aplica | reproducido |
| P2-refit | Fase 2 (Anexo A §4) | Full-refit espectroscópico | Réplicas no convergidas aceptadas | **números** | reproducido | reproducido |
| P2-pgd | Fase 2 (Anexo A §4) | `solve_A_nnls_pgd` | Multiplicadores constantes por λ ignorados | **números** | corrección ausente (API) | ídem |
| P2-nmr12 | Fase 2 (Anexo A §7) | RMN 1:2 | Fallo del optimizador reportado como éxito | **números** | reproducido | reproducido |
| P2-D5 | Fase 2 #3 | Convenciones (matriz de 105 casos) | Correlaciones de fijos 0 o 1, SE finito con ν ≤ 0, etc. | **números** (y alguna caída) | 48 de 60 casos de la GUI | 48 GUI + 21 ácido-base |
| P2-D4 | Fase 2 #4 | Errores de RMN, constante no identificable | `shape mismatch` / SE = 0 | **v0.2.0: solo caída**; `main`: caída y números | reproducido: 4 de 4 métodos caen | 1 de 4 cae; 2 de 4 dan **SE = 0** |
| RR5 | Revisión #5 | Cinética | Fijar un parámetro → `ValueError` de límites | caída | reproducido | reproducido |
| RR7 | Revisión #7 | Bootstrap 1:1 (1 parámetro) | Todo bootstrap falla (`np.corrcoef` escalar) | caída | reproducido | reproducido |
| P2-D1 | Fase 2 #1 | API de grafos cinéticos, Python < 3.14 | `super()` en una dataclass `slots=True` | caída (la GUI no usa esa API) | reproducido | reproducido |
| P2-D3 | Fase 2 #2 | Potenciometría con pKa fijo | `IndexError` en errores | caída (el ajuste es correcto) | no aplica | reproducido |
| P1-8 | Fase 1 #8 | Contexto ácido-base | `x or []` con arreglos | caída | no aplica | reproducido |
| RR11 | Revisión #11 | Grafos con ciclos | Ciclos consistentes rechazados | caída | **ausente** (la prueba pasa) | ausente |

**Aclaración sobre el fallo de RMN (P2-D4).** En **v0.2.0 solo provoca una caída**: los cuatro
métodos lanzan `shape mismatch` y no se muestra ningún número. El cambio silencioso de números
(**SE = 0** en los bootstrap linealizado y one-step, que se lee como "parámetro fijo") se
reproduce en **`main`**, que no tiene release. La pregunta del mensaje anterior ("cambian
números en el binario publicado") mezclaba las dos columnas. En v0.2.0, los defectos que sí
cambian números en silencio son los marcados **números** con "reproducido" en esa columna.

**Coma decimal (punto 10).** Sí dispara el reemplazo. En v0.2.0, `'0,5'` como valor inicial
pasa a **1.0** en espectroscopía/RMN y a **0.0** en cinética, y un límite con coma se trata como
"sin límite". En esta rama y en `fix/v0.2.1` se rechaza con el mensaje
`K1: the initial value ('0,5') is not a number. Enter a number (use '.' as decimal separator).`

**Puntos de partida antiguos de pKa (punto 7).** `python -m validation.pka_start_check
--seeds 20` (`validation/results/pka_start_check.json`) usa datos en el orden correcto: dos
ligandos dipróticos, [8.9, 4.5] y [8.0, 4.0], en RMN, UV-Vis y potenciometría. Los ajusta
partiendo de cuatro puntos: los valores por defecto antiguos (5, 7), el ejemplo antiguo de la
documentación (4.50, 8.90), los nuevos valores por defecto (7, 5) y los valores verdaderos.
**Los 480 ajustes recuperan los valores verdaderos** (|Δ| < 0.1). Ningún caso termina con
el orden intercambiado ni en otra rama. Con estos diseños, los arranques antiguos no cambiaban
los valores ajustados. El defecto de los ejemplos era de interpretación: atribuir 4.50 a HL.

### 6.3 Aviso y versión correctora (preparados, sin publicar)

- **Borrador de aviso:** `docs/KNOWN_ISSUES_v0.2.0.md`. Recoge los módulos afectados, qué
  revisar y cómo.
- **Rama `fix/v0.2.1`**, creada desde `247c4df` (el commit de v0.2.0). Es **solo local**: sin
  push, sin tag y sin release. Contenido:
  - cherry-picks limpios de infraestructura: `30c5ceb` (dependencias y `constraints.txt`),
    `2c90413` (`tests.yml`), `8e5aa64` (release condicionado a pruebas) y `2624598`
    (ubuntu-24.04 y acciones Node 24; sin `campaign.yml`, que no existe en v0.2.0);
  - `pytest.ini` y `conftest.py` de `0485c8a`;
  - D1 (`24e2335`, una línea), necesaria para que el CI pase en 3.11;
  - **la corrección de errores silenciosos D2a/b/c**: validación de las tablas de parámetros
    de espectroscopía, RMN y cinética desde el texto escrito. Esto incluye además los límites
    invertidos de cinética (P1-1) en la GUI. Es un port aditivo: `param_validation.py`
    procede de `6e62398` y los dos métodos `validate_parameter_table` de `ce99054`/`b70fbae`;
    el cherry-pick directo choca solo con `acid_base_tab.py`, que no existe en v0.2.0;
  - las 4 fallas previas de v0.2.0 como `xfail(strict=True)`, con su causa: un `.xlsx` de
    prueba que falta en el tag, ciclos (RR11), una firma desfasada en una prueba y `absgrp`.
- **Tamaño:** 8 commits; 21 archivos, +510 / −32 líneas. En código de la aplicación, 8
  archivos, +145 / −1; el resto son CI, dependencias y pruebas.
- **Pruebas locales:** Python 3.11 y 3.14, **104 aprobadas y 4 xfail**; sin GUI (como Windows
  en CI), 83 aprobadas y 4 xfail.
- **Lo que quedó fuera**, porque su cherry-pick no aplica limpiamente (conflictos de contenido
  con el código de v0.2.0):
  - RR1–RR4 y RR7 (`7124267`, 5 conflictos);
  - P1-1/2/4/5/6 en el núcleo (`535d1f0`, 8 conflictos; `a1ef955`, 5);
  - P2-nmr12 (`0b63824`, 4);
  - las convenciones D5 (`fe1ef79`, `390ae95`, `996a98e`);
  - el full-refit espectroscópico (P2-refit) y `solve_A_nnls_pgd` (P2-pgd), en commits WIP
    mezclados.

  Portarlos exige reescribirlos sobre v0.2.0 y validarlos de nuevo: es trabajo nuevo, no un
  backport.

**v0.2.1 corrige, por tanto, 4 de los 14 defectos de "números" reproducidos o
confirmados en v0.2.0:** D2a, D2b, D2c y, solo en la GUI, P1-1. Otros 4 (RR3 en cinética,
P1-2, P1-10 y P2-pgd) no se pudieron verificar en v0.2.0, porque falta la API que usa su
prueba. El aviso debe dejarlo claro.

## 7. Fe de erratas de la Fase 1

El informe de la Fase 1 (`docs/STABILITY_PHASE1_REPORT.md`) daba por corregidos estos
puntos **en todos los caminos**. La matriz de 105 casos ejecutada sobre el estado final de
la Fase 1 (`0b63824`, Python 3.11) muestra que no lo estaban:

- **Fila 10 y §2.2 ("ν ≤ 0 → SE = NaN").** Era cierto para el SE analítico, pero:
  - en el analítico las correlaciones seguían siendo finitas (1 en la diagonal; matriz
    identidad en ácido-base) en espectroscopía, RMN y ácido-base;
  - los bootstrap linealizado, one-step y full-refit daban **SE finito** (por ejemplo,
    6.02 en espectroscopía linealizado) en espectroscopía, RMN, cinética y ácido-base.
- **Filas 4–5 y §2.2 ("no identificable → fila/columna NaN").**
  - Las correlaciones de no identificables no eran NaN en espectroscopía y cinética.
  - El full-refit daba SE finito en espectroscopía, RMN, cinética y ácido-base.
  - En ácido-base, el bootstrap linealizado caía (`pka_values contains non-finite values`).
  - El analítico de RMN caía (`shape mismatch`).
- **§2.1 ("Cov_fijo,· = 0").** La covarianza era 0, pero las **correlaciones** con fijos se
  mostraban como 0 (y la diagonal como 1), no como "no definidas", en espectroscopía, RMN,
  cinética y ácido-base analítico. En los bootstrap de cinética el SE de un fijo era
  5.8·10⁻¹⁷, no exactamente 0 (efecto cosmético).
- **§6 (validaciones numéricas).** Los tres casos ácido-base RMN usaban pKa₁ < pKa₂, es
  decir, un sistema cooperativo en la convención de HM fit. Sus cifras de la tabla de la
  §6 no describen un ácido diprótico ordinario. Se corrigieron en `0c742e1`; las pruebas
  siguen pasando con las mismas tolerancias.

Todas estas violaciones están corregidas en esta rama. La matriz pasa 105 de 105 y se
ejecuta en CI.

## 8. Convención de pKa (decisión del autor)

Se mantiene la convención interna: pKa_n = log β_n − log β_{n−1} con constantes
acumuladas de protonación. Por tanto pKa₁ (HL ⇌ L + H⁺) es el sitio **más básico**, y un
ligando no cooperativo cumple pKa₁ > pKa₂ > … Lo implementado en esta fase:

- **(a) Etiquetas.** Cada etiqueta de pKa nombra su equilibrio: tabla de valores iniciales,
  texto de resultados, resumen de potenciometría y tabla de errores. Ejemplo:
  `pKa1 (HL ⇌ L + H⁺)`.
- **(b) Orden invertido.** Hay una advertencia bajo la tabla, en el registro del ajuste
  (valores iniciales) y en el informe (valores ajustados). Los valores **no se reordenan**.
  Los valores iniciales por defecto pasan a ser una escalera decreciente.
- **(c) Documentación y exportaciones.**
  - `docs/ACID_BASE_POTENTIOMETRY.md` tiene un apartado nuevo y ejemplos corregidos.
  - En la exportación, la hoja `pKa` tiene la columna `Equilibrium` y se añade la hoja
    `pKa_convention`.

Pruebas: `tests/test_pka_convention_labels.py`, 8 pruebas (núcleo, exportación XLSX y GUI
offscreen). El selector de convención queda para la Fase 4.

## 9. Cambios esperados en resultados científicos

### 9.1 Valores ajustados

- **Sin cambios** para configuraciones válidas.
- **Cambian** cuando la tabla de parámetros de espectroscopía o RMN contenía texto no
  numérico. Antes el ajuste arrancaba en 1.0 o perdía el límite; ahora no arranca y explica
  el problema.
- **Ácido-base con valores iniciales por defecto.** Ahora son decrecientes (diprótico 7, 5;
  triprótico 9, 7, 5) en lugar de crecientes. **[Sospecha]** Con datos que determinan
  ambos pKa el mínimo es el mismo; si el arranque anterior llevaba a la rama cooperativa,
  el resultado puede cambiar. No se ha cuantificado.

### 9.2 Errores estándar e intervalos

- **Potenciometría con pKa fijo.** Ahora aparece la tabla de errores. El fijo tiene SE = 0,
  IC degenerado y correlación "—"; los libres tienen SE normal. Antes no había tabla.
- **ν ≤ 0, cualquier técnica y método.** SE e IC son NaN, mostrados como "—" con su causa,
  en lugar de cifras finitas. La cinética ya no trunca ν a 1.
- **No identificables.** SE NaN en full-refit (antes era finito) y en los bootstrap de RMN
  (antes era 0). El análisis de RMN ya no cae.
- **Correlaciones con fijos o no identificables.** "—" (NaN) en lugar de 0, o de 1 en la
  diagonal.
- **Cinética, sin covarianza.** El SE es NaN, no 0; el 0 queda reservado para "fijo".
- **Porcentaje de error.** Sigue al SE: es NaN cuando el SE no está definido.

### 9.3 Interpretación de resultados previos

La lectura de la campaña G (`docs/MONTE_CARLO_RESULTS.md` §8) del escenario RMN
"2 pKa 4.0/7.5" como "problema de diseño/condicionamiento… escapadas del mínimo"
**queda superada**. Ese escenario simula pKa₁ = 4.0 < pKa₂ = 7.5, un sistema cooperativo
en la convención de HM fit en el que los datos determinan sobre todo la suma. Los sesgos
medios de pKa₁ y pKa₂ son simétricos (−0.870 y +0.871), lo que es coherente con una suma
conservada.

### 9.4 Presentación y exportaciones

- Cabeceras de la tabla de errores: "Estimate (pKa)" o "SE (log β)" según la técnica, en
  lugar de "log10K".
- Celdas indefinidas: "—" con la causa en la ayuda emergente.
- Exportaciones:
  - columna `Note` con la causa;
  - NaN numérico en lugar de texto;
  - en ácido-base, columna `Equilibrium` y hoja `pKa_convention`.
- Cinética: el combo "Algorithm" tiene una sola opción ("Trust-region reflective
  (bounded)"); los arranques múltiples se configuran en su control propio. Una
  configuración guardada con "multistart" sigue cargando 10 arranques.

**Valores por defecto de métodos de error y del estimador de ε: sin cambios.** Dependen de
las decisiones 2.3 y 2.4, todavía pendientes.

## 10. Estado de CI

- Workflow `tests` (`.github/workflows/tests.yml`), en cada push y PR:
  - Linux, Python 3.11 y 3.14: suite completa, GUI offscreen;
  - Windows, Python 3.11 y 3.14: `-m "not gui and not slow"`.
- `release.yml` construye solo si pasa el trabajo `tests`. En esta fase no se disparó
  ningún release.
- Workflow manual `campaign` para la campaña completa.
- Recuentos tomados de los JUnit de CI (recolectadas / aprobadas / fallidas / xfail /
  omitidas):

| Ejecución | Linux 3.11 | Linux 3.14 | Windows 3.11 | Windows 3.14 |
|---|---|---|---|---|
| Antes: [42114949753](https://github.com/PJGV333/GUI_HM_fit/actions/runs/42114949753) (`0737301`, etapa 2.0) | 288 / 283 / 0 / 4 / 1 | 288 / 283 / 0 / 4 / 1 | 172 / 170 / 0 / 1 / 1 | 172 / 170 / 0 / 1 / 1 |
| [42123409391](https://github.com/PJGV333/GUI_HM_fit/actions/runs/42123409391) (`ea1557e`, 2.5) | 421 / 416 / 0 / 4 / 1 | 421 / 416 / 0 / 4 / 1 | 185 / 183 / 0 / 1 / 1 | 185 / 183 / 0 / 1 / 1 |
| Después: [42139346429](https://github.com/PJGV333/GUI_HM_fit/actions/runs/42139346429) (`d70d26c`, último commit de código) | 433 / 428 / 0 / 4 / 1 | 433 / 428 / 0 / 4 / 1 | 189 / 187 / 0 / 1 / 1 | 189 / 187 / 0 / 1 / 1 |

Los commits posteriores a `d70d26c` solo modifican documentación. Todas las ejecuciones citadas terminaron en *success*.

Entorno local de la campaña y de las reproducciones: Python 3.14.9 / 3.11.15, NumPy 2.4.2,
SciPy 1.17.1, pandas 3.0.1, PySide6 6.10.2, Matplotlib 3.10.8 (CachyOS Linux, 32 núcleos).

### 10.1 `constraints.txt` y versiones de Python

`requirements_qt.txt` y `requirements-dev.txt` fijan rangos con cota superior.
`constraints.txt` es un `pip freeze` del entorno en que pasó la suite completa: fija la
versión exacta de cada paquete, incluidas las dependencias transitivas. CI y `release.yml`
instalan con:

```sh
pip install -r requirements_qt.txt -r requirements-dev.txt -c constraints.txt
```

`-c` no instala nada por sí mismo; solo impone versiones a lo que piden los `-r`. **El mismo
archivo vale para 3.11 y 3.14** porque cada versión fijada publica ruedas para ambas
(NumPy 2.4.2, SciPy 1.17.1, pandas 3.0.1 y contourpy 1.3.3: `Requires-Python >= 3.11`;
PySide6 6.10.2: `>= 3.9, < 3.15`). CI lo verifica en Linux y en Windows con las dos
versiones.

**3.9 y 3.10 no son viables:**

- **Dependencias.** Las versiones fijadas exigen ≥ 3.11 (`Requires-Python` de NumPy, SciPy,
  pandas y contourpy). pip no encuentra distribución para 3.9 ni 3.10, así que
  `constraints.txt` no se puede cumplir.
- **Pila alternativa.** Soportar 3.10 exigiría una segunda pila, más antigua y no probada
  (NumPy ≤ 2.2, SciPy ≤ 1.15, pandas ≤ 2.3), con su propio `constraints` y su fila de CI.
  Además Python 3.10 terminó su soporte en octubre de 2026.
- **Código en 3.9.** **[Defecto comprobado por inspección]** El código no se ejecuta en
  3.9: `@dataclass(slots=True)` (`hmfit_core/graph/chemical_graph.py:47`) requiere 3.10, y
  hay anotaciones `X | None` evaluadas en ejecución (`hmfit_core/solvers/nr_conc.py:174`).
- **Código en 3.10.** **[Sospecha]** El código podría funcionar con la pila antigua (no se
  encontraron construcciones exclusivas de 3.11), pero no se ha probado.

Por eso el mínimo documentado es 3.11 (`0737301`) y no se bajaron dependencias.

## 11. Problemas abiertos y recomendaciones para la Fase 3

### Abiertos

1. **Campaña 2.2 incompleta.** Faltan el tier 1 restante (≈ 12 600 réplicas), los tiers 2–4
   y el diagnóstico. Sin ella no hay decisiones 2.3 ni 2.4.
2. **[Defecto comprobado] Rendimiento del one-step** (perfil de una réplica UV-Vis 1:1,
   10 puntos, B = 1000):
   - por método: analítico 0.65 s, linealizado 0.67 s, one-step 5.1 s por variante;
   - dentro del one-step, `_as_onp` (`hmfit_core/utils/errors.py:352`) importa `jax` en
     cada llamada: unas 28 000 importaciones fallidas, ≈ 25 % del tiempo;
   - la especiación a k̂ se recalcula en cada réplica aunque no depende de los datos
     remuestreados: ≈ 50 % del tiempo.

   Afecta también al bootstrap one-step de la GUI.
3. **[Limitación metodológica]** En ácido-base, un pKa casi no identificable (fuera del
   rango de pH) recibe SE finito: el criterio de rango del Jacobiano no lo detecta.
4. Cuatro pruebas `xfail(strict=True)` heredadas: tres de ácido-base GUI (Fase 4) y una de
   `absgrp` (Fase 3).
5. Empaquetado AppImage sin `constraints.txt` (fila 14).
6. **Casos externos 1:2.** El ajuste con la configuración por defecto de la GUI dio K₂ =
   4.15 (UV-Vis) y 0.0001 en el límite (RMN), frente a un valor verdadero de 2.5. Es la
   identificabilidad 1:2 del diseño, pendiente de comparar con BindFit y HypNMR.
7. Windows solo ejecuta pruebas no GUI en CI.

### Recomendaciones para la Fase 3

- **Optimizar** el punto 2 de la lista anterior sin cambiar resultados: comprobar `jax` una
  sola vez y calcular C(k̂) y dC/dlog K una vez por análisis. Añadir una prueba de
  equivalencia con las mismas semillas y el mismo resultado hasta el redondeo.
  Estimación: one-step de 5 s a 1–1.5 s.
- **Validar con una campaña final dedicada** (propuesta del autor; pendiente de decidir).
  Durante el desarrollo, solo la campaña *smoke* en CI. Al final, una campaña completa en
  dos etapas: N = 100 en todas las celdas y ampliación solo en las indeterminadas. Esta
  estructura debe pre-registrarse antes de ejecutarse.
- Detección de casi no identificabilidad en ácido-base: perfil de verosimilitud o umbral de
  condicionamiento, validado con la campaña.
- Unificación de los tres motores de error, prevista para la Fase 3 y no hecha en esta
  fase. La matriz de 105 casos es la red de seguridad.

### Preguntas para el autor

1. ¿Se integra `fix/stable-phase2-completion` en `fix/uncertainty-reliability`? Es un
   avance rápido de 59 commits.
2. ¿Se reanuda la campaña tal cual (varias horas) u optimización primero más campaña final
   (§11)? ¿Se aplazan los tiers 3–4?
3. ¿Se acepta el cambio de los valores iniciales por defecto de pKa a una escalera
   decreciente?
4. Los defectos D2 y D4/D5 cambian números en el binario publicado v0.2.0. ¿Se emite una
   versión correctora o un aviso a los usuarios?
5. ¿Se añade `-c constraints.txt` a los scripts de empaquetado (AppImage y Flatpak)?
6. ¿Qué criterio de identificabilidad se acepta para el diseño 1:2 de los casos externos?

---

## Anexo A. Informe anterior de la Fase 2 (sin cambios)

Texto íntegro del informe en `0b63824`, que cubría las Etapas A–H. Incluye, en §4, la
corrección original de la identificabilidad en el one-step de RMN. Se comprobó de nuevo en
esta fase: se usa el Jacobiano del ajuste base y no el de la última réplica
(`hmfit_core/utils/errors.py`, `bootstrap_one_step_nmr`; prueba
`tests/test_nmr_onestep_identifiability_base.py`).

### HM fit — Fase 2: validación estadística de incertidumbres

> **Estado: PARCIALMENTE VALIDADA (dictamen provisional B).** La investigación del
> sesgo UV-Vis (Etapas A–D) está completada y documentada en
> `docs/UVVIS_BIAS_INVESTIGATION.md`: causa establecida, sin defecto de
> implementación en el núcleo. Permanecen abiertas la ampliación del arnés
> (Etapas E–G), la comparación de métodos de incertidumbre (Etapa F), las
> campañas intermedia/confirmatoria y la Etapa H de rendimiento.

#### 1. Base y entorno

La rama `fix/stable-phase2-statistical-validation` parte de `f36d6c8`. El informe de Fase 1 está disponible en el commit `14a6f10`, pero no está presente en el árbol del commit base `f36d6c8`. La línea base se obtuvo con `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q --tb=line -p no:cacheprovider`, desde la raíz del repositorio. Se instaló `odfpy` en el entorno virtual antes de esta ejecución.

| Entorno | Versión |
|---|---:|
| Python | 3.14.8 |
| NumPy | 2.4.2 |
| SciPy | 1.17.1 |
| PySide6 | 6.10.2 |

#### 2. Suite

| Ejecución | Recolectadas | Aprobadas | Fallidas | Omitidas | Tiempo |
|---|---:|---:|---:|---:|---:|
| Línea base | 204 | 199 | 4 | 1 | 41.27 s |
| Estado anterior, pasada 1 | 219 | 214 | 4 | 1 | 34.01 s |
| Estado anterior, pasada 2 | 219 | 214 | 4 | 1 | 34.55 s |
| Estado actual, pasada 1 | 225 | 220 | 4 | 1 | 74.62 s |
| Estado actual, pasada 2 | 225 | 220 | 4 | 1 | 72.71 s |
| Rama de finalización, baseline tras integración | 225 | 220 | 4 | 1 | 42.44 s |
| Rama de finalización, correcciones parciales | 232 | 227 | 4 | 1 | 56.31 s |
| Modo absoluto y nuevos tests, suite completa | 255 | 250 | 4 | 1 | 37.20 s |
| Arnés piloto provisional, suite completa | 260 | 255 | 4 | 1 | 42.79 s |
| Resolución separable y piloto, suite completa | 261 | 256 | 4 | 1 | 49.78 s |
| Investigación de sesgo A–D (nueve pruebas nuevas), suite completa | 270 | 265 | 4 | 1 | 39.38 s |

En las dos pasadas del estado de trabajo, los identificadores de los cuatro fallos coinciden exactamente con los de la línea base:

- `tests/test_acid_base.py::test_custom_model_dimensions_start_without_forced_roles`
- `tests/test_acid_base.py::test_gui_equation_editor_generates_canonical_model`
- `tests/test_acid_base.py::test_gui_potentiometry_validation_warns_for_operational_ph`
- `tests/test_spectro_absgrp_equivalence.py::test_absgrp_shared_groups_reduce_optical_components`

Ejecutar únicamente `pytest tests/` no reproduce esa línea base porque excluye pruebas descubiertas desde la raíz.

#### 3. Pesos por punto: estado parcial

[Implementado y probado con casos sintéticos; **no validado por campaña**] La nueva opción de configuración `weights_matrix` acepta pesos estadísticos no negativos por punto y longitud de onda. Su raíz cuadrada multiplica el residuo; los pesos preexistentes por longitud de onda siguen siendo multiplicadores del residuo. Las rutas modificadas incluyen el estimador por columnas, el Jacobiano VarPro proyectado, el recuento de grados de libertad y la transferencia de pesos al contexto de incertidumbres y al refit. No se han establecido aún todas las garantías de compatibilidad.

[Defecto comprobado, corregido en el estado de trabajo] El descenso por gradiente ponderado con un número fijo de iteraciones no aseguraba la minimización en casos mal escalados: en un experimento determinista con concentraciones del orden de 10⁻³, el objetivo obtenido con 300 iteraciones fue `17.014161726420298`, frente a `13.254238295371566` de una minimización independiente. La ruta ponderada utiliza ahora un Newton semisuave proyectado con comprobación del gradiente proyectado y búsqueda de paso. Superó tres pruebas sintéticas frente a una referencia independiente y una exploración de 50 semillas con cotas, suavizado y pesos cero. Esto no equivale a validación de cobertura; el sistema denso tiene un límite explícito de 4000 coeficientes.

[Limitación metodológica] La covarianza analítica existente utiliza el estimador de absortividades sin restricciones; no debe interpretarse como una validación de la incertidumbre del estimador restringido. El Jacobiano proyectado de VarPro no es, en general, toda la derivada del residuo perfilado: la derivada completa también contiene un término en el espacio de las columnas lineales.

[Limitación de pruebas] En el entorno de esta ejecución no está instalado JAX; no se ha verificado la ruta con ese backend. Tampoco se ha completado una validación integral de todos los caminos de remuestreo y refit con pesos por punto.

#### 4. Auditoría de la rama de finalización

Se integró `fix/uncertainty-reliability` en la rama de finalización mediante un merge ordinario; se conservan los historiales del checkpoint `fb9ea9e` y del merge de Fase 1 `14a6f10`.

[Defecto comprobado, corrección parcial] El refit espectroscópico aceptaba parámetros finitos incluso si el optimizador informaba falta de convergencia, y las excepciones del solver interno se convertían en `1e50`, considerado finito. Ahora las excepciones del solver químico y lineal se clasifican, la cancelación se propaga, se exige éxito del optimizador y se reevalúa el resultado verificando parámetros, cotas y residuo finitos. Se han probado fallos de convergencia, fallo lineal y cancelación, así como un refit sintético válido; faltan pruebas específicas para todos los modos de fallo y optimizadores. El contador full-refit ahora registra intentos y causas y rechaza parámetros no finitos. Las réplicas descartadas pueden concentrarse en regiones extremas: los intervalos condicionados al éxito pueden resultar sesgados o estrechos; aún no se ha cuantificado este efecto.

[Defecto comprobado, corregido en la API numérica] En `solve_A_nnls_pgd`, multiplicadores constantes por longitud de onda distintos de uno se ignoraban aunque `mu` es una penalización absoluta. Se corrigió en esa API y se contrastó un caso analítico. En el procesador se conserva deliberadamente la ruta histórica para pesos que no varían entre puntos, pues el cambio inmediato produjo tres regresiones en las pruebas doradas de `soft_bound`. Por tanto la coherencia general pesos/regularización del procesador **sigue pendiente**; no se afirma corregida en todos los modos. En `_exact_soft_penalty_columns` la penalización se reescala relativamente a los pesos: es una convención distinta de la penalización absoluta y requiere una opción explícita y validación.

[Defectos comprobados, corregidos] El bootstrap one-step de RMN usa ahora el Jacobiano del ajuste original para clasificar identificabilidad, no el de la última réplica. Para parámetros ácido-base todos fijos, la correlación se expresa como no definida (`NaN`), manteniendo SE cero. Ambos cambios tienen regresiones específicas.

[Límite de ejecución] La suite se ejecutó con 227 aprobadas antes de añadir dos pruebas adicionales de clasificación de fallos; las nueve pruebas focalizadas de `tests/test_phase2_completion_critical.py` pasaron después. Los dos intentos posteriores de la suite completa agotaron los límites de 180 s y 300 s sin producir salida. Una prueba acotada mostró que Python arrancó pero la importación de `pytest` no terminó dentro del límite; no se ha determinado la causa. Por ello no se considera verificada la suite completa del último estado. No se detuvieron procesos ajenos.

[Diagnóstico posterior de pytest] En el checkpoint `85b901a`, Python y la importación de `pytest` volvieron a funcionar: `pytest 9.0.2`, importación medida con `-X importtime` ≈81 ms; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest --version` y una prueba aislada también funcionaron. `pip check` no encontró dependencias rotas. En la medición de recuperación había 51 GiB de memoria disponible y `vmstat` no mostraba espera apreciable de I/O en la muestra; estos datos no demuestran la causa de los bloqueos anteriores. No se modificaron ni terminaron procesos ajenos. Con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen`, la suite completa del checkpoint terminó en 35.41 s: **229 aprobadas, 4 fallos preexistentes y 1 omitida**. JAX no está instalado.

[Implementado y probado con casos sintéticos, no validado por campaña] Se añadió la opción explícita `weight_regularization_mode="absolute"`; `"legacy"` continúa como valor por defecto para preservar los proyectos existentes. En modo absoluto se minimiza el residuo ponderado más penalizaciones absolutas `mu` y `alpha`; el objetivo perfilado del optimizador y el full-refit usan los mismos términos. Se compararon 16 combinaciones de pesos uniformes, por λ, por punto y cero, con/sin cotas y suavizado, frente a SciPy y condiciones KKT; también se probó el caso por λ con suavizado y la propagación a la API/refit. La opción absoluta **no admite NNLS duro con suavizado** y falla explícitamente; la covarianza analítica existente no incorpora las penalizaciones. La ruta `legacy` conserva la incompatibilidad histórica descrita arriba y no debe confundirse con la función objetivo absoluta. No se han validado todos los casos de rango deficiente ni el backend JAX.

[Benchmark preliminar, no campaña] `validation/benchmark_weighted.py` ejecuta el Newton denso en procesos independientes con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; resultados reales en `validation/results/summary/weighted_benchmark.csv`. Para 12 puntos: 3×100, 0.00377 s y 81 820 KiB RSS; 5×300, 0.0632 s y 188 380 KiB; 8×500, 0.719 s y 594 120 KiB. La norma infinita del gradiente fue inferior a 1.9×10⁻¹⁴ en esos tres casos. 10×1000 se omitió por el límite de 4000 coeficientes; su Hessiana densa sola requeriría 800 000 000 bytes. Son mediciones únicas con ruido sintético, no benchmarks estadísticos; el solver no expone iteraciones y no se ejecutó una referencia independiente a esas dimensiones.

[Infraestructura parcial] El paquete `validation/` ya contiene un ejecutor reproducible, checkpoints por réplica con checksum, reanudación y comprobación de commit, huella de código, configuración y versiones; se probaron ejecución serial/paralela, corrupción y cambio de fuente. **Solo admite UV-Vis 1:1 con ruido i.i.d. y error analítico**, por lo que no satisface aún la cobertura de técnicas ni de métodos de la campaña solicitada. Véase `validation/README.md`. No iniciar dos coordinadores sobre el mismo directorio: aún no hay bloqueo entre coordinadores. El archivo de configuración `validation/configs/uvvis_11_pilot.json` define 20 conjuntos exploratorios; no se ha completado todavía una campaña confirmatoria.

[Piloto exploratorio UV-Vis 1:1] Con `validation/configs/uvvis_11_pilot.json`, código `dc88860ed0d5256d14785210785b6b5cafe87753`, 8 puntos y ruido normal σ=0.0002, se ejecutaron 20 conjuntos independientes; 20 ajustes convergieron según el procesador. La cobertura del IC t95 para log K verdadero fue **19/20 = 0.95** (error Monte Carlo 0.0487; Wilson 95 % [0.7639, 0.9911]), sesgo 0.00962, RMSE 0.01415, SD 0.01065 y SE medio 0.01625 (SE/SD 1.525). Tiempo sumado de trabajadores 3.19 s; esto no es tiempo de pared. Los resultados están condicionados al éxito; el piloto no demuestra calibración ni sustenta recomendaciones. Se comprobó la reanudación sin duplicados y la igualdad SHA-256 del CSV antes y después de reanudar. Resumen en `validation/results/summary/uvvis_11_pilot_dc88860.json` y datos completos del pequeño piloto y manifiesto en `validation/results/pilot/uvvis_11_dc88860/`; los checkpoints de ejecución permanecen fuera del repositorio. Para reproducir, usar el código del commit indicado y ejecutar el comando del README.

[Escalabilidad parcial] En el modo absoluto, cuando `alpha_smooth=0`, el problema es separable por longitud de onda y ya se resuelve por columnas en lugar de construir una Hessiana global. Una prueba sintética con 3 especies y 1400 longitudes de onda (4200 coeficientes) recuperó los coeficientes de referencia; el caso con suavizado sigue usando la Hessiana densa y conserva el límite de 4000. No se han implementado ni comparado aún solvers dispersos.

[Referencia de solver disperso con suavizado (sonda, validación)] `validation/sparse_weighted_probe.py` replica el Newton semisuave ponderado con penalización usando estructura dispersa (`scipy.sparse`): la Hessiana es exactamente bloque-diagonal por longitud de onda (término de datos) más `alpha·cron(I_s, L)` con `L = D2ᵀD2` (anchura de banda 2). Contra la ruta densa de producción, en 8 casos (α 0–10⁻⁵, μ 0–5·10⁻², tamaños 36–2400): diferencia máxima de coeficientes ≤9.3·10⁻⁹ (caso de conjunto activo fuertemente activo; típicamente ≤2·10⁻¹¹), brecha de objetivo ≤1.1·10⁻¹⁹ y residual KKT de primer orden ≤4·10⁻¹⁴ en ambas rutas. Tiempos: 600 incógnitas 0.027 s denso vs 0.007 s disperso; 2400 incógnitas 0.805 s vs 0.120 s (6.7×); y el camino disperso resuelve en 0.53 s el caso de 4200 coeficientes que la ruta densa rechaza por su límite. Resultados en `validation/results/sparse_weighted_probe.json`. Decisión: no sustituir aún la implementación densa en producción; la evidencia de equivalencia numérica queda registrada y la adopción (umbral de tamaño, mismo test de optimalidad) requiere su propia prueba de regresión en el núcleo. Para tamaños pequeños el disperso es más lento por overhead de construcción: solo compensa por encima de ~600 incógnitas.

[Auditoría `legacy` vs `absolute`, sonda puntual] `validation/legacy_vs_absolute_probe.py` (resultado commitado en `validation/results/legacy_absolute_probe.json`) ajusta un dataset del piloto con tres ponderaciones (unidad, rampa por longitud de onda, rampa por punto) bajo ambas convenciones mediante la objetivo perfilada de producción. En este régimen (μ=10⁻², sin suavizado) la diferencia de log K entre convenciones es ≤1.5·10⁻⁸ y las objetivo cruzadas coinciden a 10⁻⁸: la divergencia de convenciones (la ruta `legacy` reescala μ respecto a los pesos; `absolute` añade la penalización explícitamente a la objetivo) es numéricamente irrelevante cuando los coeficientes penalmente activos son pocos y pequeños. La auditoría completa (regímenes de penalización fuerte, cotas activas, pesos extremos, suavizado `absolute` con solver disperso) sigue pendiente; una sonda de un dataset no la sustituye.

#### 5. Investigación del sesgo UV-Vis (Etapas A–D, completadas)

Documento completo con tablas, figuras y datos: `docs/UVVIS_BIAS_INVESTIGATION.md`.
Resultados decisivos, todos reproducibles desde los artefactos commitados:

- Reproducción bit a bit del piloto en `dc88860` y en HEAD (20/20 filas idénticas); la diferencia de `source_sha256` se explicó por finales de línea CRLF/LF del checkout original.
- Ruido cero: el procesador recupera log K = 3 con desviación −2.5·10⁻¹⁴; argmin exacto de la objetivo perfilada idéntico a 10 dígitos para OLS, penalización suave y NNLS. Sin defecto generador–ajuste.
- Comparación pareada (mismos datos para los tres proyectores, argmin escalar exacto, referencia independiente de equilibrio cerrado): el estimador sin restricciones es insesgado (−0.0003 ± 0.0044); el penalizado/NNLS tiene sesgo +0.0096 ± 0.0024 y 20/20 réplicas sobre el verdadero. NNLS y penalización μ=10⁻² coinciden a 10⁻⁶; la saturación ocurre desde μ≈10⁻⁴ (CᵀC~10⁻⁵).
- Mecanismo: E[RSS_pen(θ) − RSS_ols(θ)] del recorte de coeficientes negativos crece ~5400× al alejarse θ por debajo del verdadero; el mínimo de E[RSS_pen] queda por encima del valor verdadero y el de E[RSS_ols] en él. La asimetría no persiste sin restricciones.
- Error estándar: el Jacobiano implementado es la derivada proyectada de VarPro y coincide con su identidad a 6.5·10⁻⁶; el SE describe el estimador no restringido (SE/SD = 0.83; tiende a 1.02 con σ→0), mientras el ajuste usa el restringido (SD 0.0107 ⇒ SE/SD = 1.53). La cobertura 19/20 del piloto procede de intervalos ~3× más anchos que la dispersión real del estimador. No se modificó el núcleo: no hay defecto que corregir; hay acoplamiento estimador↔covarianza por documentar y cuantificar con bootstrap.
- Nueve pruebas de regresión en `tests/test_uvvis_bias_reference.py` fijan: recuperación sin ruido (tres proyectores y producción), química cerrada vs NewtonRaphson, argmin exacto = Powell archivado, referencia independiente, identidad del Jacobiano, regímenes de μ y reproducibilidad de estimaciones archivadas del piloto.
- Matriz de 21 escenarios pareados (20–40 réplicas) con configuraciones en `validation/configs/bias/` y resúmenes en `validation/results/bias/`; figuras regenerables con `validation.make_bias_figures`.

#### 6. Comparación de incertidumbre (Etapas E–F, intermedia completada)

Extensión del arnés (`validation/uncertainty_experiment.py`,
`validation/uncertainty_methods.py`): matriz pareada estimador × método sobre
el mismo ajuste; 9 métodos (analítica, HC1, wild lineizado elemental/espectral,
one-step espectral/longitud de onda, full-refit wild elemental/espectral y
remuestreo); modelos de ruido iid, AR(1) ρ=0.6 y heterocedástico; campañas
reanudables con JSONL, manifiesto (sha256 de config, commit, versiones).

Campaña intermedia v2 completada: 280 réplicas y ~108 000 re-ajustes
bootstrap, tasa de fallo 0 %. Hallazgos con datos en
`docs/MONTE_CARLO_RESULTS.md` y `validation/results/unc/`:

- El sesgo positivo del estimador restringido persiste bajo los tres
  regímenes de ruido (+0.0075 a +0.0125, ≥6σ); el OLS sigue insesgado.
- iid/OLS: analítica y wild elemental calibran (SE/SD ≈ 0.92; cov95
  0.92–0.93; cov68 0.58–0.62 algo bajos). HC1 sobre-cubre apenas (0.98).
- El full-refit es el único que reproduce escala y sesgo del estimador
  restringido, pero sus percentiles crudos cubren 0.05–0.10: un IC
  defendible exige corrección explícita de sesgo (pendiente de implementación
  y validación).
- AR(1): los métodos elementales sub-cubren gravemente (0.58–0.73); el
  wild por espectro es la defensa mínima y aun así no calibra con 8 puntos.
- Hetero: el wild elemental conserva la estructura de varianza (0.90–0.96);
  agrupar por espectro la destruye (0.69–0.73).
- Las colas de la distribución de error (m=8) dominan la SD MC del OLS y
  limitan la interpretación de SE/SD.

Campañas confirmatorias i.i.d. completadas (n=500, B=1000; análisis en
`docs/MONTE_CARLO_RESULTS.md` §10): sesgo del restringido +0.00829 ± 0.00038
(22σ) con 487/500 réplicas por encima del valor verdadero; OLS analítico
calibrado (cov95 0.950 ± 0.010, cov68 0.678). HC2/HC3 y perfilado χ² añadidos
al zoo y evaluados (§10.3, §10.5): ninguno repara la inferencia del estimador
restringido; BCa sólo es válido sin frontera. Full-refit confirmatorio
completado (§10.6 MC: el percentil colapsa a 0.07–0.11
y el básico recupera 0.75 con frontera activa; sin restricción la analítica
i.i.d. —0.95— supera al remuestreo completo —0.81—). Mientras no exista
intervalo con sesgo corregido que alcance [0.90,0.97] en el caso restringido, la
cobertura de ese estimador sigue sin calibrar.

#### 7. Trabajo pendiente y dictamen

- ~~Completar y validar el estimador ponderado para cotas suaves y suavizado, sin alterar los resultados por defecto.~~ (Etapas G–H: validación emparejada del camino ponderado y solver disperso con guard documentado; ruta densa inalterada bajo el guard.)
- ~~Crear y verificar el arnés `validation/` para todas las técnicas y métodos solicitados, incluidos checkpoints y manifiesto reproducible.~~ (`uncertainty_experiment.py` + zoo completo; configs congelados y manifiestos con hash.)
- ~~HC2/HC3 implementados y contrastados en la capa de validación (MC §10.3:
  equivalen a una reescala del analítico, HC2≡HC3 por palanca ~10⁻⁶, ciegos a
  AR(1)).~~ Las opciones explícitas ε ≥ 0 (soft-penalty/NNLS frente a OLS) ya
  estaban implementadas y quedan cuantificadas por la matriz pareada; no hay
  defecto que justifique cambiar los predeterminados.
- ~~Ejecutar la campaña con los tamaños, escenarios y criterios de aceptación fijados antes de iniciarla; analizar por separado los casos con error correlacionado no modelado.~~ (confirm-v1: i.i.d. n=500, AR(1) n=150, full-refit n=150; hetero a escala intermedia.)
- ~~Investigar y corregir el hueco de producción NMR 1:2.~~ Diagnostificado y
  resuelto (MC §10.7): el "eco" era (i) un generador inconsistente con la
  convención de núcleos equivalentes del proyector y (ii) el flag
  `param_opt_failed` nunca consultado, que convertía un optimizador inválido
  en eco con `success=True`; corregido en `process_nmr_data` (fallo explícito)
  y fijado con 4 pruebas reales en `tests/test_nmr_12_gap.py` (recuperación
  K1/K2 en régimen identificable, autodiagnóstico SE de K2 plano).
- ~~Elaborar las tablas con errores Monte Carlo, el resumen CSV, las recomendaciones respaldados por datos, los tiempos y el dictamen final.~~ (`results/unc/summary_table.csv`, 201 filas; tablas con IC MC en MC §10.)
- ~~Realizar los commits por etapa; reservar la declaración de fase terminada para cuando se cumplan todos los criterios.~~

**Dictamen: B+ — todos los objetivos obligatorios (A–H) ejecutados con
evidencia publicada; no se certifica A por dos condiciones explícitas:**
(i) el criterio de cobertura [0.90, 0.97] para el estimador **restringido**
no lo alcanza ninguno de los 11 métodos del zoo (mejor 0.75, bootstrap básico
full-refit), límite científico documentado con recomendación práctica
(validación externa/corrección de sesgo; el analítico OLS i.i.d. sí calibra,
0.950 ± 0.010); y (ii) cuatro fallos de tests preexistentes a la Fase 2 y
ajenos a ella (test_acid_base GUI, test_spectro_absgrp_equivalence) siguen
abiertos en la rama. El hueco NMR 1:2 quedó diagnosticado y resuelto
(convención + fallos honestos, §7). Cualquier commit que incluya este informe
sigue siendo un checkpoint; la fusión a `main` y la certificación final
requieren decisión del mantenedor.

#### 8. Etapa H: solver disperso con suavizado y auditoría legacy/absolute

**Solver disperso (`hmfit_core/utils/nnls_utils._solve_weighted_penalty_newton_sparse`).**
Gemelo exacto del Newton semisuave denso (mismo layout de variables, mismo
conjunto activo, misma búsqueda de paso proyectada, mismas tolerancias) con
ensamblado CRS de `scipy.sparse` y SuperLU sobre el conjunto libre. Por
encima del guard denso de 4000 coeficientes la ruta densa fallaba
explícitamente; ahora se despacha automáticamente a la ruta dispersa.
Equivalencia verificada contra la referencia densa en la capa de validación
(`validation/sparse_weighted_probe.py`, datos en
`validation/results/stageH/sparse_scale_cases.json`) y fijada por pruebas
(`tests/test_weighted_solver_sparse.py`):

| coeficientes | diff máx. coef. | gap objetivo | residuo KKT (disperso) | densa | dispersa |
|---:|---:|---:|---:|---:|---:|
| 600 | 1.2·10⁻¹² | 1.1·10⁻¹⁹ | 1.4·10⁻²⁰ | 0.034 s | 0.010 s |
| 2400 | 7.2·10⁻¹¹ | 0 | 2.3·10⁻²⁰ | 0.944 s | 0.142 s |
| 4200 | — (densa no soportada) | — | 1.4·10⁻²⁰ | error explícito antes | 0.61 s |

End-to-end por el procesador (absolute + suavizado + pesos por punto,
4200 coeficientes): 0.07 s, finito y óptimo de primer orden. Casos pequeños
(< guard) conservan la ruta densa bit a bit; no hay cambio numérico para
tamaños previously supportados. La sustitución de la densa por debajo del
guard queda deliberamente descartada hasta reunir más evidencia.

**Auditoría `legacy` frente a `absolute` (`validation/legacy_absolute_audit.py`,
`validation/results/stageH/legacy_absolute_audit.json`).** Convenciones:
`legacy` minimiza el objetivo histórico con penalizaciones `mu`/`alpha`
*relativas* y **ignora las escalas uniformes de pesos** (el argmin de un
escalar positivo por columna cancela); `absolute` minimiza
½Σw²(CA−Y)² + ½μ‖min(A,0)‖² + ½α⟨A,LA⟩ con penalizaciones absolutas.
Identidades verificadas numéricamente (dif ≤ 2.9·10⁻⁹ = tolerancia PGD):
(a) sin pesos, `legacy(μ)` ≡ `absolute(μ)`; (b) con peso uniforme w,
`legacy(μ)` ≡ `absolute(μ·w²)` (y α·w²). Diferencias esperadas con el mismo
μ: peso uniforme w=3 → 2.3·10⁻³ en coeficientes; pesos por λ → 3.9·10⁻⁴;
pesos por punto → 1.4·10⁻⁴. Con pesos no uniformes **no existe** una
reescala única que convierta un modo en el otro: son estimadores distintos.
Asimetría API documentada y conservada: `nnls_hard` + suavizado funciona en
`legacy` (PGD) y lanza `NotImplementedError` en `absolute`.
Implicación para reproducir resultados antiguos: proyectos con pesos
uniformes que pasen a `absolute` deben usar μ_abs = μ·w̄² y α_abs = α·w̄²
para conservar la solución; con pesos variables, cambiar de modo cambia el
estimador y debe revalidarse.
