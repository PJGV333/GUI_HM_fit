# HM fit — Revisión de fiabilidad y hoja de ruta hacia la versión estable

*Revisión: noviembre 2026 · rama `fix/uncertainty-reliability` · base: commit `5077e2e`*

## 1. Resumen ejecutivo

HM fit tiene una arquitectura sólida (núcleo numérico separado de la GUI, proyección variable,
diagnósticos de identificabilidad, cuatro métodos de error) y la mayoría de los ajustes recuperan
los parámetros verdaderos. El problema principal para una versión estable **no era el ajuste, sino
la confianza en las incertidumbres que se reportan**: varios caminos producían errores estándar o
intervalos que no correspondían a la variabilidad real del estimador, a veces por órdenes de
magnitud, sin ninguna advertencia para el usuario.

Para auditarlo se usó el criterio estándar: **simulación Monte Carlo con parámetros conocidos**,
comparando la desviación estándar empírica del estimador (SD) con el SE reportado (ideal
SE/SD ≈ 1) y la cobertura real del IC 95 % (ideal ≈ 95 %).

## 2. Hallazgos críticos (corregidos en esta rama)

| # | Módulo | Defecto | Evidencia | Corrección |
|---|---|---|---|---|
| 1 | Cinética | Bootstrap *one-step LM* con amortiguamiento `λ·I` no escalado. Con `k` en M⁻¹s⁻¹, `JᵀJ ≈ 1e-7` y λ=1e-3 domina | SD bootstrap 0.0006 vs real ≈ 4.5 (**~7000× menor**) | Amortiguamiento de Marquardt `λ·diag(JᵀJ)` (`lm_damped_step`), invariante a unidades. Aplicado también a Espectroscopía y RMN |
| 2 | Espectroscopía | El solver de absortividades por defecto (`soft_penalty`, μ=1e-2, 300 iteraciones de gradiente) no converge a escala molar (`CᵀC ~ 1e-5`): recorta ε negativas sin reajustar el resto | log K = 3.043 ± 0.004 con verdadero 3.000 (sesgo ≈ 3.5 SD del estimador de mínimos cuadrados) y varianza artificialmente reducida | Solución exacta por Newton semiliso (conjunto activo) para `soft_penalty`; NNLS exacto (Lawson–Hanson) para `nnls_hard`. Sesgo residual 0.006 (efecto físico de ε ≥ 0) |
| 3 | Todos | Grados de libertad sin restar los parámetros lineales (espectros puros, desplazamientos límite, matriz A) que la proyección variable elimina | RMN ácido-base, 10 puntos: SE 11 % bajo y cobertura IC95 **86.5 %** | `dof = n − p_no_lineales − p_lineales` en Espectroscopía, RMN, Ácido-base y Cinética |
| 4 | Todos | IC analíticos con z = 1.96 aunque haya pocos g.l. | t(7) = 2.36 vs 1.96 | Cuantiles t de Student con los g.l. del residuo |
| 5 | Cinética | Fijar cualquier parámetro → `ValueError: Each lower bound must be strictly less than each upper bound` | Reproducido | Los parámetros fijos se excluyen del vector de optimización (SciPy y JAX); varianza 0 |
| 6 | Cinética | Bootstrap *full refit* con `max_nfev = max_iter = 30` y sin verificar convergencia: las réplicas no convergidas quedan cerca de θ̂ y estrechan los IC | Revisión de código | Presupuesto `max_iter·(p+1)·2` evaluaciones; réplicas no convergidas se descartan y se reportan |
| 7 | Espectroscopía/RMN | Con un solo parámetro (caso 1:1, el más común) **todo bootstrap fallaba** (`np.corrcoef` devuelve un escalar) | `IndexError` reproducido en el código original | `np.atleast_2d` en todos los cálculos de correlación |
| 8 | Espectroscopía/RMN | Una máscara de fijos vacía (`[]`) marcaba todos los parámetros como fijos → SE = 0 en bootstrap | SE = 4.6e-16 | Máscara vacía/inconsistente = nada fijo |
| 9 | Ácido-base, Espectroscopía | Matriz de correlación con `np.divide(..., where=)` sin `out=` → memoria sin inicializar | `UserWarning` de NumPy | `out=np.zeros_like(...)` |
| 10 | Cinética | Número de condición sin escalar (depende de las unidades); IC lineales simétricos en k (pueden ser negativos) | — | Condición de J con columnas normalizadas; IC multiplicativos para parámetros ajustados en log10 |
| 11 | Grafos | Los ciclos termodinámicos (receptores heteroditópicos: H+C+A → HCA por dos rutas) eran rechazados aunque fueran consistentes | `test_multicomponent_solver` fallaba | Se acepta el ciclo si es termodinámicamente consistente; si no, error explícito |

### Mejoras metodológicas del bootstrap

* **Remuestreo por punto de titulación / espectro (bloque).** El *wild bootstrap* elemento a
  elemento supone ruido independiente entre longitudes de onda; en UV-Vis real hay deriva de línea
  base y errores de dilución por espectro. Con ese tipo de ruido (simulación cinética):

  | Método | SE/SD | Cobertura IC95 |
  |---|---|---|
  | Analítico | 2.0 | 100 % (intervalos el doble de anchos) |
  | Wild por elemento | 1.9 | 100 % |
  | **Wild por espectro (nuevo, por defecto)** | **0.94** | **91 %** |

  Con ruido i.i.d. el remuestreo por espectro sigue siendo razonable (SE/SD 0.86–0.97). Se añadió
  el selector *Resample by* en la pestaña Errors (todas las técnicas).
* **Corrección global tipo HC1** (corregido en la Fase 1; antes se llamaba erróneamente
  "corrección de leverage"): los residuos se reescalan por el factor único √(n/(n − p_total))
  antes de remuestrear, incluyendo los parámetros lineales. No corrige el leverage individual de
  cada observación (eso serían HC2/HC3).
* **Semilla siempre registrada:** si el usuario no da semilla se genera una y se muestra en el
  resumen, de modo que todo resultado bootstrap es reproducible.
* **Avisos de calidad:** si > 5 % de réplicas fallan o hay < 30 réplicas exitosas, el resumen lo
  advierte (las réplicas que fallan suelen ser las extremas → IC demasiado estrechos).

### Verificación tras las correcciones

| Caso | SE/SD antes | SE/SD después | Cobertura IC95 después |
|---|---|---|---|
| Cinética A→B→C, analítico (150 réplicas) | 0.96 | **1.00** | 95.3 % |
| Ácido-base RMN, 10 puntos (200 réplicas) | 0.89 | **1.01** | ≈ 95 % (con t) |
| Cinética 2.º orden, one-step bootstrap | 0.0001 | **≈ 1** | — |

Espectroscopía: ver sección 4 (Monte Carlo con el solver corregido).

## 3. Estado de las pruebas

* Antes: 143 pasan, **8 fallan** (incluye un archivo `.xlsx` de prueba ignorado por `.gitignore`, por
  lo que la prueba nunca pudo pasar en un clon limpio).
* Después: 158 pasan, **4 fallan**, + 11 pruebas nuevas en `tests/test_uncertainty_reliability.py`
  que fijan cada defecto corregido.
* Fallos pendientes (requieren decisión del autor):
  1. `test_custom_model_dimensions_start_without_forced_roles` — la GUI ahora asigna roles
     `analyte/spectator` por defecto; ¿actualizar la prueba o la GUI?
  2. `test_gui_equation_editor_generates_canonical_model` — un CSV `pH,signal` no se detecta
     automáticamente como espectroscopía y exige columna de volumen (**problema de UX real**).
  3. `test_gui_potentiometry_validation_warns_for_operational_ph` — faltan las advertencias de
     pH operacional > 14 y volumen inicial cero.
  4. `test_absgrp_shared_groups_reduce_optical_components` — el optimizador Powell se detiene en
     puntos distintos según el valor inicial (también sin grupos: RMS 9.4e-6 vs 4.4e-6). Ver §5.
* `tests/test_nmr_regression.py` depende de `test_2_tit_RMN3.xlsx`, que no existe en el repositorio:
  la prueba "pasa" sin ejecutar nada. Conviene incluir el archivo en `tests/data/` o generar los
  datos sintéticamente.
* Recomendación: **CI en GitHub Actions** (Linux, `QT_QPA_PLATFORM=offscreen`) que corra toda la
  suite en cada push; hoy no hay forma de saber que una prueba se rompió.

## 4. Espectroscopía: Monte Carlo con el solver corregido

Titulación 1:1 sintética (log K = 3, 10 puntos × 31 λ, 30 réplicas), con el solver de
absortividades corregido:

| Ruido | Media log K | SD empírica | Analítico SE/SD | Lineal por punto | Lineal por elemento | **Full refit por punto** |
|---|---|---|---|---|---|---|
| i.i.d. (σ = 2e-4) | 3.0046 | 0.0086 | 2.02 | 1.77 | 1.93 | **1.27** |
| + desplazamiento por espectro (σ = 3e-3) | 3.0425 | 0.037 | 5.83 | 1.49 | 5.84 | **0.91** |

Antes de la corrección del solver, con ruido i.i.d. la media era 3.0423 (sesgada) y la SD
empírica 0.0034: el estimador era sesgado y con varianza artificialmente baja.

Lectura:

* En estas simulaciones el **SE analítico resultó 2–6× más ancho** que la dispersión real, porque
  se calcula con absortividades sin restricción mientras el ajuste impone ε ≥ 0. *Nota de la
  Fase 1:* esto se observó en dos diseños concretos y **no demuestra que el SE analítico sea
  siempre conservador**; simplemente no está derivado para el estimador restringido. Desde la
  Fase 1 HM fit informa cuántos coeficientes están en la restricción y advierte. La corrección de
  fondo (Jacobiano condicional al conjunto activo, con su validación cerca de la frontera ε = 0)
  queda como prioridad de la Fase 2.
* El **bootstrap full-refit por punto** es el método mejor calibrado en ambos escenarios y debe
  presentarse como referencia ("auditoría") para resultados publicables.
* Con deriva de línea base no modelada, el sesgo (3.04) domina sobre el ruido: ningún método de
  error lo corrige — hay que modelar la línea base (opción *baseline* existente) o excluir la región.


## 5. Recomendaciones para la versión estable

### 5.1 Fiabilidad numérica (prioridad alta)

1. **Optimizador por defecto en Espectroscopía/RMN:** Powell sobre el RMS escalar es lento y se
   detiene antes del mínimo. Ya existe el Jacobiano de proyección variable: usar Gauss-Newton /
   Levenberg–Marquardt explícito (coherente con la preferencia del proyecto) como opción por
   defecto, con Powell/Nelder–Mead como respaldo. Reportar siempre `success`, número de iteraciones
   y gradiente final.
2. **Coherencia estimador–error:** el error analítico se calcula con absortividades sin restricción
   (pinv), mientras que el ajuste usa ε ≥ 0. Si hay restricciones activas, el SE analítico no
   está derivado para el estimador usado (sin dirección garantizada del error). *Implementado en
   la Fase 1:* aviso con el número de restricciones activas y recomendación del full-refit.
3. **Motor único de incertidumbre.** Hoy hay tres implementaciones paralelas (`model_opt_plots`,
   `kinetics_model_opt_plots`, `acid_base_errors`) con la misma lógica copiada. Mover todo a un
   `hmfit_core/uncertainty.py` con una interfaz única (`residual_fn`, `refit_fn`, `groups`,
   `n_linear`) y un solo widget de errores. Eliminaría la clase de bugs que se encontraron aquí.
4. **Profile likelihood** para parámetros mal identificados (el estándar en equilibrio y cinética
   cuando los IC son asimétricos o no acotados), con gráfico de perfil en la pestaña Errors.
5. **Benchmarks de validación publicados:** un conjunto de 10–15 casos con referencia (BindFit,
   HypNMR/Roelens, Hyperquad para potenciometría, KinTek/COPASI para cinética) ejecutados en CI.
   Es el mejor argumento de fiabilidad frente a revisores y usuarios.
6. Potenciometría: modelos de actividad (Davies) y corrección de CO₂ están en la lista de
   limitaciones; para estudiantes basta documentarlo, pero para investigación al menos Davies.

### 5.2 UI — Cinética y Ácido-base/Potenciometría

Observaciones de la UI renderizada a 1440×900:

* **Panel izquierdo recortado:** botones cortados (`Restore selec…`, tercer botón de Datasets,
  `Process` y siguiente, `Compute` en Errors) y barra de desplazamiento horizontal. Fijar un ancho
  mínimo del panel izquierdo (`QSplitter.setSizes`) y usar `QFlowLayout`/dos filas para las barras
  de botones.
* **Idiomas mezclados:** menú "Ayuda" y log "Listo. Carga datasets…" en español; resto en inglés.
  Centralizar textos (`self.tr()` + archivos `.ts`) y ofrecer ES/EN — muy valioso para estudiantes.
* **Ácido-base → Errors:** el encabezado dice `Estimate log10K` aunque el parámetro sea pKa.
* **Cinética → Optimization:** el combo *Algorithm* (`least_squares` / `trust-region` /
  `multistart`) no cambia nada (todas las opciones usan `trf`) y *multistart* duplica la casilla
  Multi-start. Sustituir por "Trust-region (con límites)" / "Levenberg–Marquardt (sin límites)".
* **Cinética → Model:** el editor de mecanismo vacío no muestra ejemplo. Añadir plantillas
  (A→B, A→B→C, A+B⇌C, Michaelis–Menten) y validación en vivo con resaltado del error.
* **Flujo guiado (modo estudiante):** un indicador de pasos *Datos → Modelo → Ajuste → Errores →
  Exportar* con estado (✓/⚠) y el botón principal siempre visible en la parte inferior.
* **Resultados con semáforo:** junto a cada parámetro, un indicador de confiabilidad combinando
  identificabilidad (condición, correlación máx.), %error y convergencia, con una frase en lenguaje
  llano ("pKa2 y pKa3 están fuertemente correlacionados (r = 0.98): los datos no los distinguen").
* **Tooltips** en todos los campos técnicos (wild, λ, tol, NNLS…). Se añadieron en *Error method* y
  *Resample by* como modelo.
* **Gráfico de residuos por defecto** después del ajuste y aviso automático si los residuos muestran
  estructura (prueba de rachas / Durbin–Watson): es la señal más útil de modelo incorrecto.

### 5.3 Ingeniería y publicación

* Retirar archivos obsoletos (`nmr_processor_prev.py`, `hmfit_crash.log`, JSON de configuración en
  la raíz, `__pycache__` versionado).
* Fijar versiones en `requirements_qt.txt` (p. ej. `PySide6>=6.8,<6.11`, `numpy<3`) para builds
  reproducibles.
* Versión semántica (`1.0.0` para la estable) y `CHANGELOG.md`; los commits `HMfit_pys_v1.xx` no
  permiten saber qué cambió.
* Publicar en Zenodo (DOI) y un artículo de software (JOSS o *J. Chem. Educ.* por el enfoque
  docente); exige pruebas, documentación y ejemplos, que es justo lo que pide la versión estable.

## 6. Cambios de comportamiento a tener en cuenta

* Los SE analíticos aumentan ligeramente (corrección de g.l.) y los IC usan t: en titulaciones
  pequeñas los intervalos son más anchos — y correctos.
* Los valores de log K en Espectroscopía con `soft_penalty`/`nnls_hard` pueden cambiar en la
  tercera cifra decimal: el solver ahora converge.
* El bootstrap remuestrea por punto de titulación por defecto; el modo anterior sigue disponible
  (*Resample by → Data element*).
