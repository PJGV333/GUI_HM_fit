# Resultados Monte Carlo — comparación de métodos de incertidumbre (Etapa F)

Todos los números de este documento son resultados **efectivamente ejecutados**
en la campaña intermedia de diseño v2 (ver enmienda en
`docs/STATISTICAL_VALIDATION.md` §2.1). Ninguna cifra es extrapolada.

- Configuraciones congeladas: `validation/configs/unc/unc_*.json` (campo
  `"design": "v2"`).
- Datos crudos por réplica: `validation/results/unc/<escenario>/rows.jsonl`.
- Proveniencia por escenario: `summary.json` y `manifest.json` (commit,
  versiones, sha256 de la configuración). Semillas deterministas:
  `SeedSequence([base_seed, r])`; iid base 20261104, AR(1) 20261105,
  hetero 20261106.
- Generador de tablas y figuras: `python -m validation.make_unc_report`
  (tablas en `validation/results/unc/table.md`, figuras en
  `docs/figures/uncertainty/`).

## 1. Diseños y tamaños

| escenario | estimador | ruido | n réplicas | B wild/one-step | B full-refit |
|---|---|---|---:|---:|---:|
| `unc_pen_int60` | penalización suave μ=10⁻² | iid σ=2·10⁻⁴ | 60 | 1000 | 100 |
| `unc_ols_int60` | OLS (sin restricción) | iid | 60 | 1000 | 100 |
| `unc_pen_ar1` | μ=10⁻² | AR(1) ρ=0.6 | 40 | 1000 | 100 |
| `unc_ols_ar1` | OLS | AR(1) | 40 | 1000 | 100 |
| `unc_pen_hetero` | μ=10⁻² | hetero rampa 1→3 | 40 | 1000 | 100 |
| `unc_ols_hetero` | OLS | hetero | 40 | 1000 | 100 |

Todos los métodos se aplican al **mismo ajuste de la misma réplica**
(comparación pareada). El estimador de log K es el argmin escalar exacto de
la objetivo perfilada de producción (equivalencia con Powell bloqueada en
`tests/test_uvvis_bias_reference.py`). Tasa de fallo de los ~108 000 re-ajustes
bootstrap: 0.0 %.

Métodos: `analytic` (covarianza VarPro + t de Student), `hc1` (inflación
global HC1), `lin_wild_el`/`lin_wild_spec` (wild linealizado por elemento /
por espectro), `onestep_spec`/`onestep_wl` (one-step wild por espectro /
por longitud de onda), `refit_wild_el`/`refit_wild_spec`/`refit_resample`
(full-refit con wild elemental / wild por espectro / remuestreo elemental de
residuos). Para los bootstrap se reporta cobertura percentil (`cov95`) y
percentil invertida centrada ("basic", corrige el sesgo del bootstrap).

## 2. Estimador y dispersión Monte Carlo

| escenario | sesgo log K ± IC (MC) | SD MC | RMSE |
|---|---:|---:|---:|
| pen iid | **+0.00752 ± 0.00114** | 0.00881 | 0.01153 |
| OLS iid | −0.00237 ± 0.00228 | 0.01766 | 0.01768 |
| pen AR(1) | **+0.01252 ± 0.00180** | 0.01139 | 0.01683 |
| OLS AR(1) | +0.00330 ± 0.00451 | 0.02855 | 0.02839 |
| pen hetero | **+0.01212 ± 0.00160** | 0.01012 | 0.01571 |
| OLS hetero | −0.00766 ± 0.00587 | 0.03712 | 0.03745 |

El sesgo positivo del estimador restringido es significativo (≥6σ) en los
tres regímenes de ruido y **no** proviene de un defecto del optimizador ni
del generador (Etapa B: argmin exacto, referencia cerrada). El estimador OLS
no muestra sesgo significativo. Con ruido mal especificado (AR(1), hetero)
el sesgo del estimador restringido **aumenta** (≈+0.012).

## 3. Cobertura y calibración por método

Tablas completas: `validation/results/unc/table.md` (regenerable). Resumen
de `cov95` (cobertura percentil 95 %; "basic" entre paréntesis cuando
difiere):

| método | pen iid | OLS iid | pen AR1 | OLS AR1 | pen het | OLS het |
|---|---:|---:|---:|---:|---:|---:|
| analytic | 0.983 | 0.933 | 0.975 | 0.725 | 1.000 | 0.900 |
| hc1 | 1.000 | 0.983 | 1.000 | 0.900 | 1.000 | 0.975 |
| lin_wild_el | 0.967 | 0.917 | 0.950 | 0.725 | 1.000 | 0.900 |
| lin_wild_spec | 0.850 | 0.733 | 0.975 | 0.750 | 1.000 | 0.725 |
| onestep_spec | 0.850 | 0.717 | 0.975 | 0.775 | 1.000 | 0.725 |
| onestep_wl | 0.850 | 0.717 | 0.975 | 0.775 | 1.000 | 0.725 |
| refit_wild_el | 0.083 (0.750) | 0.750 (0.817) | 0.225 (0.400) | 0.600 (0.600) | 0.050 (0.600) | 0.850 (0.750) |
| refit_wild_spec | 0.100 (0.750) | 0.633 (0.583) | 0.350 (0.425) | 0.725 (0.725) | 0.050 (0.600) | 0.650 (0.625) |
| refit_resample | 0.083 (0.767) | 0.817 (0.750) | 0.200 (0.425) | 0.575 (0.550) | 0.050 (0.600) | 0.850 (0.775) |

Razones SE/SD (error típico estimado ÷ SD Monte Carlo; 1 = calibrado):

| método | pen iid | OLS iid | pen AR1 | OLS AR1 | pen het | OLS het |
|---|---:|---:|---:|---:|---:|---:|
| analytic | 1.83 | 0.92 | 1.45 | 0.58 | 3.37 | 0.92 |
| hc1 | 2.33 | 1.16 | 1.84 | 0.73 | 4.27 | 1.16 |
| lin_wild_el | 1.84 | 0.91 | 1.41 | 0.55 | 3.47 | 0.96 |
| lin_wild_spec | 1.68 | 0.72 | 2.22 | 0.80 | 3.10 | 0.69 |
| onestep_spec | 1.68 | 0.72 | 2.21 | 0.80 | 3.11 | 0.69 |
| onestep_wl | 1.67 | 0.72 | 2.21 | 0.80 | 3.11 | 0.69 |
| refit_wild_el | 0.88 | 0.72 | 0.61 | 0.43 | 0.88 | 0.74 |
| refit_wild_spec | 0.82 | 0.57 | 0.79 | 0.63 | 0.84 | 0.54 |
| refit_resample | 0.87 | 0.72 | 0.62 | 0.45 | 0.90 | 0.72 |

Errores MC de las coberturas: ±0.03 (n=60) a ±0.07 (n=40) por método; las
lecturas por debajo de ±0.10 son indicativas.

## 4. Lectura de los resultados

1. **OLS iid**: `analytic`/`lin_wild_el` calibran casi perfectos
   (SE/SD 0.92, cobertura 0.92–0.93); `hc1` sobre-cubre ligeramente
   (0.983). Los métodos de bloque por espectro (`*_spec`) reducen la
   varianza un 28 % y cubren 0.72–0.73: agrupar en 8 bloques destruye la
   estructura i.i.d. — el remedio es peor que la enfermedad si el ruido es
   de verdad elemental.
2. **pen iid**: la covarianza analítica describe el estimador **no**
   restringido y sobreestima la escala (SE/SD 1.8), pero sus intervalos
   anchos cubren 0.98. La cobertura es "accidentalmente buena" — la
   cobertura 68 % ya falla (0.78–0.87 ≫ 0.68). Ningún método linearizado
   captura el sesgo (sesgo boot ≈ 0) porque linealiza alrededor del
   estimador restringido ya desplazado.
3. **Full-refit sobre el estimador restringido**: es el único método cuya
   escala coincide con la dispersión real (SE/SD 0.82–0.90) y cuya
   distribución reproduce el sesgo positivo (sesgo boot +0.0053 vs sesgo MC
   +0.0075). Pero el intervalo percentil queda centrado en θ̂ sesgado y
   cubre 0.05–0.10; invertido ("basic") sube a 0.75–0.77 y aún se queda
   corto. **El intervalo correcto requiere corrección explícita de sesgo +
   ensanchar B; con B=100 y n=60 no alcanza el 95 %.**
4. **AR(1)**: todo lo elemental (analytic, hc1, wild elemental, remuestreo
   elemental) sub-cubre OLS (0.575–0.725; 68 % cae a 0.25–0.575). Los
   wild/one-step por espectro mejoran (0.75–0.775) pero la granularidad de
   bloque (m=8) no basta para calibrar. Con el estimador restringido, los
   métodos anchos cubren por exceso (0.95–1.0) y el full-refit percentil
   vuelve a fallar (0.20–0.35).
5. **Hetero**: el wild elemental reproduce bien la varianza OLS (0.90–0.96)
   — conserva la varianza por observación — mientras el agrupamiento por
   espectro (0.69) y el remuestreo elemental de residuos (refit_resample
   cubre menos y con mal IC) la distorsionan. La corrección global HC1
   "funciona" (0.975) por inflación uniforme, no por robustez.
6. **Colas**: para OLS iid, retirar la única réplica peor reduce la SD MC
   de 0.0203 a 0.0104 (mediana |err| 0.013, máximo 0.054). La SD MC en
   m=8 está dominada por colas de no-linealidad; la comparación SE/SD debe
   leerse junto a las colas, y las coberturas heredadas de ese SD tienen
   incertidumbre adicional. El estimador restringido tiene curtosis 4.1
   (activación de la restricción).

## 5. Contra los criterios de aceptación precriteriados

Criterios (`docs/STATISTICAL_VALIDATION.md` §2): cov95 ∈ [0.90, 0.97],
cov68 ∈ [0.62, 0.74], n ≥ 40 para declarar calibración, fallos < 2 %.

- iid/OLS: `analytic`, `hc1`, `lin_wild_el` cumplen cov95 y cov68
  (salvo hc1 cov95 0.983, marginalmente sobre cubierto).
- iid/pen: ningún método cumple cov68 (todos ≥0.75); full-refit no cumple
  cov95 percentil; "básico" se acerca. **Ningún método calibrado.**
- AR(1)/OLS: ningún método cumple cov95 (máx 0.90 hc1; espec 0.75–0.775).
- AR(1)/pen: métodos anchos violan cov68 (sobre-cubren); full-refit viola
  cov95. Ninguno calibrado.
- hetero/OLS: `analytic`, `hc1`, `lin_wild_el` cumplen rango 95 % y
  `hc1`/`lin_wild_el` el 68 % (0.70–0.70); `*_spec` y refit fallan.
- hetero/pen: todos sobre-cubren (cov95=1.0) o colapsan (full-refit).
- Fallos de refit: 0 % en los ~108 000 re-ajustes. ✓

Ninguna campaña de esta tabla tiene n ≥ 500 (confirmatoria), por lo que
ninguna afirmación de calibración es todavía definitiva: AR(1) y hetero
quedan en n=40 (indicativo) y iid pasa a campaña confirmatoria n=500
(`conf_pen_iid`, `conf_ols_iid`, B=1000) más una confirmatoria dedicada a
full-refit (n=150, B=250).

## 6. Consecuencias para la producción (provisionales)

- Para el estimador restringido por defecto, la SE analítica VarPro debe
  etiquetarse como incertidumbre del proyectado **no restringido**: es
  conservadora en escala (1.4–3.4×) pero no describe el estimador real
  (sesgo no capturado, cov68 fallando).
- Un intervalo defendible para el estimador restringido requiere
  full-refit + corrección de sesgo explícita; los percentiles crudos de
  bootstrap son **incorrectos** para θ̂ sesgado y no deben exponerse como
  IC95 sin más.
- Bajo ruido correlatado intra-espectro, la covarianza elemental
  subestima gravemente; un wild por espectro es la defensa mínima, y con
  pocos puntos (8) su cobertura sigue siendo insuficiente → informar al
  usuario del límite.

Estas consecuencias se confirmarán en la etapa confirmatoria antes de
cambiar ningún valor por defecto.

## 7. Etapa G (inicial): NMR 1:1 por la ruta de producción

Arnés nuevo `validation/nmr_experiment.py` (mismo estilo: JSONL reanudable,
manifiesto, generador idéntico en ecuaciones a `tests/phase1_cases.nmr_case`
— seños por fracción unida del huésped, dos señales, ruido ppm — ajuste vía
`process_nmr_data` de producción con Powell, NewtonRaphson, log K inicial
2.5, cotas [0, 6]). Cobertura evaluada con la SE analítica de producción y
t de Student con dof 21 (2·12 observaciones linealizadas − 1 log K − 2
desplazamientos; convención asumida declarada, no extraída del manifiesto).

| escenario | ruido (ppm) | sesgo ± IC (MC) | SD MC | SE media | SE/SD | cov95 | cov68 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nmr_11_iid` (n=40) | 2·10⁻³ | +0.0020 ± 0.0014 | 0.00880 | 0.00760 | 0.86 | 0.925 | 0.575 |
| `nmr_11_hi` (n=40) | 5·10⁻³ | +0.0080 ± 0.0029 | 0.01808 | 0.01952 | 1.08 | 0.975 | 0.700 |

Lectura inicial (n=40, indicativa): en NMR 1:1 sin restricciones espectrales,
la incertidumbre analítica de producción calibra razonablemente (SE/SD
0.86–1.08, cov95 0.93–0.98). Queda un sesgo pequeño positivo que crece con
el ruido (+0.002 → +0.008, ≈0.4 SD en el régimen ruidoso): coherente con el
sesgo de curvatura de mínimos cuadrados no lineales, no con activación de
restricciones. Datos: `validation/results/nmr/`, configs
`validation/configs/nmr/`. Pendiente para G: 1:2, múltiples señales,
parámetros fijos, datos faltantes, ácido-base, potenciometría y cinética.

## 8. Etapa G (inicial): ácido-base y potenciometría (n=40 c/u)

Arnés `validation/ph_experiment.py` sobre `run_acid_base` + errores
analíticos de producción; generadores espejo de
`tests/phase1_cases.acid_base_nmr_case` y `.potentiometry_case`. Semillas
20261120–22. Cobertura con SE analítica y t-Student con dof declarado por
producción (20 y 58).

| escenario | pKa verdaderos | parámetro | sesgo medio | sesgo mediana | SE/SD | cov95 | cov68 |
|---|---|---|---:|---:|---:|---:|---:|
| NMR shifts 2 pKa | 4.0 / 7.5 | pKa1 | −0.870 ± 0.367 | +0.165 | 1.03 | 0.775 | 0.60 |
| NMR shifts 2 pKa | 4.0 / 7.5 | pKa2 | +0.871 ± 0.367 | −0.164 | 1.03 | 0.775 | 0.625 |
| NMR shifts pKa cercanos | 4.0 / 4.4 | pKa1 | −0.007 ± 0.004 | −0.009 | 1.08 | 0.950 | 0.70 |
| NMR shifts pKa cercanos | 4.0 / 4.4 | pKa2 | +0.003 ± 0.003 | +0.000 | 1.03 | 0.950 | 0.675 |
| Potenciometría 2 pKa | 4.0 / 7.5 | pKa1 | −0.013 ± 0.004 | −0.009 | 1.03 | 0.975 | 0.60 |
| Potenciometría 2 pKa | 4.0 / 7.5 | pKa2 | +0.013 ± 0.004 | +0.008 | 1.03 | 1.000 | 0.60 |

Lectura: la calibración analítica de la ruta ácido-base es correcta en
escala (SE/SD 1.03–1.08 en todos los escenarios). El escenario NMR con
pKa muy separados muestra un **problema de diseño/condicionamiento**, no
de error típico: la SD total (2.3 unidades) está dominada por saltos de
rama en una fracción de réplicas (sesgo de mediana solo ±0.16 frente a
sesgo medio ±0.87; tasa de intercambio de etiquetas 0 — no es confusión
de etiquetas sino escapadas del mínimo). Con pKa próximos el mismo
estimador calibra perfecto (cov95 0.95). La potenciometría calibra bien
en ambos pKa (cov95 0.975/1.0, sesgos ≤0.013). Confirmar con más
réplicas y con múltiples puntos iniciales antes de cualquier
recomendación; datos en `validation/results/ph/`.

## 9. Etapa G (inicial): cinética (n=40 c/u, semillas 20261130–33)

Arnés `validation/kinetics_experiment.py` sobre `hmfit.kinetics`
(`fit_global`, absorbtividades por longitud de onda linealizadas con VarPro,
dof de producción). Esquemas: primer orden A→B; consecutivas A→B→C;
reversible A⇌B; segunda orden A+B→C.

| escenario | parámetro | verdadero | sesgo relativo ± IC (MC) | SD | fallos |
|---|---|---:|---:|---:|---:|
| 1st order | k1 | 1.2 | +0.14 % ± 0.20 % | 0.0148 | 0 |
| consecutive | k1 | 1.2 | +1.22 % ± 0.82 % | 0.0624 | 0 |
| consecutive | k2 | 0.4 | −0.13 % ± 1.4 % | 0.0358 | 0 |
| reversible | k1 | 1.2 | −25 % ± 8.8 % | 0.671 | 0 |
| reversible | k2 | 0.4 | +74 % ± 27 % | 0.677 | 0 |
| 2nd order | k1 | 5000 | −0.025 % ± 0.023 % | 7.35 | 0 |

Hallazgo central (reversible): con absorbtividades libres por longitud de
onda, un relajamiento mono-exponencial identifica con precisión la **tasa de
relajación** k1+k2 = 1.595 ± 0.029 (verdadero 1.600) pero **no** separa
k1 de k2: las réplicas se reparten sobre la curva k1+k2≈cte (p. ej.
[1.566, 0.0004], [1×10⁻⁵, 1.600], [0.997, 0.605]). No es un fallo del
optimizador (0 fallos, el invariante se recupera): es **no-identificabilidad
estructural** del diseño (los amplitudes espectrales libres absorben la
relación de equilibrio). Para A⇌B hay que fijar un k o conocer un espectro
puro. Primer orden, consecutivas y segunda orden: sesgos ≤1.2 % (k1 de
consecutivas, 1.5σ), estimator insesgado a este nivel de ruido. Datos:
`validation/results/kin/`.
