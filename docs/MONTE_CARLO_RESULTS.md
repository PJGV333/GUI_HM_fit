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

> **Corrección (Fase 2, 2026-11-05).** En la convención de HM fit (pKa₁ = log β₁, sitio más
> básico) los escenarios "4.0 / 7.5" y "4.0 / 4.4" tienen pKa₁ < pKa₂: describen sistemas
> **cooperativos**, no ácidos dipróticos ordinarios, y los datos determinan sobre todo la suma
> pKa₁ + pKa₂ (obsérvense los sesgos medios simétricos −0.870 / +0.871). La lectura
> "problema de diseño/condicionamiento… escapadas del mínimo" queda superada. Véase
> `docs/STABILITY_PHASE2_REPORT.md` §9.3; los diseños con pKa₁ > pKa₂ están en la campaña
> pre-registrada de la Fase 2.

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

## 10. Campañas confirmatorias (n=500, B=1000) y zoo de intervalos extendido

Diseño **congelado antes del análisis** (`validation/configs/unc/conf_*.json`,
semillas fijas por réplica `SeedSequence([base_seed, r])`, manifiestos con
commit y hashes): `conf_pen_iid` (soft-penalty μ=10⁻²) y `conf_ols_iid`,
n=500, σ=2·10⁻⁴, m=8, log K=3, g_max=2·10⁻³; bootstrap linealizado wild por
elemento y por espectro B=1000; one-step (espectro y wild) B=1000; full-refit
(el, espectro, remuestreo) B=1000 — este último en curso al redactar (§10.5).
Datos: `validation/results/unc/conf_*`.

### 10.1 Estimador y signo (el test del signo que motivó la investigación)

| estimador | n | sesgo ± EM(MC) | SD | RMSE | frac(θ̂ > 3) |
|---|---:|---:|---:|---:|---:|
| soft-penalty μ=10⁻² | 500 | **+0.00829 ± 0.00038** (22σ) | 0.00860 | 0.01194 | **487/500 = 0.974 ± 0.007** |
| OLS sin restricción | 500 | −0.00072 ± 0.00072 (1σ) | 0.01617 | 0.01617 | 240/500 = 0.480 ± 0.022 |

Con n=500 el sesgo del estimador restringido es inequívoco (22σ) y
unidireccional (97.4 % por encima); el no restringido es compatible con cero
sesgo y simétrico. El fenómeno pertenece al estimador, no al generador.

### 10.2 Cobertura confirmatoria i.i.d. (B=1000 por método)

| método | pen: SE/SD | pen: cov95 | pen: cov68 | ols: SE/SD | ols: cov95 | ols: cov68 |
|---|---:|---:|---:|---:|---:|---:|
| analítico VarPro | 1.900 | 0.994 ± 0.003 | 0.792 | 1.014 | **0.950 ± 0.010** | 0.678 |
| HC1 | 2.411 | 1.000 | 0.862 | 1.287 | 0.988 ± 0.005 | 0.794 |
| wild el. (linealizado) | 1.926 | 0.984 ± 0.006 | 0.802 | 1.021 | 0.942 ± 0.010 | 0.680 |
| wild espectro (linealizado) | 1.706 | 0.880 ± 0.015 | 0.736 | 0.807 | 0.826 ± 0.017 | 0.588 |
| one-step espectro | 1.706 | 0.882 ± 0.014 | 0.736 | 0.807 | 0.818 ± 0.017 | 0.576 |
| one-step wild | 1.705 | 0.876 ± 0.015 | 0.734 | 0.806 | 0.828 ± 0.017 | 0.594 |

Lectura: para **OLS i.i.d.** el SE analítico VarPro está calibrado a n=500
(95 %: 0.950 ± 0.010; 68 %: 0.678) y el wild por elemento lo reproduce
(0.942). HC1 infla 1.29 → sobre-cubre. Los métodos en bloque destruyen la
calibración i.i.d. (0.83). Para el **restringido** el analítico/wild-el van
1.9× inflados: cov95 0.994 es un artefacto de amplitud (ancho 0.064 frente a
SD 0.0086) y cov68 0.79 delata la mala calibración. Ninguna cobertura del 95 %
aquí debe leerarse como validez sin la amplitud y el 68 % junto.

### 10.3 HC2/HC3 en el contexto VarPro (`validation/uncertainty_methods.py`)

Implementación: Ω=diag(r²/(1−h)ᵖ) con h la palanca de la proyección lineal
(fila i: ‖C_i‖²/Σ‖C‖²; aquí h≈2·10⁻⁶ ⇒ **HC2≡HC3 numéricamente**), factor
finito MacKinlay–White n/dof sobre Ω y sandwich con el Jacobiano proyectado.
Campañas n=60 (OLS, B_refit=0, base_seed 20261125–27; `validation/results/unc/hc_*`):

| ruido | analítico ratio/cov95 | HC1 | HC2≡HC3 |
|---|---:|---:|---:|
| iid | 0.902 / 0.917 | 1.145 / 0.983 | 0.896 / 0.883 |
| AR(1) ρ=0.6 | 0.636 / 0.783 | 0.807 / 0.883 | 0.640 / 0.800 |
| hetero | 0.901 / 0.933 | 1.144 / 0.967 | 0.908 / 0.917 |

Conclusión: la familia HC2/HC3 de residuos **no aporta nada** en este problema
— la palanca de la proyección es despreciable, el sandwich diagonal equivale a
un ajuste de escala del analítico y sigue ciego a correlación (AR1 ratio
0.64). Las correcciones HC2/HC3 *de residuos* (MacKinlay–White) no son los
estimadores de covarianza robustos HC2/HC3 de la regresión lineal con palanca
relevante; transplantar la fórmula lineal aquí es un callejón. Para
correlación el instrumento válido es el remuestreo en bloque (§3).

### 10.4 Bootstrap full-refit: percentíl, básico y BCa (n=40–60, B_refit=250, base_seed 20261114–17)

`validation/results/unc/bca_*`; BCa con z₀ percentíl y aceleración jackknife
sobre los m=8 espectros.

| escenario | método base | cov95 perc | cov95 básico | cov95 BCa | ancho perc | ancho BCa |
|---|---|---:|---:|---:|---:|---:|
| pen i.i.d. | wild el. | 0.10 | **0.70** | 0.23 | 0.026 | 0.017 |
| pen AR1 | wild espec. | 0.30 | **0.62** | 0.28 | 0.027 | 0.015 |
| pen hetero | wild espec. | 0.17 | **0.53** | 0.23 | 0.029 | 0.018 |
| ols i.i.d. | wild el. | 0.88 | 0.93 | **0.93** | 0.049 | 0.049 |

BCa sólo iguala al intervalos básico cuando no hay frontera (OLS); con
restricción activa empeora todo (aceleración jackknife inestable sobre 8
puntos + distribución bootstrap asimétrica sesgada arriba). Ningún intervalo
estándar de refit alcanza 95 % con el estimador restringido en m=8: el mejor
es el básico (≈0.55–0.62), lejos del nominal.

### 10.5 Intervalos por perfilado sobre la MISMA objetivo restringida

Método `profile`: {θ : rms_prof(θ)² − rms_prof(θ̂)² ≤ s²·t²_{0.975}}, umbral
χ²₁/σ̂² de rigor, resolución por expansión+bisección a ambos lados (asimétrico
por construcción). Campañas n=120 (iid) y n=60 (AR1), `results/unc/prof_*`:

| escenario | analítico cov95 | perfil cov95 ± EM | perfil cov68 | ancho perfil |
|---|---:|---:|---:|---:|
| pen iid | 0.975 ± 0.014 | **0.725 ± 0.041** | 0.333 | 0.029 |
| pen AR1 | 0.967 ± 0.023 | 0.633 ± 0.062 | 0.383 | 0.027 |

El perfilado ΔRSS calibrado ingenuamente **infracubre gravemente** para el
estimador restringido: cerca de la frontera de actividad de la penalización el
perfil se aplana y el umbral χ²₁ queda anticonservador respecto a la
distribución real del estimador (cov68 0.33 ≪ 0.68). Confirma la advertencia
de la Etapa D: la covarianza/umbral del problema *sin* restricción no es
válida para el estimador *restringido*, ni siquiera en forma perfilada.

### 10.6 Confirmatoria full-refit (n=150, B_refit=250, semillas 20261111)

Configs congelados `conf_pen_refit.json` / `conf_ols_refit.json`; métodos de
re-ajuste completo (250 re-ajustes completos por réplica) más wild linealizado
B=1000 en el mismo escenario.

| método full-refit | restringido percentil / básico | OLS percentil / básico |
|---|---|---|
| wild elemental | 0.11 / 0.75 | 0.81 / 0.81 |
| wild por espectro | 0.07 / 0.73 | 0.63 / 0.63 |
| remuestreo observaciones | 0.08 / 0.75 | 0.81 / 0.82 |

- Sesgo en estas semillas: **+0.00817 ± 0.00072** (11σ), SD 0.00884. El
  interval **percentil**, centrado en θ̂ ya sesgado, duplica el sesgo y colapsa
  a 0.07–0.11; el **básico** (que lo resta) recupera 0.73–0.75, el mejor
  full-refit con la restricción activa, aún lejos del 95 % nominal. Confirma a
  escala confirmatoria la jerarquía observada con B=40–250 (§10.4).
- OLS (sesgo −0.002 ± 0.0015): el full-refit (0.81–0.82) cubre **menos** que la
  analítica VarPro i.i.d. (0.95, §10.1): con m=8 el re-ajuste añade ruido de
  remuestreo sin aportar información; cuando el modelo de ruido es correcto la
  analítica domina. Por espectros (bloques que rompen el i.i.d.) cae a 0.63,
  coherente con §4.1.
- **Conclusión Etapa F:** ningún interval muestread por re-ajuste completo
  alcanza cobertura nominal con la frontera activa (mejor 0.75, básico); sin
  restricción la analítica i.i.d. sigue siendo la mejor calibrada. La
  recomendación práctica de corrección de sesgo por validación externa sigue en
  pie.

### 10.6-bis Confirmatoria AR(1) (n=150, B=500, semillas 20261220–21)

Diseño de la misma familia confirm-v1 con `noise_model=ar1, rho=0.6`
(`validation/configs/unc/conf_*_ar1.json`, datos en `results/unc/conf_*_ar1`):

| método | pen: ratio | pen: cov95 | pen: cov68 | ols: ratio | ols: cov95 | ols: cov68 |
|---|---:|---:|---:|---:|---:|---:|
| analítico VarPro | 1.509 | 0.973 ± 0.013 | 0.680 | **0.644** | **0.793 ± 0.033** | **0.493** |
| HC1 | 1.915 | 1.000 | 0.713 | 0.818 | 0.907 ± 0.024 | 0.553 |
| wild el. linealizado | 1.519 | 0.960 ± 0.016 | 0.673 | 0.633 | 0.760 ± 0.035 | 0.487 |
| wild espectro | 2.350 | 0.947 ± 0.018 | 0.840 | 0.882 | 0.833 ± 0.030 | 0.673 |
| one-step espectro | 2.337 | 0.933 ± 0.020 | 0.827 | 0.882 | 0.827 ± 0.031 | 0.653 |
| one-step wild | 2.345 | 0.947 ± 0.018 | 0.833 | 0.884 | 0.827 ± 0.031 | 0.647 |

A n=150 queda confirmado con EM pequeña: bajo AR(1) no modelado el error
analítico y el wild elemental colapsan (cobertura 95 real 0.76–0.79, 68 real
0.49 — es decir, el SE subestima ~1.6×); el remuestreo en bloque por espectro
recupera la escala (ratio 0.88) pero aun no calibra (0.83) con m=8. El sesgo
del restringido persiste (+0.01056 ± 0.00089, 12σ). Las defensas elementales
de la Etapa D/F son inútiles para correlación intra-espectro; se necesita o
modelar AR(1) explícitamente o más puntos.

Nota BCa (n=60, i.i.d., pen): percentíl 0.10 / básico 0.70 / BCa 0.23 —
refuerza §10.4: BCa perjudica bajo restricción activa.

### 10.7 NMR 1:2: el "hueco de producción" se resolvió en convención + identidad (diagnóstico, no parche cosmético)

La afirmación inicial (§10.7 anterior: "el ajuste no optimiza log K, eco de
iniciales") se investigó a fondo y **quedó sustituida** por tres hechos
medidos sobre HEAD:

1. **Convención del generador.** El proyector de corrimientos
   (`project_coeffs_block_onp_frac`) pondera por núcleos equivalentes: en la
   señal del núcleo G, HG2 entra con peso 2 y el denominador es la concentración
   total de G. El generador sintético del repro original dividía C_HG2 por
   [G]_tot con peso 1 y excluyó G de `non_absorbent_species` (`nas=[1]`), lo
   que además expulsaba a G libre del diseño y cambiaba el denominador a
   "G ligado". Con datos **consistentes con la convención** (dos núcleos,
   `nas=[]`, `stoichiometry_map` de 4 filas), `process_nmr_data` recupera
   ambas constantes: K1 3.043 ± 0.017 y K2 4.484 ± 0.013 (verdad 3.0/4.5,
   σ=2·10⁻³, 24 puntos, semilla 11).
2. **Identificabilidad de K2.** En regímenes donde 2·K2·[H]·[G] ≪ 1 (HG2
   <1 % del G total) la dirección de K2 es plana: el ajuste devuelve K2 al
   límite de la caja con SE analítica de miles (3.9·10³), es decir, la propia
   SE autodiagnostica el régimen no identificable. No es un defecto del
   optimizador sino del diseño experimental.
3. **Defecto real acotado y corregido: eco silencioso con optimizador
   inválido.** `optimizer="least_squares"` no existe en
   `scipy.optimize.minimize`; la bandera `param_opt_failed` del bucle IRLS se
   establecía pero **nunca se consultaba**, y el pipeline devolvía las
   constantes iniciales con `success=True` (SE incluida). Corregido en
   `nmr_processor.process_nmr_data`: si todas las ejecuciones fallan, se
   devuelve `success=False` con mensaje explícito en vez del eco. Fijado por
   `tests/test_nmr_12_gap.py` (ahora 4 pruebas reales, sin xfail: recuperación
   1:2, autodiagnóstico de K2 plano, error explícito por longitud de K, y
   fallo honesto de optimizador inválido).

Consecuencia: la MC de NMR publicada (§5, 1:1 con `powell`) sigue válida; su
generador coincide con el modelo de ajuste (peso huésped/H). Queda documentado como convención: para datos de dos
núcleos no excluir especies padre de señal en `nas` y ponderar el
`stoichiometry_map` por núcleos por especie. La limitación "MC de NMR solo
1:1" se levanta: el camino 1:2 de `Free` está verificado; la extensión MC
1:2 queda como trabajo futuro opcional, no como hueco.
