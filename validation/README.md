# Validation harness — provisional

This is **not** the Phase 2 confirmatory campaign. It currently implements one
production-backed scenario only: UV-Vis 1:1, i.i.d. normal response noise and
analytic uncertainty, using the existing spectroscopy processor and Phase 1
synthetic generator. It uses a two-sided Student-t interval based on the
production VarPro residual degrees of freedom. The fit uses the existing default
soft-penalty absorptivity estimator; analytic uncertainty remains unconstrained,
so its coverage is a research question rather than a guaranteed property.
Other techniques, noise models and bootstrap methods are rejected, not simulated.

For a bounded exploratory pilot (never label its coverage confirmatory):

```sh
QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m validation.run_campaign validation/configs/uvvis_11_pilot.json "$TMPDIR/hmfit-uvvis-pilot"
.venv/bin/python -m validation.analyze "$TMPDIR/hmfit-uvvis-pilot/results.csv"
```

The manifest includes the complete configuration, Git commit, SHA-256 of tracked
Python sources plus `validation/*.py`, dependency versions and backend. Each
replicate has a seed derived from the base seed and replicate index, a
checksummed, atomically replaced checkpoint and a row with convergence, estimate,
SE, interval, runtime and fit evaluation count. `results.csv` is sorted by index.
Resume by re-running the same command; mismatched code, configuration or versions
and corrupted checkpoints are refused. A source change detected while the run is
active stops checkpointing. `workers` uses spawned processes, with BLAS thread
environment set to one before their creation. Run only **one coordinator per
output directory**: cross-coordinator locking is not yet implemented.

The summary reports fit failure fraction and, conditional on successful fits,
bias, RMSE, Monte Carlo SD, mean reported SE, SE/SD, interval widths, coverage,
coverage Monte Carlo standard error and a Wilson interval. It does not yet report
Monte Carlo errors for every metric. Raw data stay outside the repository; do not
commit large run directories. Unsupported confirmatory cells remain incomplete.
