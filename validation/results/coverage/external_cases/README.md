# Synthetic cases for external comparison (BindFit / HypNMR)

Replicate 0 of each pre-registered cell (`validation/coverage_campaign.py`). Concentrations in M; UV-Vis signals are absorbances at the wavelength in the column name (nm); NMR signals in ppm.

| case | model | true log K | HM fit log K (GUI default settings) | file |
|---|---|---|---|---|
| uvvis-11zero-10-iid-rec | 1:1 | 3 | 3.0008 | uvvis-11zero-10-iid-rec.csv / .xlsx |
| uvvis-11zero-10-iid-poor | 1:1 | 3 | 3.0040 | uvvis-11zero-10-iid-poor.csv / .xlsx |
| uvvis-12zero-12-iid-rec | 1:2 | 3, 2.5 | 3.0012, 4.1475 | uvvis-12zero-12-iid-rec.csv / .xlsx |
| nmr-11-10-iid1-rec | 1:1 | 3 | 2.9938 | nmr-11-10-iid1-rec.csv / .xlsx |
| nmr-12-12-iid1-rec | 1:2 | 3, 2.5 | 2.9684, 0.0001 | nmr-12-12-iid1-rec.csv / .xlsx |
