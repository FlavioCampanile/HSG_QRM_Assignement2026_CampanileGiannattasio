# HSG QRM Assignment 2026 - Credit portfolio VaR/ES

Group: Flavio Massimo Campanile, Pietro Tommaso Giannattasio. Deadline: **1 Dec 2026, 9:00**
(email to Ms. Langenegger, subject "7,320 Quantitative Risk Management: Assignment").

## Set-up (once)

```
conda activate climada_env          # the environment that has CLIMADA (see environment.yml)
pip install -e .                    # makes the `qrm` package in src/ importable
```

## Run

```
python scripts/run_all.py           # questions iv-xi and the inputs of the report
```

or step by step:

| Script | Question | Needs | Time |
|---|---|---|---|
| `01_data_and_factor_models.py` | iv | - | seconds |
| `02_simulate_losses.py` | v-vii | - | seconds |
| `03_var_es.py` | viii | 02 | ~1 min |
| `04_rolling_var_es.py` | ix | - | ~15 min (`--plot-only` reuses `results/tables/rolling_var_es.csv`) |
| `05_basel_irb.py` | x | 03 | seconds |
| `06_storm_catalogue.py` | xi, step 1 | **CLIMADA**, internet at the first run | 1-5 min |
| `07_climate_overlay.py` | xi, step 2 | 06 | ~3 min |
| `07b_climate_channels.py` | xi, extensions (PD/LGD channels of Pozdyshev et al. 2025, climate-adjusted IRB, gust stress) | 06, 07 | ~5 min |
| `08_build_report.py` | - | all | seconds |

Script 06 downloads the winter-storm hazard for Switzerland through the CLIMADA data API, evaluates
the Welker impact function at the counterparties and writes `results/climate/storm_catalogue.npz`.
Script 07 reads that file and does not need CLIMADA. `python scripts/07_climate_overlay.py --demo`
runs the overlay on a synthetic catalogue to test the code; its outputs carry the suffix `_DEMO`
and are never used by the report.

## Layout

| Path | Content |
|---|---|
| `data/raw/` | the two Excel files of `qrm26HSG-data.zip` |
| `src/qrm/data.py` | loading, cleaning, weekly / overlapping returns |
| `src/qrm/factor_models.py` | M1, M2, M3: maximum likelihood and simulation |
| `src/qrm/credit_model.py` | s_k, thresholds d_k, conditional PD, losses |
| `src/qrm/risk_measures.py` | empirical VaR and ES (lecture-notes definitions) |
| `src/qrm/rolling.py` | question ix |
| `src/qrm/basel_irb.py` | question x |
| `src/qrm/climada_io.py` | question xi: hazard and impact function from CLIMADA (only module importing CLIMADA) |
| `src/qrm/climate.py` | question xi: storm years, exact law of the worst event, overlay |
| `src/qrm/pipeline.py` | shared set-up, common random numbers, output helpers |
| `scripts/` | one script per block of questions; they write `results/tables` and `results/figures` |
| `report/` | LaTeX sources (class `roffin.cls`); upload this folder to Overleaf |
| `docs/decisions.md` | methodological choices and their justification |

## Report

`report/main.tex` inputs one file per section from `report/sections/`. The text is ordinary LaTeX and
can be edited directly (also on Overleaf). Numbers are written as `\val{q8.M2.VaR_ref|usd}` and are
defined in `report/generated/numbers.tex`, which `scripts/08_build_report.py` regenerates from the
JSON files in `results/tables/`; the same script copies tables and figures to `report/tables` and
`report/figures`. A value that does not exist yet is printed in the PDF as a bold `[key]`, and text
that still has to be written is marked `[TO DO: ...]`: search the PDF for `[` before submitting.

`report/main_preview.pdf` is a compiled copy of the current state.
