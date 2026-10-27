# Heterogeneous geographic generalisability of machine learning models for meropenem resistance prediction in Klebsiella pneumoniae

Authors: Samson Samuel Ikhayere (ORCID 0009-0003-9533-2314), Ejiakhila Amelia Omono (ORCID 0009-0001-9407-1226). University of Benin, Benin City, Nigeria.

This repository holds the analysis code behind the manuscript. Results, figures, logs and processed tables are archived on Zenodo (DOI: [ZENODO DOI TO FILL]). All reported values come from a run on Windows 10 with Python 3.11.17, pandas 2.3.3, numpy 2.4.6, scikit-learn 1.9.1 and XGBoost 3.2.0 (see `results/software_versions.json` and `env/requirements.txt`).

## Contents (in the Zenodo archive; the repository holds `scripts/` and `env/`)
- `scripts/`: analysis code. `00b` to `06` download the data from BV-BRC and build the cohort and feature matrix. `07` to `22` run the analysis and make the figures and tables. `run_all.py` runs `00_setup` and `04` to `22` in order from the processed tables (the download scripts are not re-run).
- `results/`: result tables (CSV and JSON). `table1` to `table7` feed the main-text tables and `tableS1` to `tableS12b` feed the supplementary tables. Table S12 and S12b come from script 22.
- `figures/`: main and supplementary figures (PNG and PDF).
- `logs/`: log of each script. `22_temporal_sensitivity.log` is the terminal output copied from the run.
- `env/requirements.txt`: package versions.
- `manuscript/`: the manuscript and supplementary file as submitted.

## Data
Raw data were downloaded from the BV-BRC public API (https://www.bv-brc.org) on 5 October 2026 (two tables: Klebsiella pneumoniae genomes, and meropenem susceptibility records). The exact queries and SHA256 checksums are in `provenance/`. The processed tables (`processed/`) are the starting point for `run_all.py`.

## How to run
`pip install -r env/requirements.txt`, then `python scripts/run_all.py` from this folder.

## Notes
- Regions are continental. Central America and the Caribbean are placed in North America. "Not recorded" means no country in BV-BRC.
- Scripts 21 (post hoc analyses) and 22 (temporal cut-off sensitivity) were added after the main analysis. Script 22 uses every collection year in the data as a cut-off; the paper's main analysis used 2016.
- Random seed 42 is used for model fits, random splits, bootstrap resamples and learning-curve draws. The MLST-grouped split has no random element.
