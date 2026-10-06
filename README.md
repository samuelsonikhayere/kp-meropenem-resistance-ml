# Geographic generalisability of meropenem resistance prediction in Klebsiella pneumoniae

Analysis pipeline accompanying the manuscript:

**Heterogeneous Geographic Generalisability of Machine Learning Models for Meropenem Resistance Prediction in Klebsiella pneumoniae**

## Overview

Machine learning models predicting antimicrobial resistance from genomic features are
increasingly reported, yet their geographic generalisability remains poorly characterised.
This repository contains the complete, reproducible pipeline used to evaluate meropenem
resistance prediction on 3,930 verified K. pneumoniae phenotypes retrieved from the
Bacterial and Viral Bioinformatics Resource Center (BV-BRC) via public API.

## Pipeline

All scripts live in scripts/ and are numbered in execution order. See the full
pipeline table in the accompanying manuscript (Methods section).

## Requirements

- Python 3.11
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, requests, tqdm, openpyxl, tabulate

## Reproducing the analysis

    conda create -n amr python=3.11 -y
    conda activate amr
    pip install -r env/requirements.txt

    python scripts/00_setup.py
    python scripts/00b_download_raw.py
    # ... run scripts in numbered order ...

Raw data are re-downloaded from the public BV-BRC API; no data are stored in this
repository. Processed datasets and figures are archived at [Zenodo DOI].

## Data and code availability

- Code: this repository
- Data: [Zenodo DOI]
- Raw source: https://www.bv-brc.org

## License

MIT License (see LICENSE file).

## Contact

Samson Samuel Ikhayere - samuelsonikhayere@gmail.com
University of Benin, Nigeria
