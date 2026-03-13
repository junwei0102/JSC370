# Midterm Project (OpenAlex): Publication Trends, Open Access, and Citations in Machine Learning Research at the University of Toronto

Author: Junwei Quan

## Project overview

This project studies machine-learning research at the University of Toronto from 2015 to 2025 using the OpenAlex API. The report analyzes:

- publication trends over time
- open-access trends
- international collaboration trends
- citation patterns and their correlates

The corpus is defined using a keyword-based search over OpenAlex works, combined with an institution filter for the University of Toronto.

## Repository contents

- `midterm.qmd`: Quarto report source
- `requirements.txt`: Python package requirements

## Data and reproducibility

This project is designed to be reproducible from the OpenAlex API.

The repository does **not** include the generated dataset files such as:

- `data/openalex_ml_works_full.pkl`
- `data/openalex_ml_rowlevel.pkl`

These files are intermediate outputs created by the analysis pipeline after querying OpenAlex. They are not included in the repository for two reasons:

1. the analysis can regenerate them directly from the API, so they are not required as fixed inputs;
2. excluding generated `.pkl` files keeps the repository smaller and avoids committing derived snapshots that may become outdated.

To reproduce the report, you need an OpenAlex API key and must render the Quarto document locally.

## Setup

Create and activate a conda environment, then install the required Python packages:

```bash
conda create -n jsc370-midterm python=3.11
conda activate jsc370-midterm
pip install -r requirements.txt