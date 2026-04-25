# JSC370 Final Project

## Predicting Citation Impact in University of Toronto Machine-Learning Research Using OpenAlex

This repository contains a Quarto website and analysis pipeline for a JSC370 final project on University of Toronto machine-learning publications indexed by OpenAlex. The project studies how publication metadata relate to citation impact and how well those metadata can predict citation outcomes.

Project website: <https://junwei0102.github.io/JSC370/final/>

## Research focus

The dataset includes keyword-defined machine-learning works affiliated with the University of Toronto from 2015 to 2025. The analysis addresses three questions:

1. How did publication output, open-access share, and international collaboration change over time?
2. How well can publication metadata predict citation impact?
3. Which variables are most important in the best-performing prediction model?

The main outcome is `log(1 + citations per year)`. The final report compares Ridge regression, Random Forest, and XGBoost using held-out test-set RMSE, MAE, and R-squared.

## Data source

The project uses the OpenAlex API:

- Base API: <https://api.openalex.org>
- Works endpoint: <https://api.openalex.org/works>
- Institution: University of Toronto
- OpenAlex institution ID: `I185261750`
- Study window: 2015 to 2025
- Work types: `article` and `preprint`
- Search query: `"machine learning" OR "deep learning" OR "neural network" OR "neural networks" OR "artificial intelligence"`

Saved datasets:

- `data/openalex_ml_works_raw.parquet`
- `data/openalex_ml_works_raw.csv`
- `data/openalex_ml_works_clean.parquet`
- `data/openalex_ml_works_clean.csv`

## Repository contents

Key source files:

- `index.qmd`: homepage and project summary
- `report.qmd`: main report
- `viz.qmd`: interactive visualizations
- `scripts/01_fetch_openalex.py`: data collection and feature engineering
- `scripts/02_fit_models.py`: model training, evaluation, and variable importance
- `_quarto.yml`: website configuration

Generated outputs:

- `tables/model_performance.csv`
- `tables/variable_importance.csv`
- `figures/variable_importance.png`
- `models/best_citation_model.joblib`
- `docs/`: rendered website output for GitHub Pages

## Reproducibility

### Requirements

- Python 3
- Quarto

Install Python packages:

```bash
pip install -r requirements.txt
```

Optional OpenAlex credentials can be set as environment variables:

```bash
export OPENALEX_API_KEY="your_key_here"
export OPENALEX_EMAIL="your_email_here"
```

### Rebuild the full project

From the repository root:

```bash
python scripts/01_fetch_openalex.py
python scripts/02_fit_models.py
quarto render
```

If you only want to rebuild the website from the committed data and model outputs, run:

```bash
quarto render
```

## Outputs

The rendered site includes:

- a homepage summarizing the project and key findings
- a full report in HTML and PDF
- an interactive visualization page built with Plotly

The best-performing model is saved to `models/best_citation_model.joblib`, and permutation importance results are saved in both table and figure form.

## Links

- Website: <https://junwei0102.github.io/JSC370/>
- GitHub repository: <https://github.com/junwei0102/JSC370/tree/main/final>