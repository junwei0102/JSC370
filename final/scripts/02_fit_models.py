from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


DATA_PATH = Path("data/openalex_ml_works_clean.parquet")
TABLE_DIR = Path("tables")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

TABLE_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def group_rare_categories(series: pd.Series, top_n: int = 25) -> pd.Series:
    """Keep top categories and group all others into 'Other'."""
    s = series.fillna("Unknown").astype(str)
    top_values = s.value_counts().head(top_n).index
    return np.where(s.isin(top_values), s, "Other")


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Important: do not use cited_by_count, log_cites, or citations_per_year as predictors.
    # They are directly related to the outcome and would create data leakage.

    df["venue_top"] = group_rare_categories(df["venue"], top_n=30)
    df["topic_top"] = group_rare_categories(df["primary_topic_name"], top_n=30)

    # Title keyword indicators.
    title_lower = df["title"].fillna("").str.lower()
    df["title_has_deep_learning"] = title_lower.str.contains("deep learning", regex=False).astype(int)
    df["title_has_neural"] = title_lower.str.contains("neural", regex=False).astype(int)
    df["title_has_ai"] = title_lower.str.contains("artificial intelligence|\\bai\\b", regex=True).astype(int)
    df["title_has_model"] = title_lower.str.contains("model", regex=False).astype(int)

    outcome = "log_cites_per_year"

    numeric_features = [
        "publication_year",
        "authors_count",
        "countries_distinct_count",
        "institutions_distinct_count",
        "log_authors",
        "log_countries",
        "log_institutions",
        "title_length",
        "title_has_deep_learning",
        "title_has_neural",
        "title_has_ai",
        "title_has_model",
    ]

    categorical_features = [
        "type",
        "is_oa",
        "oa_status",
        "international",
        "has_doi",
        "venue_top",
        "topic_top",
    ]

    keep_cols = numeric_features + categorical_features + [outcome]
    model_df = df[keep_cols].copy()

    # Drop rows where outcome is missing.
    model_df = model_df.dropna(subset=[outcome]).copy()

    X = model_df[numeric_features + categorical_features]
    y = model_df[outcome]

    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def evaluate_model(model_name: str, fitted_model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = fitted_model.predict(X_test)

    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_test, pred)
    else:
        rmse = np.sqrt(mean_squared_error(y_test, pred))

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def fit_models(X_train, y_train):
    preprocessor = make_preprocessor(X_train)

    models = {}

    ridge_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(random_state=370)),
        ]
    )

    ridge_grid = {
        "model__alpha": [0.1, 1.0, 10.0, 50.0],
    }

    models["Ridge regression"] = GridSearchCV(
        ridge_pipe,
        ridge_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    rf_pipe = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X_train)),
            (
                "model",
                RandomForestRegressor(
                    random_state=370,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf_grid = {
        "model__n_estimators": [300],
        "model__max_depth": [8, 14, None],
        "model__min_samples_leaf": [1, 5],
    }

    models["Random Forest"] = GridSearchCV(
        rf_pipe,
        rf_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    if HAS_XGBOOST:
        xgb_pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X_train)),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=370,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )

        xgb_grid = {
            "model__n_estimators": [300, 600],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.08],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
        }

        models["XGBoost"] = GridSearchCV(
            xgb_pipe,
            xgb_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

    fitted = {}

    for model_name, model in models.items():
        print(f"\nFitting {model_name}...")
        model.fit(X_train, y_train)
        fitted[model_name] = model.best_estimator_
        print(f"Best parameters for {model_name}:")
        print(model.best_params_)

    return fitted


def save_permutation_importance(best_model_name, best_model, X_test, y_test):
    print(f"\nComputing permutation importance for {best_model_name}...")

    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=370,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_sd": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    importance.to_csv(TABLE_DIR / "variable_importance.csv", index=False)

    # Save top 15 for report figure.
    top = importance.head(15).sort_values("importance_mean")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title(f"Variable importance from {best_model_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "variable_importance.png", dpi=300)
    plt.close()

    return importance


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cannot find data/openalex_ml_works_clean.parquet. "
            "Run scripts/01_fetch_openalex.py first."
        )

    df = pd.read_parquet(DATA_PATH)

    X, y = prepare_model_data(df)

    # Stratify by publication year so old and recent papers are represented in both sets.
    stratify_col = X["publication_year"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=370,
        stratify=stratify_col,
    )

    fitted_models = fit_models(X_train, y_train)

    performance_rows = []

    for model_name, model in fitted_models.items():
        metrics = evaluate_model(model_name, model, X_test, y_test)
        performance_rows.append(metrics)

    performance = (
        pd.DataFrame(performance_rows)
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )

    performance.to_csv(TABLE_DIR / "model_performance.csv", index=False)

    best_model_name = performance.loc[0, "Model"]
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, MODEL_DIR / "best_citation_model.joblib")

    importance = save_permutation_importance(best_model_name, best_model, X_test, y_test)

    print("\nModel performance:")
    print(performance.round(3).to_string(index=False))

    print("\nTop variable importance:")
    print(importance.head(10).round(4).to_string(index=False))

    print("\nSaved:")
    print("  tables/model_performance.csv")
    print("  tables/variable_importance.csv")
    print("  figures/variable_importance.png")
    print("  models/best_citation_model.joblib")


if __name__ == "__main__":
    main()