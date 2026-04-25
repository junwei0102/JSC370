import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


OA_BASE = "https://api.openalex.org"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PARAMS = {
    "institution_id": "I185261750",
    "start_year": 2015,
    "end_year": 2025,
    "type_filter": "article|preprint",
    "ml_query": '"machine learning" OR "deep learning" OR "neural network" OR "neural networks" OR "artificial intelligence"',
    "as_of_year": 2026,
}


def oa_get_json(path: str, query: dict | None = None, max_tries: int = 6, timeout_s: int = 30) -> dict:
    query = dict(query or {})

    # Optional: use API key if you have one set in your terminal.
    api_key = os.getenv("OPENALEX_API_KEY", "")
    if api_key:
        query["api_key"] = api_key

    # Optional but recommended by OpenAlex.
    email = os.getenv("OPENALEX_EMAIL", "")
    if email:
        query["mailto"] = email

    url = f"{OA_BASE}{path}"
    headers = {"User-Agent": "JSC370-final-project"}

    for i in range(1, max_tries + 1):
        try:
            response = requests.get(url, params=query, headers=headers, timeout=timeout_s)
        except requests.RequestException:
            time.sleep(min(2 ** (i - 1), 32))
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** (i - 1), 32))
            continue

        raise RuntimeError(f"OpenAlex API error HTTP {response.status_code}: {response.text[:500]}")

    raise RuntimeError("OpenAlex API request failed after retries.")


def fetch_works_for_year(year: int) -> list[dict]:
    out = []
    page = 1

    while True:
        dat = oa_get_json(
            path="/works",
            query={
                "search": PARAMS["ml_query"],
                "filter": ",".join(
                    [
                        f"authorships.institutions.id:{PARAMS['institution_id']}",
                        f"publication_year:{year}",
                        f"type:{PARAMS['type_filter']}",
                    ]
                ),
                "per_page": 100,
                "page": page,
            },
        )

        results = dat.get("results", []) or []
        if not results:
            break

        out.extend(results)

        if len(results) < 100:
            break

        page += 1
        time.sleep(0.2)

    return out


def flatten_work(work: dict) -> dict:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    primary_topic = work.get("primary_topic") or {}
    open_access = work.get("open_access") or {}
    authorships = work.get("authorships") or []
    ids = work.get("ids") or {}

    return {
        "id": work.get("id"),
        "doi": ids.get("doi"),
        "title": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "is_oa": open_access.get("is_oa"),
        "oa_status": open_access.get("oa_status"),
        "authors_count": len(authorships),
        "countries_distinct_count": work.get("countries_distinct_count"),
        "institutions_distinct_count": work.get("institutions_distinct_count"),
        "venue": source.get("display_name"),
        "primary_topic_name": primary_topic.get("display_name"),
    }


def clean_and_engineer_features(works: pd.DataFrame) -> pd.DataFrame:
    df = works.copy()

    df = df.drop_duplicates(subset="id")
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df.dropna(subset=["id", "publication_year"]).copy()
    df["publication_year"] = df["publication_year"].astype(int)

    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")

    numeric_cols = [
        "cited_by_count",
        "authors_count",
        "countries_distinct_count",
        "institutions_distinct_count",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cited_by_count"] = df["cited_by_count"].fillna(0).clip(lower=0)

    for col in ["type", "oa_status", "venue", "primary_topic_name"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    df["is_oa"] = df["is_oa"].map({True: "OA", False: "Closed"}).fillna("Unknown")

    df["international"] = np.where(df["countries_distinct_count"] > 1, "Yes", "No")
    df["has_doi"] = np.where(df["doi"].notna(), "Yes", "No")

    df["citation_age_years"] = np.maximum(
        1,
        PARAMS["as_of_year"] - df["publication_year"] + 1,
    )
    df["citations_per_year"] = df["cited_by_count"] / df["citation_age_years"]

    df["log_cites"] = np.log1p(df["cited_by_count"])
    df["log_cites_per_year"] = np.log1p(df["citations_per_year"])

    df["log_authors"] = np.log1p(df["authors_count"].fillna(0).clip(lower=0))
    df["log_countries"] = np.log1p(df["countries_distinct_count"].fillna(0).clip(lower=0))
    df["log_institutions"] = np.log1p(df["institutions_distinct_count"].fillna(0).clip(lower=0))

    df["title_length"] = df["title"].fillna("").astype(str).str.len()

    return df


def main():
    all_works = []

    for year in range(PARAMS["start_year"], PARAMS["end_year"] + 1):
        year_works = fetch_works_for_year(year)
        print(f"{year}: fetched {len(year_works)} works")
        all_works.extend(year_works)

    raw = pd.DataFrame([flatten_work(w) for w in all_works])
    clean = clean_and_engineer_features(raw)

    raw.to_parquet(DATA_DIR / "openalex_ml_works_raw.parquet", index=False)
    clean.to_parquet(DATA_DIR / "openalex_ml_works_clean.parquet", index=False)

    raw.to_csv(DATA_DIR / "openalex_ml_works_raw.csv", index=False)
    clean.to_csv(DATA_DIR / "openalex_ml_works_clean.csv", index=False)

    print("\nDone.")
    print(f"Raw rows: {len(raw):,}")
    print(f"Clean rows: {len(clean):,}")
    print("Saved:")
    print("  data/openalex_ml_works_raw.parquet")
    print("  data/openalex_ml_works_clean.parquet")


if __name__ == "__main__":
    main()