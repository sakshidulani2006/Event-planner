"""
Reusable data-loading and basic preprocessing helpers for the
Event Planning & Recommendation System.

Scope (Phase 2): loading, validating, cleaning and encoding the venue dataset.
No model is trained here.

The dataset (dataset/event_venues.csv) is SYNTHETIC demo data created for
development and demonstration. It is not real-world data.

Column notes
------------
catering_available : "Yes" = in-house catering. "No" = the venue has no
    in-house kitchen; price_per_plate is then the typical rate of partner /
    external caterers, so it is still a positive number.
capacity           : maximum number of guests the venue can hold.
price_per_plate    : INR per guest.
venue_cost         : INR, venue rental for one day.

Quick check from the project root:
    python ml/preprocessing.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "event_venues.csv"

# ─── Schema ───────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "venue_id", "venue_name", "city", "venue_type", "event_type", "capacity",
    "cuisine", "price_per_plate", "venue_cost", "rating", "parking_available",
    "indoor_outdoor", "catering_available", "decoration_available",
]
NUMERIC_COLUMNS = ["capacity", "price_per_plate", "venue_cost", "rating"]
CATEGORICAL_COLUMNS = ["city", "venue_type", "event_type", "cuisine", "indoor_outdoor"]
BINARY_COLUMNS = ["parking_available", "catering_available", "decoration_available"]
TEXT_COLUMNS = ["venue_id", "venue_name"] + CATEGORICAL_COLUMNS + BINARY_COLUMNS

VALID_INDOOR_OUTDOOR = {"Indoor", "Outdoor", "Both"}
_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


# ─── Loading ──────────────────────────────────────────────────────────────
def load_dataset(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Read the venue CSV into a DataFrame (no cleaning applied)."""
    csv_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


# ─── Validation ───────────────────────────────────────────────────────────
def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems. An empty list means valid."""
    problems: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return [f"Missing required columns: {missing_cols}"]

    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    for col, n in null_counts[null_counts > 0].items():
        problems.append(f"{n} missing value(s) in '{col}'")

    if df["venue_id"].duplicated().any():
        problems.append("Duplicate venue_id values found")
    if df.duplicated(subset=[c for c in REQUIRED_COLUMNS if c != "venue_id"]).any():
        problems.append("Duplicate records found (identical apart from venue_id)")

    for col in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().sum() > df[col].isna().sum():
            problems.append(f"Non-numeric value(s) in '{col}'")
    for col in ["capacity", "price_per_plate", "venue_cost"]:
        if (pd.to_numeric(df[col], errors="coerce") <= 0).any():
            problems.append(f"'{col}' must be positive")
    ratings = pd.to_numeric(df["rating"], errors="coerce")
    if ((ratings < 1) | (ratings > 5)).any():
        problems.append("'rating' must be between 1 and 5")

    for col in BINARY_COLUMNS:
        bad = ~df[col].astype(str).str.strip().str.lower().isin(_YES | _NO)
        if bad.any():
            problems.append(f"'{col}' has values other than Yes/No")
    bad_io = ~df["indoor_outdoor"].astype(str).str.strip().str.title().isin(VALID_INDOOR_OUTDOOR)
    if bad_io.any():
        problems.append("'indoor_outdoor' must be Indoor, Outdoor or Both")

    return problems


# ─── Cleaning ─────────────────────────────────────────────────────────────
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: strip whitespace, coerce numeric columns, drop exact
    duplicates and rows missing key fields, and convert Yes/No columns to 1/0.
    Returns a new DataFrame; the input is not modified.
    """
    out = df.copy()

    for col in TEXT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["indoor_outdoor"] = out["indoor_outdoor"].str.title()
    for col in BINARY_COLUMNS:
        out[col] = yes_no_to_int(out[col])

    out = out.drop_duplicates(subset=[c for c in out.columns if c != "venue_id"])
    out = out.dropna(subset=["venue_id", "city", "event_type", "capacity",
                             "price_per_plate", "venue_cost", "rating"])

    for col in ["capacity", "price_per_plate", "venue_cost"]:
        out[col] = out[col].astype(int)
    return out.reset_index(drop=True)


def yes_no_to_int(series: pd.Series) -> pd.Series:
    """Map Yes/No style values to 1/0 (unknown values become NA)."""
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.map(lambda v: 1 if v in _YES else 0 if v in _NO else pd.NA).astype("Int64")


# ─── Encoding ─────────────────────────────────────────────────────────────
def encode_features(
    df: pd.DataFrame,
    categorical_columns: Iterable[str] = CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """
    One-hot encode the categorical columns of an already-cleaned DataFrame.
    Numeric and 0/1 columns are kept as they are. Identifier columns
    (venue_id, venue_name) are dropped.
    """
    features = df.drop(columns=["venue_id", "venue_name"], errors="ignore")
    return pd.get_dummies(features, columns=list(categorical_columns), dtype=int)


def preprocess(path: Optional[str | Path] = None, encode: bool = False) -> pd.DataFrame:
    """
    Load -> validate -> clean (-> encode). Raises ValueError if validation
    finds problems, so bad data never passes silently.
    """
    df = load_dataset(path)
    problems = validate_dataset(df)
    if problems:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(problems))
    df = clean_dataset(df)
    return encode_features(df) if encode else df


def get_unique_options(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted unique values of each categorical column (useful for form/API validation later)."""
    return {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_COLUMNS}


# ─── Quick manual check ───────────────────────────────────────────────────
if __name__ == "__main__":
    raw = load_dataset()
    issues = validate_dataset(raw)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from {DEFAULT_DATASET_PATH}")
    print("Validation:", "OK" if not issues else issues)
    if not issues:
        cleaned = clean_dataset(raw)
        print(f"After cleaning: {cleaned.shape[0]} rows")
        print(f"Encoded feature matrix: {encode_features(cleaned).shape}")
