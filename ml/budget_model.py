"""
Budget prediction model (Phase 4) for the Event Planning & Recommendation System.

Goal: predict `estimated_total_cost` from event and venue information.

*** IMPORTANT - READ THIS BEFORE TRUSTING ANY NUMBER ***
1. The dataset (dataset/event_venues.csv) is SYNTHETIC demo data.
2. It has NO `guests` column. Guest counts used here are SYNTHETIC too: they are
   randomly generated for each venue (between ~35% and 100% of its capacity)
   purely so the pipeline has something to learn from.
3. The target is computed as
       estimated_total_cost = venue_cost + guests * price_per_plate
   and BOTH venue_cost and price_per_plate are also model inputs. The target is
   therefore a direct formula of the features. A high R^2 only shows that the
   forest can approximate that formula - it does NOT mean the model can forecast
   real-world event budgets. This is a demonstration ML pipeline.

Pipeline: load -> preprocess -> add synthetic guests + target -> ColumnTransformer
(impute + one-hot / impute) -> RandomForestRegressor -> evaluate -> save.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:  # works both as `ml.budget_model` and when run as a script
    from .preprocessing import DEFAULT_DATASET_PATH, preprocess
except ImportError:
    from preprocessing import DEFAULT_DATASET_PATH, preprocess

# ─── Constants ────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "budget_predictor.pkl"

TARGET = "estimated_total_cost"
CATEGORICAL_FEATURES = ["event_type", "city", "venue_type", "cuisine", "indoor_outdoor"]
NUMERIC_FEATURES = [
    "capacity", "price_per_plate", "venue_cost", "rating",
    "parking_available", "catering_available", "decoration_available",
    "guests",  # SYNTHETIC - see module docstring
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLES_PER_VENUE = 10          # synthetic guest counts generated per venue
MIN_FILL, MAX_FILL = 0.35, 1.0  # guests as a fraction of venue capacity
MIN_GUESTS = 10

SYNTHETIC_GUESTS_NOTE = (
    "The dataset has no guest counts. `guests` was randomly generated per venue "
    f"({MIN_FILL:.0%}-{MAX_FILL:.0%} of capacity, min {MIN_GUESTS}) for training only, and "
    "the target is venue_cost + guests * price_per_plate."
)


# ─── 1. Data ──────────────────────────────────────────────────────────────
def load_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load, validate and clean the venue dataset (Yes/No columns become 1/0)."""
    return preprocess(path or DEFAULT_DATASET_PATH)


def _to_model_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert feature columns to plain object / float dtypes with np.nan for gaps.
    pandas' nullable types (string, Int64) use pd.NA, which scikit-learn's imputers
    cannot compare, so missing values would crash instead of being imputed.
    """
    out = frame.copy()
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


def create_training_data(
    df: pd.DataFrame,
    samples_per_venue: int = SAMPLES_PER_VENUE,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Build the training table.

    For every venue we draw `samples_per_venue` SYNTHETIC guest counts between
    MIN_FILL and MAX_FILL of its capacity (never above capacity, never below
    MIN_GUESTS), round them to the nearest 5, then compute the target:
        estimated_total_cost = venue_cost + guests * price_per_plate
    """
    if samples_per_venue < 1:
        raise ValueError("samples_per_venue must be at least 1")
    if df.empty:
        raise ValueError("Cannot create training data from an empty dataset")

    rng = np.random.default_rng(seed)  # fixed seed -> reproducible synthetic guests
    data = df.loc[df.index.repeat(samples_per_venue)].reset_index(drop=True)

    fill = rng.uniform(MIN_FILL, MAX_FILL, size=len(data))
    guests = np.round(data["capacity"].to_numpy() * fill / 5) * 5
    # keep guests inside [MIN_GUESTS, capacity]; capacity itself can be a non-multiple of 5
    guests = np.clip(guests, MIN_GUESTS, data["capacity"].to_numpy())
    data["guests"] = guests.astype(int)

    data[TARGET] = data["venue_cost"] + data["guests"] * data["price_per_plate"]
    return _to_model_dtypes(data)


# ─── 2. Model ─────────────────────────────────────────────────────────────
def build_model(n_estimators: int = 200, random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Preprocessing + RandomForestRegressor in one Pipeline.

    - categorical columns: fill missing with the most frequent value, then one-hot
      encode (unknown categories at prediction time are ignored instead of crashing)
    - numeric columns: fill missing with the median (trees need no scaling)
    """
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer([
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("num", numeric, NUMERIC_FEATURES),
    ])
    forest = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    return Pipeline([("preprocess", preprocessor), ("model", forest)])


def _check_training_frame(data: pd.DataFrame) -> None:
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    if data[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    if (data[TARGET] <= 0).any():
        raise ValueError("Target must be positive")


def train_model(data: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """
    Split (80/20, random_state=42), fit the pipeline on the training part.
    Returns (fitted_model, X_test, y_test).
    """
    _check_training_frame(data)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """MAE, RMSE and R^2 on held-out data (values in INR, except R^2)."""
    pred = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "n_test": int(len(y_test)),
        "mean_target": float(np.mean(y_test)),
    }


def evaluate_unseen_venues(data: pd.DataFrame, random_state: int = RANDOM_STATE) -> dict:
    """
    Stricter extra check: hold out whole VENUES (grouped split).

    The 80/20 row split above puts different guest counts of the SAME venue in
    both train and test, which flatters the score. Here no test venue is seen in
    training. Still synthetic data and a formula target - a sanity check only.
    """
    _check_training_frame(data)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_state)
    train_idx, test_idx = next(splitter.split(data, groups=data["venue_id"]))
    model = build_model(random_state=random_state)
    model.fit(data.iloc[train_idx][FEATURES], data.iloc[train_idx][TARGET])
    return evaluate_model(model, data.iloc[test_idx][FEATURES], data.iloc[test_idx][TARGET])


# ─── 3. Save / load ───────────────────────────────────────────────────────
def save_model(model: Pipeline, metrics: Optional[dict] = None,
               path: Optional[str | Path] = None) -> Path:
    """Save the pipeline plus metadata with joblib. Creates ml/models/ if needed."""
    target_path = Path(path) if path else DEFAULT_MODEL_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics or {},
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": SYNTHETIC_GUESTS_NOTE,
    }
    joblib.dump(bundle, target_path, compress=3)  # compressed: forest files get large
    return target_path


def load_model(path: Optional[str | Path] = None) -> Pipeline:
    """Load a saved pipeline. Only load .pkl files you created yourself (pickle is not safe for untrusted files)."""
    source = Path(path) if path else DEFAULT_MODEL_PATH
    if not source.exists():
        raise FileNotFoundError(f"No saved model at {source}. Train and save one first.")
    return joblib.load(source)["model"]


# ─── 4. Prediction ────────────────────────────────────────────────────────
def _canonical(value: str, valid: list[str], label: str) -> str:
    """Case-insensitive match against the valid options, else a clear error."""
    key = str(value).strip().lower()
    for v in valid:
        if v.lower() == key:
            return v
    raise ValueError(f"Unknown {label} '{value}'. Valid options: {sorted(valid)}")


def select_venue_profile(df: pd.DataFrame, event_type: str, city: str,
                         guests: int, cuisine: str) -> tuple[pd.Series, str]:
    """
    The user gives event type / city / guests / cuisine, but the model also needs
    venue features. We pick ONE real dataset venue that is "typical" for the request:

      candidates = venues that can hold `guests`, matched progressively:
        1) same city + event type + cuisine
        2) same city + event type          (cuisine relaxed)
        3) same city                       (event type relaxed)
        4) any city                        (city relaxed)
      typical venue = the candidate with the median price_per_plate.

    Returns (venue_row, how_it_was_matched). Raises ValueError if nobody can hold `guests`.
    """
    fits = df[df["capacity"] >= guests]
    in_city = fits[fits["city"] == city]
    steps = [
        ("city + event type + cuisine", in_city[(in_city["event_type"] == event_type)
                                                & (in_city["cuisine"] == cuisine)]),
        ("city + event type (cuisine relaxed)", in_city[in_city["event_type"] == event_type]),
        ("city only (event type relaxed)", in_city),
        ("any city (city relaxed)", fits),
    ]
    for label, cand in steps:
        if not cand.empty:
            ordered = cand.sort_values(["price_per_plate", "venue_id"]).reset_index(drop=True)
            return ordered.iloc[(len(ordered) - 1) // 2], label  # lower median for even counts
    raise ValueError(f"No venue in the dataset can hold {guests} guests "
                     f"(max capacity is {int(df['capacity'].max())}).")


def predict_budget(model: Pipeline, event_type: str, city: str, guests: int,
                   cuisine: str, df: Optional[pd.DataFrame] = None) -> dict:
    """
    Predict the estimated total cost (INR) for a request.

    Venue-related inputs come from `select_venue_profile`. The result also holds
    the plain formula value for the same venue, so you can see how close the
    forest gets to the formula it was trained to imitate.
    """
    data = load_data() if df is None else df

    # ---- validate input ----
    if isinstance(guests, bool) or not isinstance(guests, (int, np.integer)):
        raise ValueError("guests must be a whole number")
    if guests < MIN_GUESTS:
        raise ValueError(f"guests must be at least {MIN_GUESTS}")
    event = _canonical(event_type, data["event_type"].unique().tolist(), "event type")
    place = _canonical(city, data["city"].unique().tolist(), "city")
    cuis = _canonical(cuisine, data["cuisine"].unique().tolist(), "cuisine")

    venue, matched = select_venue_profile(data, event, place, int(guests), cuis)

    # ---- build the one-row input the pipeline expects ----
    # Every venue feature comes from the REAL venue record (so price_per_plate,
    # cuisine, event_type etc. stay consistent with each other). Only `guests`
    # comes from the user. If the match was relaxed, the venue's own event type /
    # cuisine are used - see `venue_match` in the result.
    row = pd.DataFrame([{**{c: venue[c] for c in FEATURES if c != "guests"},
                         "guests": int(guests)}])[FEATURES]
    row = _to_model_dtypes(row)
    # missing values in the row are handled by the pipeline's imputers
    predicted = float(model.predict(row)[0])
    formula = float(venue["venue_cost"] + guests * venue["price_per_plate"])

    return {
        "predicted_total_cost": round(predicted),
        "formula_total_cost": round(formula),
        "difference": round(predicted - formula),
        "venue_used": f"{venue['venue_id']} {venue['venue_name']} ({venue['venue_type']}, {venue['city']})",
        "venue_match": matched,
        "venue_cuisine": venue["cuisine"],
        "venue_event_type": venue["event_type"],
        "inputs": {"event_type": event, "city": place, "guests": int(guests), "cuisine": cuis},
    }
