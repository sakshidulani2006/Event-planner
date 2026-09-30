"""
Model Training Script for Starlit Serenity Budget Predictor.

Trains a RandomForestRegressor model on the venue dataset with documented
synthetic guest counts to predict total event cost.

Formula Target:
    estimated_total_cost = venue_cost + (guests * price_per_plate)

Evaluation Metrics:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Coefficient of Determination (R²)

Saves model to:
    ml/models/budget_predictor.pkl

Usage:
    python ml/train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add ml directory and project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from budget_model import (
    DEFAULT_MODEL_PATH,
    SYNTHETIC_GUESTS_NOTE,
    create_training_data,
    evaluate_model,
    evaluate_unseen_venues,
    load_data,
    save_model,
    train_model,
)


def run_training() -> dict:
    print("=" * 65)
    print("STARLIT SERENITY - BUDGET MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load and clean venue dataset
    print("\n[1/4] Loading venue dataset...")
    venues = load_data()
    print(f"      Loaded {len(venues)} venues across {venues['city'].nunique()} cities.")

    # 2. Generate training data with documented synthetic guests
    print("\n[2/4] Generating training data with synthetic guest counts...")
    print(f"      Note: {SYNTHETIC_GUESTS_NOTE}")
    train_df = create_training_data(venues, samples_per_venue=10, seed=42)
    print(f"      Created {len(train_df)} training samples.")
    print(f"      Guest range: {int(train_df['guests'].min())} to {int(train_df['guests'].max())}")
    print(f"      Target cost range: ₹{int(train_df['estimated_total_cost'].min()):,} to ₹{int(train_df['estimated_total_cost'].max()):,}")

    # 3. Train RandomForestRegressor
    print("\n[3/4] Fitting RandomForestRegressor (80/20 train-test split)...")
    model, X_test, y_test = train_model(train_df)
    metrics = evaluate_model(model, X_test, y_test)

    print("      Test Set Evaluation:")
    print(f"      • Mean Absolute Error (MAE) : ₹{metrics['MAE']:,.2f}")
    print(f"      • Root Mean Squared Error (RMSE): ₹{metrics['RMSE']:,.2f}")
    print(f"      • R² Score                  : {metrics['R2']:.4f}")
    print(f"      • Mean Target in Test Set   : ₹{metrics['mean_target']:,.2f}")
    print(f"      • Relative MAE              : {(metrics['MAE'] / metrics['mean_target']) * 100:.2f}%")

    # Stricter cross-venue validation
    unseen_metrics = evaluate_unseen_venues(train_df)
    print("\n      Cross-Venue Validation (Venues held out entirely):")
    print(f"      • MAE  : ₹{unseen_metrics['MAE']:,.2f}")
    print(f"      • RMSE : ₹{unseen_metrics['RMSE']:,.2f}")
    print(f"      • R²   : {unseen_metrics['R2']:.4f}")

    # 4. Save model artifact
    print("\n[4/4] Saving trained model artifact...")
    saved_path = save_model(model, metrics=metrics, path=DEFAULT_MODEL_PATH)
    print(f"      Successfully saved to: {saved_path}")
    print("\nTraining completed successfully!\n")
    return metrics


if __name__ == "__main__":
    run_training()
