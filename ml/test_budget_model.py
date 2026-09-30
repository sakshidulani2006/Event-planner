"""
Runs the whole budget-model pipeline end to end and prints the real numbers.

    python ml/test_budget_model.py

Nothing here is hard-coded: metrics and predictions come from the trained model.
Reminder: data and guest counts are synthetic and the target is a formula of the
inputs, so these scores show a working pipeline, not real-world accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow running from any folder

import numpy as np

from budget_model import (
    DEFAULT_MODEL_PATH, SYNTHETIC_GUESTS_NOTE, create_training_data, evaluate_model,
    evaluate_unseen_venues, load_data, load_model, predict_budget, save_model, train_model,
)

EXAMPLES = [
    dict(event_type="Wedding", city="Jaipur", guests=150, cuisine="Pure Vegetarian"),
    dict(event_type="Birthday", city="Jaipur", guests=40, cuisine="Mixed / Both"),
    dict(event_type="Corporate", city="Delhi", guests=300, cuisine="Multi-Cuisine"),
]


def inr(x: float) -> str:
    return f"INR {x:,.0f}"


def main() -> None:
    print("=" * 70)
    print("1. LOAD DATA")
    venues = load_data()
    print(f"   venues: {venues.shape[0]} rows, {venues.shape[1]} columns")

    print("\n2. CREATE TRAINING DATA (guests are SYNTHETIC)")
    data = create_training_data(venues)
    print(f"   {SYNTHETIC_GUESTS_NOTE}")
    print(f"   training table: {data.shape[0]} rows "
          f"(guests {int(data['guests'].min())}-{int(data['guests'].max())}, "
          f"target {inr(data['estimated_total_cost'].min())} - {inr(data['estimated_total_cost'].max())})")
    assert (data["guests"] <= data["capacity"]).all(), "guests must not exceed capacity"

    print("\n3. TRAIN (RandomForestRegressor, test_size=0.2, random_state=42)")
    model, X_test, y_test = train_model(data)
    print(f"   train rows: {len(data) - len(y_test)}, test rows: {len(y_test)}")

    print("\n4. EVALUATE")
    m = evaluate_model(model, X_test, y_test)
    print(f"   MAE  : {inr(m['MAE'])}")
    print(f"   RMSE : {inr(m['RMSE'])}")
    print(f"   R2   : {m['R2']:.4f}")
    print(f"   (average test target = {inr(m['mean_target'])}; MAE is {m['MAE'] / m['mean_target']:.1%} of it)")
    g = evaluate_unseen_venues(data)
    print("   Stricter check - whole venues held out of training:")
    print(f"   MAE  : {inr(g['MAE'])}   RMSE : {inr(g['RMSE'])}   R2 : {g['R2']:.4f}")
    print("   NOTE: the target is venue_cost + guests * price_per_plate and both of those are inputs,")
    print("   so this measures how well the forest copies a formula, not real budget forecasting.")

    print("\n5. SAVE / RELOAD")
    path = save_model(model, metrics=m)
    reloaded = load_model(path)
    print(f"   saved to {path.relative_to(path.parents[2])}")
    same = np.allclose(reloaded.predict(X_test.head(50)), model.predict(X_test.head(50)))
    print(f"   reloaded model gives identical predictions: {same}")
    assert same

    print("\n6. SAMPLE PREDICTIONS")
    for i, ex in enumerate(EXAMPLES, 1):
        r = predict_budget(reloaded, df=venues, **ex)
        print(f"   Example {i}: {ex['event_type']}, {ex['city']}, {ex['guests']} guests, {ex['cuisine']}")
        print(f"      venue used (matched on {r['venue_match']}):")
        print(f"        {r['venue_used']}, listed cuisine: {r['venue_cuisine']}")
        print(f"      model prediction : {inr(r['predicted_total_cost'])}")
        print(f"      plain formula    : {inr(r['formula_total_cost'])}  (difference {r['difference']:+,})")

    print("\n7. INVALID INPUT HANDLING")
    bad_cases = [
        ("unknown city", dict(event_type="Wedding", city="Atlantis", guests=100, cuisine="Pure Vegetarian")),
        ("unknown event", dict(event_type="Funeral", city="Jaipur", guests=100, cuisine="Pure Vegetarian")),
        ("unknown cuisine", dict(event_type="Wedding", city="Jaipur", guests=100, cuisine="Martian")),
        ("guests = 0", dict(event_type="Wedding", city="Jaipur", guests=0, cuisine="Pure Vegetarian")),
        ("guests = 12.5", dict(event_type="Wedding", city="Jaipur", guests=12.5, cuisine="Pure Vegetarian")),
        ("guests too large", dict(event_type="Wedding", city="Jaipur", guests=5000, cuisine="Pure Vegetarian")),
    ]
    for label, kwargs in bad_cases:
        try:
            predict_budget(reloaded, df=venues, **kwargs)
            raise AssertionError(f"'{label}' should have raised ValueError")
        except ValueError as e:
            print(f"   {label:18s} -> ValueError: {str(e)[:70]}")

    print("\n8. MISSING VALUE HANDLING")
    broken = X_test.head(5).copy()
    broken.loc[broken.index[0], "rating"] = np.nan
    broken.loc[broken.index[1], "city"] = np.nan
    out = reloaded.predict(broken)
    print(f"   predictions with NaNs in rating/city: {np.round(out).astype(int).tolist()}  (no crash)")
    assert not np.isnan(out).any()

    print("\nDone. Phase 4 model runs end to end.")


if __name__ == "__main__":
    main()
