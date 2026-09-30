"""
Recommendation System Test Suite for Starlit Serenity.

Tests the explainable rule-based recommendation engine across all required scenarios:
1. Wedding | Jaipur | 150 guests | ₹200,000 | Pure Vegetarian
2. Birthday | Jaipur | 40 guests | ₹50,000 | Mixed / Both
3. Corporate | Delhi | 300 guests | ₹500,000 | Multi-Cuisine
4. No-match case (excessive guests or non-existent city)
5. Invalid inputs (negative numbers, unknown types)

Usage:
    python ml/test_recommendation.py
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

import pandas as pd
from recommender import recommend, RESULT_COLUMNS
from preprocessing import load_dataset, validate_dataset


def format_inr(amount: float | int) -> str:
    return f"₹{int(round(amount)):,}"


def run_tests():
    print("=" * 75)
    print("STARLIT SERENITY - RECOMMENDATION SYSTEM VERIFICATION")
    print("=" * 75)

    # 1. Dataset Integrity
    print("\n[TEST 1] Verifying Dataset Integrity...")
    raw = load_dataset()
    issues = validate_dataset(raw)
    assert len(issues) == 0, f"Dataset issues found: {issues}"
    print(f"  ✓ Dataset OK: {raw.shape[0]} rows, {raw.shape[1]} columns. Zero validation errors.")

    # 2. Test Case 1: Wedding in Jaipur
    print("\n[TEST 2] Scenario 1: Wedding | Jaipur | 150 guests | ₹200,000 | Pure Vegetarian")
    q1 = dict(event_type="Wedding", city="Jaipur", guests=150, budget=200000, cuisine="Pure Vegetarian")
    res1 = recommend(**q1)
    assert not res1.empty, "Expected recommendations for Scenario 1"
    print(f"  ✓ Returned {len(res1)} venues (Top {len(res1)}):")
    for _, r in res1.iterrows():
        print(f"    • Rank #{r['rank']} | {r['venue_name']} ({r['venue_type']})")
        print(f"      Score: {r['compatibility_score']}/100 | Cost: {format_inr(r['estimated_cost'])} | Rating: {r['rating']}★ | Within Budget: {r['within_budget']}")
        for reason in r["match_reasons"]:
            print(f"        - {reason}")
    top1 = res1.iloc[0]
    assert top1["city"] == "Jaipur", "City must match Jaipur"
    assert top1["capacity"] >= 150, "Capacity must be >= 150"

    # 3. Test Case 2: Birthday in Jaipur
    print("\n[TEST 3] Scenario 2: Birthday | Jaipur | 40 guests | ₹50,000 | Mixed / Both")
    q2 = dict(event_type="Birthday", city="Jaipur", guests=40, budget=50000, cuisine="Mixed / Both")
    res2 = recommend(**q2)
    assert not res2.empty, "Expected recommendations for Scenario 2"
    print(f"  ✓ Returned {len(res2)} venues:")
    for _, r in res2.iterrows():
        print(f"    • Rank #{r['rank']} | {r['venue_name']} ({r['venue_type']}) | Score: {r['compatibility_score']}/100 | Cost: {format_inr(r['estimated_cost'])}")
    top2 = res2.iloc[0]
    assert top2["city"] == "Jaipur"
    assert top2["capacity"] >= 40

    # 4. Test Case 3: Corporate in Delhi
    print("\n[TEST 4] Scenario 3: Corporate | Delhi | 300 guests | ₹500,000 | Multi-Cuisine")
    q3 = dict(event_type="Corporate", city="Delhi", guests=300, budget=500000, cuisine="Multi-Cuisine")
    res3 = recommend(**q3)
    assert not res3.empty, "Expected recommendations for Scenario 3"
    print(f"  ✓ Returned {len(res3)} venues:")
    for _, r in res3.iterrows():
        print(f"    • Rank #{r['rank']} | {r['venue_name']} ({r['venue_type']})")
        print(f"      Score: {r['compatibility_score']}/100 | Cost: {format_inr(r['estimated_cost'])} | Rating: {r['rating']}★ | Within Budget: {r['within_budget']}")
        for reason in r["match_reasons"]:
            print(f"        - {reason}")
    top3 = res3.iloc[0]
    assert top3["city"] == "Delhi"
    assert top3["capacity"] >= 300
    assert top3["event_type"] == "Corporate"

    # 5. Test Case 4: No-match case (over capacity)
    print("\n[TEST 5] Scenario 4: No-Match Scenario (Excessive Guests = 5,000)")
    q4 = dict(event_type="Wedding", city="Jaipur", guests=5000, budget=200000, cuisine="Pure Vegetarian")
    res4 = recommend(**q4)
    assert res4.empty, "Expected empty DataFrame for capacity > 5000"
    assert list(res4.columns) == RESULT_COLUMNS, "Empty result must preserve columns"
    print("  ✓ Correctly returned empty recommendations DataFrame without error.")

    # 6. Test Case 5: Invalid Input Handling
    print("\n[TEST 6] Scenario 5: Input Validation & Edge Cases")
    # Negative guests
    try:
        recommend(event_type="Wedding", city="Jaipur", guests=-10, budget=200000, cuisine="Pure Vegetarian")
        raise AssertionError("Should have raised ValueError on negative guests")
    except ValueError as e:
        print(f"  ✓ Caught invalid guests: {e}")

    # Negative budget
    try:
        recommend(event_type="Wedding", city="Jaipur", guests=50, budget=-500, cuisine="Pure Vegetarian")
        raise AssertionError("Should have raised ValueError on negative budget")
    except ValueError as e:
        print(f"  ✓ Caught invalid budget: {e}")

    # Unknown event type
    try:
        recommend(event_type="AlienCeremony", city="Jaipur", guests=50, budget=50000, cuisine="Pure Vegetarian")
        raise AssertionError("Should have raised ValueError on unknown event type")
    except ValueError as e:
        print(f"  ✓ Caught unknown event type: {e}")

    print("\n" + "=" * 75)
    print("ALL 6 RECOMMENDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_tests()
