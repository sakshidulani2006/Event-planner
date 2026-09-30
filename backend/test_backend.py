"""
Backend API Test Suite for Starlit Serenity.

Directly tests Flask endpoints using app.test_client():
- GET  /api/health
- GET  /api/options
- POST /api/recommend (Scenario 1, Scenario 2, Scenario 3, No Match, Invalid Inputs)
- POST /api/predict-budget
- GET  / (Frontend static serving)

Usage:
    python backend/test_backend.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import app


def test_api():
    print("=" * 75)
    print("STARLIT SERENITY - BACKEND API END-TO-END VERIFICATION")
    print("=" * 75)

    client = app.test_client()

    # 1. Health Check
    print("\n[TEST 1] GET /api/health")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    print(f"  ✓ Status: {data.get('status')} | App: {data.get('app')} | Venues: {data.get('total_venues')}")

    # 2. Options
    print("\n[TEST 2] GET /api/options")
    res = client.get("/api/options")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    print(f"  ✓ Cities ({len(data['cities'])}): {data['cities']}")
    print(f"  ✓ Event Types ({len(data['event_types'])}): {data['event_types']}")
    print(f"  ✓ Cuisines ({len(data['cuisines'])}): {data['cuisines']}")
    print(f"  ✓ Venue Types ({len(data['venue_types'])}): {data['venue_types']}")

    # 3. Recommendations - Test 1: Wedding in Jaipur
    print("\n[TEST 3] POST /api/recommend - Wedding | Jaipur | 150 guests | ₹200k | Pure Vegetarian")
    p1 = {
        "event_type": "Wedding",
        "city": "Jaipur",
        "guests": 150,
        "budget": 200000,
        "cuisine": "Pure Vegetarian",
    }
    res = client.post("/api/recommend", data=json.dumps(p1), content_type="application/json")
    assert res.status_code == 200
    d1 = res.get_json()
    assert d1["success"] is True
    print(f"  ✓ Returned {d1['count']} recommendations. Top venue: {d1['recommendations'][0]['venue_name']} (Score: {d1['recommendations'][0]['compatibility_score']})")

    # 4. Recommendations - Test 2: Birthday in Jaipur
    print("\n[TEST 4] POST /api/recommend - Birthday | Jaipur | 40 guests | ₹50k | Mixed / Both")
    p2 = {
        "event_type": "Birthday",
        "city": "Jaipur",
        "guests": 40,
        "budget": 50000,
        "cuisine": "Mixed / Both",
    }
    res = client.post("/api/recommend", data=json.dumps(p2), content_type="application/json")
    assert res.status_code == 200
    d2 = res.get_json()
    assert d2["success"] is True
    print(f"  ✓ Returned {d2['count']} recommendations. Top venue: {d2['recommendations'][0]['venue_name']} (Score: {d2['recommendations'][0]['compatibility_score']})")

    # 5. Recommendations - Test 3: Corporate in Delhi
    print("\n[TEST 5] POST /api/recommend - Corporate | Delhi | 300 guests | ₹500k | Multi-Cuisine")
    p3 = {
        "event_type": "Corporate",
        "city": "Delhi",
        "guests": 300,
        "budget": 500000,
        "cuisine": "Multi-Cuisine",
    }
    res = client.post("/api/recommend", data=json.dumps(p3), content_type="application/json")
    assert res.status_code == 200
    d3 = res.get_json()
    assert d3["success"] is True
    print(f"  ✓ Returned {d3['count']} recommendations. Top venue: {d3['recommendations'][0]['venue_name']} (Score: {d3['recommendations'][0]['compatibility_score']})")

    # 6. Recommendations - No Match
    print("\n[TEST 6] POST /api/recommend - No Match Case (5000 guests)")
    p4 = {
        "event_type": "Wedding",
        "city": "Jaipur",
        "guests": 5000,
        "budget": 200000,
        "cuisine": "Pure Vegetarian",
    }
    res = client.post("/api/recommend", data=json.dumps(p4), content_type="application/json")
    assert res.status_code == 200
    d4 = res.get_json()
    assert d4["success"] is True
    assert d4["count"] == 0
    print(f"  ✓ Gracefully returned 0 results: '{d4.get('message')}'")

    # 7. Recommendations - Invalid Inputs
    print("\n[TEST 7] POST /api/recommend - Invalid Input (Negative guests)")
    bad = {"event_type": "Wedding", "city": "Jaipur", "guests": -10, "budget": 200000, "cuisine": "Pure Vegetarian"}
    res = client.post("/api/recommend", data=json.dumps(bad), content_type="application/json")
    assert res.status_code == 400
    print(f"  ✓ Handled 400 error correctly: {res.get_json().get('error')}")

    # 8. Budget Prediction Endpoint
    print("\n[TEST 8] POST /api/predict-budget - ML Model Evaluation")
    pred_payload = {
        "event_type": "Wedding",
        "city": "Jaipur",
        "guests": 150,
        "cuisine": "Pure Vegetarian",
    }
    res = client.post("/api/predict-budget", data=json.dumps(pred_payload), content_type="application/json")
    assert res.status_code == 200
    pred_data = res.get_json()
    assert pred_data["success"] is True
    pred_info = pred_data["prediction"]
    print(f"  ✓ ML Predicted Cost : ₹{pred_info['predicted_total_cost']:,}")
    print(f"  ✓ Formula Cost      : ₹{pred_info['formula_total_cost']:,}")
    print(f"  ✓ Difference        : ₹{pred_info['difference']:+,}")
    print(f"  ✓ Venue Used        : {pred_info['venue_used']}")

    # 9. Frontend Serving
    print("\n[TEST 9] GET / - Frontend Serving")
    res = client.get("/")
    assert res.status_code == 200
    assert b"Starlit Serenity" in res.data
    print("  ✓ Successfully served frontend index.html via Flask")

    print("\n" + "=" * 75)
    print("ALL 9 BACKEND API TESTS PASSED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    test_api()
