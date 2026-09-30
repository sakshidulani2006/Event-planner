"""
Starlit Serenity - Flask Backend API & Application Server.

Provides RESTful endpoints for:
- GET  /                   : Serves the Starlit Serenity frontend
- GET  /api/health         : Service health check and loaded resources status
- GET  /api/options        : Available cities, event types, cuisines & dataset metrics
- POST /api/recommend      : Explainable rule-based venue recommendation engine
- POST /api/predict-budget : ML budget estimation powered by RandomForestRegressor

Built with clean project-relative paths using pathlib.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Project root and module path setup
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ML_DIR = PROJECT_ROOT / "ml"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

# Import ML modules
from ml.preprocessing import clean_dataset, load_dataset, validate_dataset
from ml.recommender import recommend
from ml.budget_model import DEFAULT_MODEL_PATH, load_model, predict_budget

# Initialize Flask application
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)  # Enable Cross-Origin requests for local development and file-based frontends

# Cache dataset and trained model at startup (do NOT retrain on requests)
_dataset_cache = None
_model_cache = None


def get_dataset():
    global _dataset_cache
    if _dataset_cache is None:
        raw = load_dataset()
        issues = validate_dataset(raw)
        if issues:
            raise RuntimeError(f"Dataset validation failed: {issues}")
        _dataset_cache = clean_dataset(raw)
    return _dataset_cache


def get_model():
    global _model_cache
    if _model_cache is None:
        if not DEFAULT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained budget model not found at {DEFAULT_MODEL_PATH}. "
                "Please run 'python ml/train_model.py' to train it first."
            )
        _model_cache = load_model(DEFAULT_MODEL_PATH)
    return _model_cache


# ─── Static Frontend Routes ───────────────────────────────────────────────
@app.route("/")
def serve_index():
    """Serve the main frontend application."""
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:path>")
def serve_static(path: str):
    """Serve static assets (css, js, images)."""
    target = FRONTEND_DIR / path
    if target.exists() and not target.is_dir():
        return send_from_directory(str(FRONTEND_DIR), path)
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# ─── API Endpoints ────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
    """System health verification endpoint."""
    try:
        df = get_dataset()
        _ = get_model()
        return jsonify({
            "status": "ok",
            "app": "Starlit Serenity",
            "version": "1.0.0",
            "total_venues": len(df),
            "model_loaded": True,
            "message": "All systems operational",
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "app": "Starlit Serenity",
            "error": str(e),
        }), 500


@app.route("/api/options", methods=["GET"])
def get_options():
    """Return available filter options dynamically from the dataset."""
    try:
        df = get_dataset()
        cities = sorted(df["city"].unique().tolist())
        event_types = sorted(df["event_type"].unique().tolist())
        cuisines = sorted(df["cuisine"].unique().tolist())
        venue_types = sorted(df["venue_type"].unique().tolist())

        stats = {
            "total_venues": int(len(df)),
            "min_capacity": int(df["capacity"].min()),
            "max_capacity": int(df["capacity"].max()),
            "min_plate_price": int(df["price_per_plate"].min()),
            "max_plate_price": int(df["price_per_plate"].max()),
            "min_venue_cost": int(df["venue_cost"].min()),
            "max_venue_cost": int(df["venue_cost"].max()),
        }

        return jsonify({
            "success": True,
            "cities": cities,
            "event_types": event_types,
            "cuisines": cuisines,
            "venue_types": venue_types,
            "stats": stats,
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommend", methods=["POST"])
def get_recommendations():
    """
    Generate explainable venue recommendations using the rule-based engine.
    
    Expected JSON payload:
        event_type: str
        city: str
        guests: int (> 0)
        budget: float / int (> 0)
        cuisine: str
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid JSON request body"}), 400

    # 1. Validation
    required = ["event_type", "city", "guests", "budget", "cuisine"]
    missing = [f for f in required if f not in data or data[f] is None or str(data[f]).strip() == ""]
    if missing:
        return jsonify({"success": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        guests = int(data["guests"])
        if guests <= 0:
            return jsonify({"success": False, "error": "Guest count must be a positive integer"}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Guests must be a valid whole number"}), 400

    try:
        budget = float(data["budget"])
        if budget <= 0:
            return jsonify({"success": False, "error": "Budget must be a positive number"}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Budget must be a valid numeric value"}), 400

    event_type = str(data["event_type"]).strip()
    city = str(data["city"]).strip()
    cuisine = str(data["cuisine"]).strip()

    # 2. Execution
    try:
        venues_df = get_dataset()
        results_df = recommend(
            event_type=event_type,
            city=city,
            guests=guests,
            budget=budget,
            cuisine=cuisine,
            top_n=5,
            df=venues_df,
        )

        if results_df.empty:
            return jsonify({
                "success": True,
                "count": 0,
                "query": {
                    "event_type": event_type,
                    "city": city,
                    "guests": guests,
                    "budget": budget,
                    "cuisine": cuisine,
                },
                "recommendations": [],
                "message": (
                    f"No venues in {city} currently match your criteria of {guests} guests "
                    f"for a {event_type}. Try adjusting your guest count or exploring neighboring cities."
                ),
            }), 200

        # Convert recommendations to dictionary list
        records = results_df.to_dict(orient="records")

        # Ensure types are JSON serializable
        for item in records:
            item["rank"] = int(item["rank"])
            item["capacity"] = int(item["capacity"])
            item["price_per_plate"] = int(item["price_per_plate"])
            item["venue_cost"] = int(item["venue_cost"])
            item["estimated_cost"] = int(item["estimated_cost"])
            item["rating"] = float(item["rating"])
            item["compatibility_score"] = float(item["compatibility_score"])
            item["within_budget"] = bool(item["within_budget"])

        return jsonify({
            "success": True,
            "count": len(records),
            "query": {
                "event_type": event_type,
                "city": city,
                "guests": guests,
                "budget": budget,
                "cuisine": cuisine,
            },
            "recommendations": records,
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/predict-budget", methods=["POST"])
def get_budget_prediction():
    """
    Predict total event budget using the trained RandomForestRegressor model.
    
    Expected JSON payload:
        event_type: str
        city: str
        guests: int (>= 10)
        cuisine: str
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid JSON request body"}), 400

    required = ["event_type", "city", "guests", "cuisine"]
    missing = [f for f in required if f not in data or data[f] is None or str(data[f]).strip() == ""]
    if missing:
        return jsonify({"success": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        guests = int(data["guests"])
        if guests < 10:
            return jsonify({"success": False, "error": "Guest count must be at least 10 for ML budget prediction"}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Guests must be a valid whole number"}), 400

    event_type = str(data["event_type"]).strip()
    city = str(data["city"]).strip()
    cuisine = str(data["cuisine"]).strip()

    try:
        model = get_model()
        venues_df = get_dataset()
        prediction_result = predict_budget(
            model=model,
            event_type=event_type,
            city=city,
            guests=guests,
            cuisine=cuisine,
            df=venues_df,
        )

        return jsonify({
            "success": True,
            "prediction": prediction_result,
            "note": "Estimated using RandomForestRegressor trained on venue dataset with synthetic guest patterns.",
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Budget prediction failed: {str(e)}"}), 500


# ─── Error Handlers ───────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "API route not found"}), 404
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "An internal server error occurred"}), 500


if __name__ == "__main__":
    print(f"Starlit Serenity backend running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
