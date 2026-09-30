# Starlit Serenity — Event Planning & Recommendation System

**Starlit Serenity** is a full-stack, modular event planning and venue recommendation system designed as an academic project. It combines an **explainable rule-based multi-factor recommendation engine** with a **Scikit-Learn `RandomForestRegressor` machine learning budget predictor** across 10 major Indian cities.

---

## 1. Project Overview

Finding and budgeting the right venue for weddings, corporate galas, birthdays, and engagements can be overwhelming. Starlit Serenity streamlines this process by:
1. Pruning incompatible venues through strict hard constraints (city location, guest capacity, and event suitability).
2. Scoring remaining venues using a weighted 100-point compatibility model across event type, budget fit, cuisine, rating, capacity utilization, and amenities.
3. Estimating event budgets using a machine learning regression pipeline trained on venue profiles and guest patterns.
4. Delivering genuine, human-readable recommendation reasons explaining each match.

> **Development & Demo Notice:**  
> The dataset (`dataset/event_venues.csv`) contains 160 curated Indian venue records created for development and academic demonstration. Historical event bookings are not included; consequently, guest counts used for training the budget regression model were synthetically generated based on venue capacities (`35%` to `100%` capacity).

---

## 2. Key Features

- **No Bloat & No Complex Dependencies**: Pure Vanilla HTML5, CSS3, ES6 JavaScript frontend paired with a Python Flask backend. Zero Node.js, MongoDB, React, or Firebase dependencies.
- **Dynamic Options**: Form dropdowns and sliders are populated directly from the dataset via `/api/options`.
- **Strict Hard Filters**:
  - City match (case-insensitive)
  - Minimum capacity threshold (`capacity >= guests`)
  - Event type compatibility verification
- **Explainable Recommendation Scoring**: 100-point breakdown with transparent weights and clear bullet-point justifications.
- **Trained Machine Learning Model**: `RandomForestRegressor` model serialized using `joblib` in `ml/models/budget_predictor.pkl`. Evaluated using MAE, RMSE, and $R^2$.
- **Real-Time Interactive Comparison**: Side-by-side view comparing ML predicted budget vs. exact formula cost.
- **1-Click Testing Presets**: Built-in test buttons for instant grading and evaluation.
- **Comprehensive Test Suite**: Automated end-to-end tests for ML training, recommendation logic, and Flask API routes.

---

## 3. System Architecture

```text
┌────────────────────────────────────────────────────────┐
│             Starlit Serenity Frontend                   │
│         (HTML5 · Vanilla CSS3 · Vanilla JS)            │
└───────────────────────────┬────────────────────────────┘
                            │ fetch() JSON HTTP Requests
                            ▼
┌────────────────────────────────────────────────────────┐
│               Flask Backend (app.py)                   │
│       GET /api/options  ·  POST /api/recommend         │
│       GET /api/health   ·  POST /api/predict-budget    │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│   Rule-Based Recommender  │  │   Budget ML Pipeline    │
│    (ml/recommender.py)    │  │  (ml/budget_model.py)   │
│ - Hard Pruning Filters    │  │ - RandomForestRegressor │
│ - 100-Pt Weighted Scoring │  │ - ColumnTransformer     │
│ - Explainable Reason Gen  │  │ - Models (.pkl via Joblib│
└─────────────┬─────────────┘  └───────────┬─────────────┘
              │                            │
              └─────────────┬──────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│             Dataset (dataset/event_venues.csv)         │
│         160 Venues · 10 Indian Cities · 14 Attributes  │
└────────────────────────────────────────────────────────┘
```

---

## 4. Folder Structure

```text
Starlit-Serenity/
├── backend/
│   ├── app.py                     # Flask REST API & static server
│   ├── requirements.txt           # Python dependencies
│   └── test_backend.py            # Automated API test suite
├── dataset/
│   └── event_venues.csv           # 160-row curated venue dataset
├── frontend/
│   ├── index.html                 # Semantic HTML5 single-page application
│   ├── css/
│   │   └── style.css              # Vanilla CSS3 design system & responsive styling
│   ├── js/
│   │   └── script.js              # Vanilla ES6 fetch client & UI controllers
│   └── assets/                    # Media assets directory
├── ml/
│   ├── preprocessing.py           # Dataset loading, validation, and cleaning
│   ├── recommender.py             # Explainable rule-based recommendation engine
│   ├── budget_model.py            # RandomForestRegressor training and inference pipeline
│   ├── train_model.py             # CLI script to train and save budget_predictor.pkl
│   ├── test_recommendation.py     # Recommendation test suite covering key scenarios
│   ├── test_budget_model.py       # End-to-end ML model verification suite
│   └── models/
│       └── budget_predictor.pkl   # Serialized trained RandomForest model
├── README.md                      # Complete system documentation
└── .gitignore                     # Git tracking rules
```

---

## 5. Recommendation Logic

The recommendation engine (`ml/recommender.py`) is deterministic, transparent, and explainable:

### Hard Filters
A venue is discarded if:
1. `city` does not match the target city (case-insensitive).
2. `capacity` is strictly less than `guests`.
3. `event_type` is incompatible with the chosen event.

### Compatibility Score Breakdown (Total: 100 points)

$$\text{Score} = S_{\text{event}} + S_{\text{cuisine}} + S_{\text{budget}} + S_{\text{capacity}} + S_{\text{rating}} + S_{\text{facilities}}$$

| Component | Weight | Scoring Mechanism |
| :--- | :---: | :--- |
| **Event Type** | **25 pts** | Exact match: 1.0 (25 pts); Compatible related event: 0.3–0.5 (7.5–12.5 pts). |
| **Cuisine** | **20 pts** | Exact match: 1.0 (20 pts); Compatible partial match (e.g. Mixed/Both): 0.5–0.7 (10–14 pts). |
| **Budget Fit** | **25 pts** | Estimated cost $\le$ budget: 1.0 (25 pts); Linear penalty for overshoot (reaches 0 at 50% over budget). |
| **Capacity Fit**| **10 pts** | Ideal venue fill rate ($\ge 50\%$ utilization) receives full 10 pts; heavily underutilized venues receive partial credit. |
| **Rating** | **15 pts** | Scaled linearly from venue star rating: $\frac{\text{rating} - 1}{4} \times 15$. |
| **Facilities** | **5 pts** | Evaluates parking (2 pts), in-house decoration (2 pts), and catering (1 pt). |

### Cost Formula
$$\text{Estimated Total Cost} = \text{venue\_cost} + (\text{guests} \times \text{price\_per\_plate})$$

---

## 6. Budget Machine Learning Model

- **Algorithm**: `RandomForestRegressor(n_estimators=200, random_state=42)`
- **Preprocessing**: `ColumnTransformer` with `SimpleImputer` and `OneHotEncoder(handle_unknown='ignore')` for categoricals; median imputation for numericals.
- **Target**: `estimated_total_cost = venue_cost + (guests × price_per_plate)`
- **Synthetic Training Data**: 1,600 samples generated from 160 venues (10 synthetic guest variations per venue between 35% and 100% capacity).
- **Evaluation Metrics (Held-out 20% Test Set)**:
  - **Mean Absolute Error (MAE)**: ₹7,602.36 (2.85% of average target)
  - **Root Mean Squared Error (RMSE)**: ₹13,561.14
  - **Coefficient of Determination ($R^2$)**: 0.9958
- **Cross-Venue Split Validation (Entire venues held out)**:
  - **MAE**: ₹27,550.80
  - **$R^2$**: 0.9564

---

## 7. API Endpoints

### `GET /api/health`
Health check and loaded status.
```json
{
  "status": "ok",
  "app": "Starlit Serenity",
  "version": "1.0.0",
  "total_venues": 160,
  "model_loaded": true
}
```

### `GET /api/options`
Returns unique categorical options and numerical limits directly from the dataset.

### `POST /api/recommend`
Generates top 5 venue recommendations.
- **Request Body**:
  ```json
  {
    "event_type": "Wedding",
    "city": "Jaipur",
    "guests": 150,
    "budget": 200000,
    "cuisine": "Pure Vegetarian"
  }
  ```
- **Response**: Returns list of top venues, compatibility scores, formula cost breakdown, budget status, and genuine recommendation reasons.

### `POST /api/predict-budget`
Infers estimated budget using the trained Random Forest model.
- **Request Body**:
  ```json
  {
    "event_type": "Wedding",
    "city": "Jaipur",
    "guests": 150,
    "cuisine": "Pure Vegetarian"
  }
  ```
- **Response**: Returns `predicted_total_cost`, analytical `formula_total_cost`, `difference`, and baseline venue profile.

---

## 8. Setup and Execution Commands

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. (Optional) Re-train the Budget ML Model
```bash
python ml/train_model.py
```

### 3. Run the Automated Test Suites
```bash
# Test the recommendation engine across all required scenarios
python ml/test_recommendation.py

# Test the ML budget model end-to-end
python ml/test_budget_model.py

# Test all Flask API endpoints
python backend/test_backend.py
```

### 4. Start the Application Server
```bash
python backend/app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

*(Note: Because CORS is enabled in Flask, you can also open `frontend/index.html` via VS Code Live Server or python http.server and it will seamlessly communicate with `http://127.0.0.1:5000`)*.

---

## 9. Verification & Test Scenarios

The system has been tested against the following scenarios:

1. **Scenario 1: Wedding in Jaipur**  
   `Wedding | Jaipur | 150 guests | ₹200,000 | Pure Vegetarian`  
   - Result: Top recommendation is **Jade Resort** (Score: 89.5/100, Cost: ₹227,000, Rating: 4.0★).
2. **Scenario 2: Birthday in Jaipur**  
   `Birthday | Jaipur | 40 guests | ₹50,000 | Mixed / Both`  
   - Result: Top recommendation is **Sunrise Kitchen & Bar** (Score: 86.4/100, Cost: ₹50,000, Within Budget).
3. **Scenario 3: Corporate in Delhi**  
   `Corporate | Delhi | 300 guests | ₹500,000 | Multi-Cuisine`  
   - Result: Top recommendation is **Regal Convention Hall** (Score: 89.1/100, Cost: ₹493,000, Within Budget).
4. **Scenario 4: No-Match Edge Case**  
   `Wedding | Jaipur | 5,000 guests | ₹200,000 | Pure Vegetarian`  
   - Result: Gracefully returns 0 venues with friendly advice message to adjust guest scale.
5. **Scenario 5: Invalid Input Edge Cases**  
   - Negative guests or budget: Returns HTTP 400 with specific error message.
   - Unknown event type: Returns HTTP 400 listing supported event types.

---

## 10. Limitations & Future Improvements

### Current Limitations
- **Synthetic Guest Counts**: The dataset does not provide historical booking data, so guest count variations are synthetically generated for regression training.
- **Fixed Extra Costs**: Decor and photography are not captured in the venue CSV; costs reflect venue rental plus catering only.
- **Single-Day Scope**: Rentals are modeled as single-day events.

### Future Improvements
- Multi-day wedding package calculations (e.g. Sangeet, Mehendi, Reception).
- Integration with live weather and seasonal peak-time multiplier pricing.
- Interactive venue floor plan visualization.
