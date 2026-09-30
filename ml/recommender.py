"""
Rule-based venue recommender for the Event Planning & Recommendation System.

Flow (as specified):
    User input (event type, city, guests, budget, cuisine)
        -> filter + calculate compatibility
        -> top 3-5 venues

This is plain filtering and weighted scoring. There is NO trained ML model here.
It runs on the SYNTHETIC demo dataset in dataset/event_venues.csv.

Hard filters (a venue is dropped if it fails):
    - city matches
    - capacity >= number of guests

Compatibility score (0-100), weights below:
    event type 25 | cuisine 20 | budget 25 | capacity fit 10 | rating 15 | facilities 5

Estimated cost = venue_cost + guests * price_per_plate (venue + catering only;
decor, photography etc. are not in the dataset). It is compared with the full
budget, so a venue can score well on budget even if extras are still to be added.

Try it from the project root:
    python ml/recommender.py
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

try:  # works both as `ml.recommender` and when run as a script
    from .preprocessing import preprocess
except ImportError:
    from preprocessing import preprocess

# ─── Weights (sum = 100) ──────────────────────────────────────────────────
WEIGHTS = {"event": 25, "cuisine": 20, "budget": 25, "capacity": 10, "rating": 15, "facilities": 5}

# ─── Friendly names accepted from the UI / forms ──────────────────────────
EVENT_ALIASES = {
    "birthday party": "Birthday", "engagement ceremony": "Engagement",
    "corporate event": "Corporate", "other": "Other Party", "party": "Other Party",
}
CUISINE_ALIASES = {
    "vegetarian": "Pure Vegetarian", "veg": "Pure Vegetarian", "pure veg": "Pure Vegetarian",
    "non-veg": "Non-Vegetarian", "non veg": "Non-Vegetarian", "nonveg": "Non-Vegetarian",
    "mixed": "Mixed / Both", "both": "Mixed / Both", "multi cuisine": "Multi-Cuisine",
    "multicuisine": "Multi-Cuisine", "pan asian": "Pan-Asian",
}

# Partial credit (0-1) when the event type is not an exact match but is similar.
RELATED_EVENTS = {
    "Wedding": {"Engagement": 0.4},
    "Engagement": {"Wedding": 0.4, "Anniversary": 0.3},
    "Anniversary": {"Birthday": 0.4, "Engagement": 0.3},
    "Birthday": {"Anniversary": 0.4, "Other Party": 0.5},
    "Other Party": {"Birthday": 0.5},
    "Corporate": {},
}

# Partial credit (0-1) for cuisine. A venue serving "Mixed / Both" can cover
# vegetarian and non-vegetarian guests, so it is a reasonable (not perfect) match.
CUISINE_PARTIAL = {
    ("Pure Vegetarian", "Mixed / Both"): 0.6, ("Pure Vegetarian", "Multi-Cuisine"): 0.5,
    ("Non-Vegetarian", "Mixed / Both"): 0.6, ("Non-Vegetarian", "Multi-Cuisine"): 0.5,
    ("Mixed / Both", "Pure Vegetarian"): 0.4, ("Mixed / Both", "Non-Vegetarian"): 0.4,
    ("Mixed / Both", "Multi-Cuisine"): 0.7, ("Multi-Cuisine", "Mixed / Both"): 0.7,
    ("Multi-Cuisine", "Continental"): 0.5, ("Multi-Cuisine", "Pan-Asian"): 0.5,
    ("Continental", "Multi-Cuisine"): 0.5, ("Pan-Asian", "Multi-Cuisine"): 0.5,
}

RESULT_COLUMNS = [
    "rank", "venue_id", "venue_name", "city", "venue_type", "event_type", "cuisine",
    "capacity", "price_per_plate", "venue_cost", "estimated_cost", "within_budget",
    "rating", "compatibility_score", "match_reasons",
]


# ─── Input normalisation ──────────────────────────────────────────────────
def _canonical(value: str, valid: list[str], aliases: dict[str, str], label: str) -> str:
    key = str(value).strip().lower()
    for v in valid:
        if v.lower() == key:
            return v
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown {label} '{value}'. Valid options: {valid}")


# ─── Scoring pieces (each returns 0-1) ────────────────────────────────────
def _event_score(wanted: str, actual: str) -> float:
    return 1.0 if wanted == actual else RELATED_EVENTS.get(wanted, {}).get(actual, 0.0)


def _cuisine_score(wanted: str, actual: str) -> float:
    return 1.0 if wanted == actual else CUISINE_PARTIAL.get((wanted, actual), 0.0)


def _budget_score(cost: float, budget: float) -> float:
    if cost <= budget:
        return 1.0
    overshoot = (cost - budget) / budget          # 0.10 = 10% over budget
    return max(0.0, 1.0 - 2.0 * overshoot)        # reaches 0 at 50% over


def _capacity_score(guests: int, capacity: int) -> float:
    fill = guests / capacity                      # how full the venue would be
    return 1.0 if fill >= 0.5 else fill / 0.5     # penalise very empty venues


def _rating_score(rating: float) -> float:
    return min(1.0, max(0.0, (rating - 1) / 4))


def _facility_score(row: pd.Series) -> float:
    return (2 * row["parking_available"] + 2 * row["decoration_available"]
            + 1 * row["catering_available"]) / 5


def _reasons(row: pd.Series, event: str, cuisine: str, guests: int,
             budget: float, cost: float) -> list[str]:
    r = []
    # Event match
    if row["event_type"] == event:
        r.append(f"Dedicated {event} venue")
    elif _event_score(event, row["event_type"]) > 0:
        r.append(f"Listed for {row['event_type']} (verified compatible with {event})")
    else:
        r.append(f"Listed for {row['event_type']}")

    # Cuisine match
    if row["cuisine"] == cuisine:
        r.append(f"Serves requested {cuisine} cuisine")
    elif _cuisine_score(cuisine, row["cuisine"]) > 0:
        r.append(f"Serves {row['cuisine']} (suitable for {cuisine} guests)")
    else:
        r.append(f"Serves {row['cuisine']}")

    # Capacity
    fill_pct = int(round((guests / row["capacity"]) * 100))
    r.append(f"Capacity: {row['capacity']} guests ({guests} guests is {fill_pct}% fill)")

    # Budget
    if cost <= budget:
        remaining = budget - cost
        r.append(f"Estimated ₹{cost:,.0f} fits within ₹{budget:,.0f} budget (₹{remaining:,.0f} buffer)")
    else:
        overshoot = cost - budget
        r.append(f"Estimated ₹{cost:,.0f} exceeds budget by ₹{overshoot:,.0f}")

    # Rating & amenities
    facilities = []
    if row.get("parking_available", 0) == 1:
        facilities.append("parking")
    if row.get("catering_available", 0) == 1:
        facilities.append("in-house catering")
    if row.get("decoration_available", 0) == 1:
        facilities.append("decoration service")

    fac_str = f" with {', '.join(facilities)}" if facilities else ""
    r.append(f"Rated {row['rating']}/5.0{fac_str}")
    return r


# ─── Public API ───────────────────────────────────────────────────────────
def recommend(
    event_type: str,
    city: str,
    guests: int,
    budget: float,
    cuisine: str,
    top_n: int = 5,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return up to `top_n` venues ranked by compatibility (best first).

    `df` is an already-preprocessed venue table; if omitted the default
    dataset is loaded. Returns an empty DataFrame (with the result columns)
    if no venue matches the hard filters.
    """
    if guests <= 0:
        raise ValueError("guests must be a positive number")
    if budget <= 0:
        raise ValueError("budget must be a positive number")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    data = preprocess() if df is None else df
    event = _canonical(event_type, sorted(data["event_type"].unique()), EVENT_ALIASES, "event type")
    cuis = _canonical(cuisine, sorted(data["cuisine"].unique()), CUISINE_ALIASES, "cuisine")

    # 1) Hard filters:
    # - City match
    # - Capacity >= guests
    # - Selected event type supported
    pool = data[
        (data["city"].str.lower() == str(city).strip().lower())
        & (data["capacity"] >= guests)
        & (data["event_type"].map(lambda v: _event_score(event, v) > 0))
    ].copy()

    if pool.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    # 2) Compatibility score (out of 100)
    pool["estimated_cost"] = pool["venue_cost"] + guests * pool["price_per_plate"]
    parts = pd.DataFrame({
        "event": pool["event_type"].map(lambda v: _event_score(event, v)),
        "cuisine": pool["cuisine"].map(lambda v: _cuisine_score(cuis, v)),
        "budget": pool["estimated_cost"].map(lambda c: _budget_score(c, budget)),
        "capacity": pool["capacity"].map(lambda c: _capacity_score(guests, c)),
        "rating": pool["rating"].map(_rating_score),
        "facilities": pool.apply(_facility_score, axis=1),
    }, index=pool.index)
    pool["compatibility_score"] = sum(parts[k] * w for k, w in WEIGHTS.items()).round(1)
    pool["within_budget"] = pool["estimated_cost"] <= budget
    pool["match_reasons"] = pool.apply(
        lambda r: _reasons(r, event, cuis, guests, budget, r["estimated_cost"]), axis=1)

    # 3) Top N (sorted by compatibility_score desc, rating desc, estimated_cost asc)
    top = (pool.sort_values(["compatibility_score", "rating", "estimated_cost"],
                            ascending=[False, False, True])
               .head(top_n).reset_index(drop=True))
    top.insert(0, "rank", top.index + 1)
    top["estimated_cost"] = top["estimated_cost"].astype(int)
    return top[RESULT_COLUMNS]


# ─── Demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = dict(event_type="Wedding", city="Jaipur", guests=150, budget=200000, cuisine="Vegetarian")
    print("Query:", query, "\n")
    results = recommend(**query)
    if results.empty:
        print("No venues found.")
    for _, r in results.iterrows():
        print(f"#{r['rank']}  {r['venue_name']}  ({r['venue_type']}, {r['event_type']})  "
              f"score {r['compatibility_score']}/100")
        print(f"    estimated INR {r['estimated_cost']:,}  |  {r['cuisine']}  |  rating {r['rating']}")
        for reason in r["match_reasons"]:
            print("    -", reason)
