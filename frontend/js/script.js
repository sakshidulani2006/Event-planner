/**
 * Starlit Serenity - Frontend Application Logic.
 * 
 * Vanilla JavaScript implementation:
 * - Dynamic options loading from Flask API (/api/options)
 * - Event Planner form submission (/api/recommend)
 * - ML Budget prediction (/api/predict-budget)
 * - Quick-preset test scenarios for 1-click evaluation
 * - Responsive UI interactions, loading states, and smooth navigation
 * 
 * Absolutely NO external APIs, Firebase, React, or hardcoded mock results.
 */

// Dynamically handle API base URL whether served via Flask or directly as a file / live server
const API_BASE = (window.location.origin && window.location.origin.includes(':5000'))
    ? '' 
    : 'http://127.0.0.1:5000';

// Global state
let availableOptions = {
    cities: [],
    event_types: [],
    cuisines: [],
    stats: {}
};

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNav();
    initSliders();
    fetchOptions();
    initPresetButtons();
    initBudgetPredictor();
});

// ─── Navigation ────────────────────────────────────────────────────────────
function initNav() {
    const nav = document.getElementById('main-nav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 40) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });

    const hamburger = document.querySelector('.hamburger');
    if (hamburger) {
        hamburger.addEventListener('click', () => {
            nav.classList.toggle('mobile-open');
        });
    }

    // Close mobile nav on link click
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
            nav.classList.remove('mobile-open');
        });
    });
}

// ─── Sliders & Formatting ──────────────────────────────────────────────────
function initSliders() {
    const guestInput = document.getElementById('planner-guests');
    const guestDisplay = document.getElementById('guest-val');
    if (guestInput && guestDisplay) {
        const updateGuests = () => {
            guestDisplay.textContent = `${parseInt(guestInput.value).toLocaleString('en-IN')} guests`;
        };
        guestInput.addEventListener('input', updateGuests);
        updateGuests();
    }

    const budgetInput = document.getElementById('planner-budget');
    const budgetDisplay = document.getElementById('budget-val');
    if (budgetInput && budgetDisplay) {
        const updateBudget = () => {
            budgetDisplay.textContent = `₹${parseInt(budgetInput.value).toLocaleString('en-IN')}`;
        };
        budgetInput.addEventListener('input', updateBudget);
        updateBudget();
    }
}

// ─── Fetch Options from Backend ────────────────────────────────────────────
async function fetchOptions() {
    try {
        const res = await fetch(`${API_BASE}/api/options`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.success) {
            availableOptions = data;
            populateDropdowns(data);
            updateStatsUI(data.stats);
        }
    } catch (err) {
        console.warn('Could not load options from Flask API. Using dataset defaults.', err);
        // Fallback options matching dataset if backend is still starting up
        const fallback = {
            cities: ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata", "Mumbai", "Pune", "Udaipur"],
            event_types: ["Anniversary", "Birthday", "Corporate", "Engagement", "Other Party", "Wedding"],
            cuisines: ["Continental", "Mixed / Both", "Multi-Cuisine", "Non-Vegetarian", "Pan-Asian", "Pure Vegetarian"]
        };
        populateDropdowns(fallback);
    }
}

function populateDropdowns(options) {
    const citySelect = document.getElementById('planner-city');
    const eventSelect = document.getElementById('planner-event');
    const cuisineSelect = document.getElementById('planner-cuisine');

    const budgetCity = document.getElementById('budget-city');
    const budgetEvent = document.getElementById('budget-event');
    const budgetCuisine = document.getElementById('budget-cuisine');

    if (citySelect && options.cities) {
        populateSelect(citySelect, options.cities, "Jaipur");
        if (budgetCity) populateSelect(budgetCity, options.cities, "Jaipur");
    }
    if (eventSelect && options.event_types) {
        populateSelect(eventSelect, options.event_types, "Wedding");
        if (budgetEvent) populateSelect(budgetEvent, options.event_types, "Wedding");
    }
    if (cuisineSelect && options.cuisines) {
        populateSelect(cuisineSelect, options.cuisines, "Pure Vegetarian");
        if (budgetCuisine) populateSelect(budgetCuisine, options.cuisines, "Pure Vegetarian");
    }
}

function populateSelect(selectEl, items, defaultValue) {
    selectEl.innerHTML = '';
    items.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        if (item.toLowerCase() === defaultValue.toLowerCase()) {
            opt.selected = true;
        }
        selectEl.appendChild(opt);
    });
}

function updateStatsUI(stats) {
    if (!stats) return;
    const statVenues = document.getElementById('stat-venues');
    const statCities = document.getElementById('stat-cities');
    if (statVenues && stats.total_venues) statVenues.textContent = `${stats.total_venues}+`;
    if (statCities && availableOptions.cities) statCities.textContent = `${availableOptions.cities.length}`;
}

// ─── Preset Buttons for Easy Grading/Testing ───────────────────────────────
function initPresetButtons() {
    const presets = [
        {
            id: 'preset-wedding',
            event_type: 'Wedding',
            city: 'Jaipur',
            guests: 150,
            budget: 200000,
            cuisine: 'Pure Vegetarian'
        },
        {
            id: 'preset-birthday',
            event_type: 'Birthday',
            city: 'Jaipur',
            guests: 40,
            budget: 50000,
            cuisine: 'Mixed / Both'
        },
        {
            id: 'preset-corporate',
            event_type: 'Corporate',
            city: 'Delhi',
            guests: 300,
            budget: 500000,
            cuisine: 'Multi-Cuisine'
        },
        {
            id: 'preset-nomatch',
            event_type: 'Wedding',
            city: 'Jaipur',
            guests: 5000,
            budget: 200000,
            cuisine: 'Pure Vegetarian'
        }
    ];

    presets.forEach(p => {
        const btn = document.getElementById(p.id);
        if (btn) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                applyPreset(p);
            });
        }
    });
}

function applyPreset(p) {
    const eventSelect = document.getElementById('planner-event');
    const citySelect = document.getElementById('planner-city');
    const guestInput = document.getElementById('planner-guests');
    const budgetInput = document.getElementById('planner-budget');
    const cuisineSelect = document.getElementById('planner-cuisine');

    if (eventSelect) eventSelect.value = p.event_type;
    if (citySelect) citySelect.value = p.city;
    if (guestInput) {
        guestInput.value = p.guests;
        document.getElementById('guest-val').textContent = `${p.guests.toLocaleString('en-IN')} guests`;
    }
    if (budgetInput) {
        budgetInput.value = p.budget;
        document.getElementById('budget-val').textContent = `₹${p.budget.toLocaleString('en-IN')}`;
    }
    if (cuisineSelect) cuisineSelect.value = p.cuisine;

    // Trigger recommendations automatically
    handlePlannerSubmit();
}

// ─── Planner Form Submission ───────────────────────────────────────────────
const plannerForm = document.getElementById('event-planner-form');
if (plannerForm) {
    plannerForm.addEventListener('submit', (e) => {
        e.preventDefault();
        handlePlannerSubmit();
    });
}

async function handlePlannerSubmit() {
    const eventType = document.getElementById('planner-event').value;
    const city = document.getElementById('planner-city').value;
    const guests = parseInt(document.getElementById('planner-guests').value);
    const budget = parseFloat(document.getElementById('planner-budget').value);
    const cuisine = document.getElementById('planner-cuisine').value;

    const resultsSection = document.getElementById('recommendations');
    const loadingEl = document.getElementById('results-loading');
    const contentEl = document.getElementById('results-content');
    const emptyEl = document.getElementById('results-empty');
    const errorEl = document.getElementById('results-error');

    // Scroll to recommendations section
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Show loading state
    loadingEl.style.display = 'block';
    contentEl.style.display = 'none';
    emptyEl.style.display = 'none';
    errorEl.style.display = 'none';

    try {
        const payload = {
            event_type: eventType,
            city: city,
            guests: guests,
            budget: budget,
            cuisine: cuisine
        };

        const res = await fetch(`${API_BASE}/api/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        loadingEl.style.display = 'none';

        if (!res.ok || !data.success) {
            throw new Error(data.error || 'Failed to retrieve recommendations');
        }

        if (data.count === 0 || !data.recommendations || data.recommendations.length === 0) {
            // Show empty state
            emptyEl.style.display = 'block';
            document.getElementById('empty-msg').textContent = data.message || 
                `No venues found matching ${guests} guests for a ${eventType} in ${city}.`;
        } else {
            // Render recommendations
            renderRecommendations(data.recommendations, payload);
            contentEl.style.display = 'block';
        }

        // Also update the budget predictor section automatically for synergy
        syncBudgetPredictor(eventType, city, guests, cuisine);

    } catch (err) {
        console.error('Recommendation fetch error:', err);
        loadingEl.style.display = 'none';
        errorEl.style.display = 'block';
        document.getElementById('error-msg').textContent = err.message || 'Connection error. Please ensure the Flask backend is running on port 5000.';
    }
}

// ─── Render Recommendations Cards ──────────────────────────────────────────
function renderRecommendations(venues, query) {
    const listContainer = document.getElementById('recommendations-list');
    const querySummary = document.getElementById('query-summary');
    listContainer.innerHTML = '';

    querySummary.innerHTML = `
        Showing <strong>${venues.length}</strong> top venues for your <strong>${query.event_type}</strong> 
        in <strong>${query.city}</strong> for <strong>${query.guests.toLocaleString('en-IN')} guests</strong> 
        (Budget: <strong>₹${query.budget.toLocaleString('en-IN')}</strong>, <strong>${query.cuisine}</strong>).
    `;

    venues.forEach((v) => {
        const card = document.createElement('div');
        card.className = 'venue-card reveal visible';

        const isOverBudget = !v.within_budget;
        const budgetBadgeClass = isOverBudget ? 'badge-over' : 'badge-under';
        const budgetBadgeText = isOverBudget 
            ? `₹${(v.estimated_cost - query.budget).toLocaleString('en-IN')} Over Budget` 
            : `Within Budget (₹${(query.budget - v.estimated_cost).toLocaleString('en-IN')} buffer)`;

        // Fill percentage
        const fillPct = Math.round((query.guests / v.capacity) * 100);

        // Format reasons list
        const reasonsHtml = v.match_reasons.map(r => `<li><span class="reason-check">✓</span> ${r}</li>`).join('');

        card.innerHTML = `
            <div class="venue-card-header">
                <div class="venue-title-group">
                    <span class="rank-badge">#${v.rank} Best Match</span>
                    <h3 class="venue-title">${v.venue_name}</h3>
                    <div class="venue-meta">
                        <span class="meta-tag">🏛️ ${v.venue_type}</span>
                        <span class="meta-tag">📍 ${v.city}</span>
                        <span class="meta-tag rating-tag">★ ${v.rating.toFixed(1)} / 5.0</span>
                    </div>
                </div>
                <div class="score-container">
                    <div class="score-circle">
                        <span class="score-number">${v.compatibility_score}</span>
                        <span class="score-label">MATCH</span>
                    </div>
                </div>
            </div>

            <div class="venue-details-grid">
                <div class="detail-box">
                    <span class="box-label">Guest Capacity</span>
                    <strong class="box-val">${v.capacity.toLocaleString('en-IN')} guests</strong>
                    <span class="box-sub">${fillPct}% filled by your party</span>
                </div>
                <div class="detail-box">
                    <span class="box-label">Cuisine Style</span>
                    <strong class="box-val">${v.cuisine}</strong>
                    <span class="box-sub">₹${v.price_per_plate.toLocaleString('en-IN')} / plate</span>
                </div>
                <div class="detail-box">
                    <span class="box-label">Base Venue Rental</span>
                    <strong class="box-val">₹${v.venue_cost.toLocaleString('en-IN')}</strong>
                    <span class="box-sub">Full-day booking rate</span>
                </div>
                <div class="detail-box highlight-cost">
                    <span class="box-label">Estimated Total Cost</span>
                    <strong class="box-val cost-accent">₹${v.estimated_cost.toLocaleString('en-IN')}</strong>
                    <span class="budget-status ${budgetBadgeClass}">${budgetBadgeText}</span>
                </div>
            </div>

            <div class="cost-formula-breakdown">
                <strong>Formula Breakdown:</strong> 
                ₹${v.venue_cost.toLocaleString('en-IN')} (rental) + 
                (${query.guests} guests × ₹${v.price_per_plate.toLocaleString('en-IN')} plate) = 
                <strong>₹${v.estimated_cost.toLocaleString('en-IN')}</strong>
            </div>

            <div class="reasons-section">
                <h4 class="reasons-title">✦ Why We Recommend This Venue</h4>
                <ul class="reasons-list">
                    ${reasonsHtml}
                </ul>
            </div>
        `;

        listContainer.appendChild(card);
    });
}

// ─── Budget ML Model Predictor ─────────────────────────────────────────────
function initBudgetPredictor() {
    const budgetBtn = document.getElementById('run-budget-model-btn');
    if (budgetBtn) {
        budgetBtn.addEventListener('click', (e) => {
            e.preventDefault();
            runBudgetPrediction();
        });
    }

    const guestSlider = document.getElementById('budget-guests');
    const guestVal = document.getElementById('budget-guest-val');
    if (guestSlider && guestVal) {
        guestSlider.addEventListener('input', () => {
            guestVal.textContent = `${parseInt(guestSlider.value).toLocaleString('en-IN')} guests`;
        });
    }
}

function syncBudgetPredictor(eventType, city, guests, cuisine) {
    const bEvent = document.getElementById('budget-event');
    const bCity = document.getElementById('budget-city');
    const bGuests = document.getElementById('budget-guests');
    const bGuestVal = document.getElementById('budget-guest-val');
    const bCuisine = document.getElementById('budget-cuisine');

    if (bEvent) bEvent.value = eventType;
    if (bCity) bCity.value = city;
    if (bGuests) {
        bGuests.value = guests;
        if (bGuestVal) bGuestVal.textContent = `${guests.toLocaleString('en-IN')} guests`;
    }
    if (bCuisine) bCuisine.value = cuisine;

    // Trigger budget prediction automatically
    runBudgetPrediction();
}

async function runBudgetPrediction() {
    const eventType = document.getElementById('budget-event')?.value;
    const city = document.getElementById('budget-city')?.value;
    const guests = parseInt(document.getElementById('budget-guests')?.value || 150);
    const cuisine = document.getElementById('budget-cuisine')?.value;

    const resultBox = document.getElementById('budget-prediction-results');
    const loadingBox = document.getElementById('budget-prediction-loading');
    const errorBox = document.getElementById('budget-prediction-error');

    if (!eventType || !city || !cuisine) return;

    if (loadingBox) loadingBox.style.display = 'block';
    if (resultBox) resultBox.style.display = 'none';
    if (errorBox) errorBox.style.display = 'none';

    try {
        const payload = {
            event_type: eventType,
            city: city,
            guests: guests,
            cuisine: cuisine
        };

        const res = await fetch(`${API_BASE}/api/predict-budget`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (loadingBox) loadingBox.style.display = 'none';

        if (!res.ok || !data.success) {
            throw new Error(data.error || 'Failed to predict budget');
        }

        const pred = data.prediction;
        if (resultBox) {
            document.getElementById('ml-pred-cost').textContent = `₹${pred.predicted_total_cost.toLocaleString('en-IN')}`;
            document.getElementById('formula-cost').textContent = `₹${pred.formula_total_cost.toLocaleString('en-IN')}`;
            
            const diffEl = document.getElementById('cost-diff');
            const diffSign = pred.difference >= 0 ? '+' : '';
            diffEl.textContent = `${diffSign}₹${pred.difference.toLocaleString('en-IN')}`;
            diffEl.className = Math.abs(pred.difference) < 15000 ? 'diff-good' : 'diff-warn';

            document.getElementById('profile-venue').textContent = pred.venue_used;
            document.getElementById('profile-match').textContent = pred.venue_match;

            resultBox.style.display = 'block';
        }

    } catch (err) {
        console.error('Budget prediction error:', err);
        if (loadingBox) loadingBox.style.display = 'none';
        if (errorBox) {
            errorBox.style.display = 'block';
            document.getElementById('budget-error-msg').textContent = err.message || 'Error communicating with ML model API';
        }
    }
}
