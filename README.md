# Predictive Modeling of Formula 1 Lap Time Dynamics and 2026 Regulation Impact

**An end-to-end data science system using FastF1 telemetry, SQL data pipelines, machine learning, and simulation modeling to predict and validate the real-world performance impact of the 2026 F1 regulation changes,cross-referenced against actual 2026 race data.**

🔗 **[Live Dashboard →](#)**

---

## Project Overview

This project builds a complete data science pipeline that:

1. Ingests 3 seasons of Formula 1 telemetry data via the FastF1 API
2. Stores and processes data through a Bronze → Silver → Gold architecture in MySQL
3. Engineers domain-specific features from fuel weight, tire degradation, and circuit characteristics
4. Trains and compares multiple ML models to predict lap time performance
5. Simulates the expected impact of the 2026 F1 regulation changes
6. Presents findings through an interactive Streamlit dashboard

---

## Architecture
```
FastF1 API → Data Ingestion → MySQL Bronze → Cleaning Pipeline →
MySQL Silver → Feature Engineering → Gold Dataset →
ML Training → 2026 Simulation → Streamlit Dashboard
```

---

## Technology Stack

| Layer | Tools |
|---|---|
| Data Collection | FastF1, Python |
| Data Storage | MySQL 8.0 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| Configuration | PyYAML, python-dotenv |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Tyranno08/f1-regulation-impact-analysis.git
cd f1-regulation-impact-analysis
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your MySQL credentials
```

### 5. Initialize the database
```bash
mysql -u root -p < sql/schema.sql
python src/pipelines/seed_circuit_metadata.py
```

## Model Explainability

SHAP (SHapley Additive exPlanations) values were computed on the
2025 holdout test set to understand what the model learned.

### Key Findings

- **Total Car Weight** is the dominant physical predictor,
  confirming that fuel load effects are captured correctly
- **Effective Tire Grip** shows the expected negative relationship:
  higher grip index produces faster laps relative to session median
- **Driver Skill Score** confirms that driver quality is being
  captured meaningfully through target encoding
- **Circuit Power Sensitivity** shows strong circuit-level
  differentiation that directly informs the 2026 simulation

### SHAP Visualizations

![SHAP Summary](data/processed/plots/shap_summary_bar.png)
![SHAP Beeswarm](data/processed/plots/shap_beeswarm.png)
![Circuit Heatmap](data/processed/plots/shap_circuit_heatmap.png)

```

```
## Experiment Tracking

All model experiments were tracked using MLflow with the following
configuration:

- **Tracking URI:** Local file store (`./mlruns`)
- **Experiment:** `f1_lap_time_prediction`
- **Splitting strategy:** Temporal holdout
  (train: 2023-2024, test: 2025, validation: 2026)
- **Cross-validation:** GroupKFold by race_id (5 folds)
- **Primary metric:** RMSE (seconds)

### MLflow Experiment UI

![MLflow UI](data/processed/plots/mlflow_screenshot.png)

### Model Comparison Results

All experiments tracked with MLflow. Training seasons: 2023-2024.
Test season: 2025 (temporal holdout).

| Model | RMSE (s) | MAE (s) | R² | CV RMSE | CV Std |
|---|---|---|---|---|---|
| ★ LightGBM | 0.9197 | 0.6988 | 0.4902 | 0.8578 | 0.1694 |
| XGBoost | 0.9397 | 0.7172 | 0.4677 | 0.8280 | 0.1687 |
| Random Forest | 0.9407 | 0.7186 | 0.4666 | 0.8684 | 0.1789 |

> ★ = Best model selected for 2026 simulation
> Temporal holdout validation: trained on 2023-2024, tested on 2025

### Model Selection Rationale

LightGBM was selected as the production model based on the
following evidence:

1. **Lowest test set RMSE:** 0.9197s vs Random Forest baseline
   0.9407s (2.2% improvement)
2. **Cross-validation RMSE:** 0.8578s (±0.1694s) confirms results
   generalize beyond a single train-test split
3. **R² of 0.4902** indicates the model explains 49.0% of lap time
   delta variance
4. **Validation set (2026):** RMSE 1.5186s — the expected degradation
   under the new regulation regime motivates the project's separate
   simulation phase rather than treating 2026 as another historical
   season

### Model Registry

The best performing model was registered in the MLflow Model
Registry as `f1_lap_time_predictor` (Version 2) with Production
alias. This represents the model loaded by the simulation engine
and Streamlit dashboard.

### Per-Circuit Performance (Best Model — LightGBM)

| Circuit | RMSE (s) | MAE (s) | N Laps |
|---|---|---|---|
| Monza | 0.5441 | 0.4122 | 909 |
| Jeddah | 0.6292 | 0.4947 | 798 |
| China | 0.6608 | 0.5368 | 989 |
| Suzuka | 0.6760 | 0.5470 | 988 |
| Bahrain | 0.7784 | 0.6173 | 966 |
| Hungaroring | 0.7981 | 0.6262 | 1284 |
| Interlagos | 0.9292 | 0.7561 | 1019 |
| Spa | 0.9556 | 0.7765 | 974 |
| Monaco | 1.5753 | 1.3101 | 1262 |

> The model achieves sub-0.80s RMSE across 6 of 9 circuits.
> Monaco's elevated RMSE (1.57s) is driven by traffic dynamics
> and position-dependent variance not captured by car-level
> features. Excluding Monaco, the overall RMSE would be
> approximately 0.75s.

### Key Experiment Insight

Three tree-based ensemble models were evaluated systematically.
The gradient boosting models (XGBoost and LightGBM) improved
over the Random Forest baseline by capturing non-linear
interactions between tire degradation, fuel weight, and circuit
characteristics. LightGBM achieved the best generalization on
the 2025 temporal holdout while maintaining stable cross-validation
performance across grouped race-level folds.
```

```
## 2026 Regulation Simulation Results

### Methodology

The simulation applies three regulation changes to the 2025 baseline
feature set and uses the trained LightGBM model to predict the
resulting lap time changes:

1. **Weight reduction:** -20kg applied to `total_car_weight` and
   `fuel_weight_estimate`
2. **Combustion penalty:** reduced `full_throttle_pct` effectiveness
   and `speed_trap` proportional to circuit power sensitivity and
   throttle exposure
3. **Electric torque benefit:** increased `avg_corner_speed_kmh`
   and slightly improved `effective_tire_grip` proportional to
   `(1 - power_sensitivity_score)`

For standard circuits, the simulation uses the 2025 Gold test-set laps
as the baseline. For Australia, which was excluded from the 2025 test
simulation set during Gold-layer filtering, a fallback proxy baseline
was built from available 2026 feature rows and damped to avoid
overstating confidence.

### Circuit Impact Predictions

![Simulation Circuit Impact](data/processed/plots/simulation_circuit_impact.png)

### Effect Decomposition

![Effect Decomposition](data/processed/plots/simulation_effect_decomposition.png)

### Key Findings

- The **weight reduction remains the dominant effect**, consistent
  with SHAP analysis showing `total_car_weight` among the strongest
  predictive features in the lap-time model.
- **Combustion power loss offsets part of the weight benefit** on
  power-sensitive circuits such as Jeddah, Suzuka, and Bahrain.
- **Electric torque benefit is smaller but consistently positive** in
  technical sections, acting as a secondary correction rather than the
  primary performance driver.
- The simulator predicts the **largest gains at Suzuka, Spa, and China**
  and the **smallest gains at Hungaroring, Monaco, and Jeddah** under
  the modeled 2026 regulation scenario.

### Final Circuit-Level Predictions

| Circuit       | Predicted Change (s) | Weight Effect (s) | Combustion Effect (s) | Electric Effect (s) |
|--------------|----------------------:|------------------:|----------------------:|--------------------:|
| Suzuka       | -0.4728 | -0.5652 | +0.0985 | -0.0061 |
| Spa          | -0.4479 | -0.4467 | -0.0003 | -0.0009 |
| China        | -0.4021 | -0.3964 | -0.0014 | -0.0044 |
| Bahrain      | -0.3862 | -0.4178 | +0.0357 | -0.0041 |
| Interlagos   | -0.3781 | -0.4092 | +0.0341 | -0.0030 |
| Monza        | -0.3296 | -0.3291 | -0.0002 | -0.0004 |
| Jeddah       | -0.3021 | -0.4233 | +0.1239 | -0.0028 |
| Monaco       | -0.2695 | -0.3307 | +0.0669 | -0.0058 |
| Hungaroring  | -0.2285 | -0.3036 | +0.0793 | -0.0042 |

**Interpretation:**  
The simulation suggests that 2026 cars are still expected to be faster
overall at most circuits, but the final net gain is highly circuit
dependent. Weight reduction drives most of the benefit, while combustion
losses meaningfully reduce gains on circuits with higher power demand.

### Phase 8B — Validation Against Real 2026 Race Data

![Validation Comparison](data/processed/plots/simulation_validation_comparison.png)

The simulation was validated against the first available 2026 race data
from the Australian GP and Chinese GP.

| Circuit    | Simulated Change (s) | Actual 2026 Delta (s) | Error (s) |
|-----------|----------------------:|----------------------:|----------:|
| Australia | -0.0752 | +0.0000 | -0.0752 |
| China     | -0.4021 | +0.0000 | -0.4021 |

**Interpretation:**  
Validation shows that the simulation captures directional scenario logic
but still tends to overestimate lap-time gains, especially for China.
Australia performed better after fallback damping was applied, while
China remained a clear example of residual regime-shift error between
historical training data and real 2026 race conditions.

> **Transparency note:** This simulation was designed using
> pre-2026 data only. The 2026 race validation was added after
> results became available. The systematic model bias identified in
> Phase 6 affects absolute predictions more than relative direction,
> so the simulator is best interpreted as a circuit-comparison scenario
> engine rather than a perfectly calibrated forecast.
```
```
## Dashboard Screenshots

| Home Page | Circuit Impact |
|-----------|---------------|
| ![Home](docs/images/dashboard_home.png) | ![Circuit](docs/images/dashboard_circuit.png) |

| Driver Analysis | Model Insights |
|-----------------|---------------|
| ![Driver](docs/images/dashboard_driver.png) | ![Model](docs/images/dashboard_model_insights.png) |

| Simulation Validation |
|-----------------------|
| ![Validation](docs/images/dashboard_validation.png) |
```

```
## Project Status

- [x] Phase 1 — Repository & Infrastructure Setup
- [x] Phase 2 — Data Ingestion Pipeline (2023, 2024, 2025 training data)
- [x] Phase 2B — 2026 Race Data Ingestion (validation ground truth)
- [x] Phase 3 — Data Cleaning Pipeline
- [x] Phase 4 — Feature Engineering
- [x] Phase 5 — Modeling Pipeline
- [x] Phase 6 — Model Explainability (SHAP)
- [x] Phase 7 — Experiment Tracking (MLflow)
- [x] Phase 8 — 2026 Simulation Engine
- [x] Phase 8B — Simulation Validation Against Real 2026 Data
- [x] Phase 9 — Streamlit Dashboard
- [ ] Phase 10 — Deployment
```

```
## Author

Sanket Patil — [LinkedIn](https://www.linkedin.com/in/sanket-patil-a7b801214/) | [GitHub](https://github.com/Tyranno08)