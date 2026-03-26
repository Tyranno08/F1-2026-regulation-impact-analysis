# src/models/shap_explainability.py

"""
SHAP Model Explainability Pipeline

Loads the best trained model from Phase 5, computes SHAP values
on the 2025 test set, generates a comprehensive set of
explainability visualizations, and saves a SHAP summary JSON
for use in the Streamlit dashboard.

SHAP (SHapley Additive exPlanations) provides mathematically
rigorous, model-agnostic feature attributions rooted in
cooperative game theory. Each SHAP value represents the
contribution of one feature to one prediction.

Plots generated:
    1. Summary bar plot      — mean absolute feature importance
    2. Beeswarm plot         — full SHAP value distribution
    3. Dependence plots      — per-feature SHAP curves
    4. Waterfall plot        — single prediction explanation
    5. Circuit heatmap       — average SHAP values per circuit

Outputs:
    data/processed/plots/shap_*.png  — all visualization files
    models/shap_summary.json         — feature importance summary
                                       for dashboard use

Usage:
    python src/models/shap_explainability.py

    Optional flags:
    --max-samples 2000    Limit SHAP computation sample size
    --circuit Monza       Generate circuit-specific waterfall plot
"""

import sys
import os
import argparse
import json
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from db_connection import get_sqlalchemy_engine
from config_loader import load_config
from logger import get_logger

logger = get_logger("shap_explainability")


# ============================================================
# Constants and Paths
# ============================================================

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )),
    "models"
)

PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )),
    "data", "processed", "plots"
)

# F1 red color for consistent plot styling
F1_RED = "#E8002D"
F1_DARK = "#15151E"
F1_SILVER = "#C0C0C0"

# Human-readable feature labels for plots
FEATURE_LABELS = {
    # ... existing entries stay ...
    "total_car_weight":                 "Total Car Weight (kg)",
    "fuel_weight_estimate":             "Fuel Weight Estimate (kg)",
    "effective_tire_grip":              "Effective Tire Grip Index",
    "tire_life":                        "Tire Age (laps)",
    "compound":                         "Tire Compound",
    "sector1_ratio":                    "Sector 1 Time Ratio",
    "sector2_ratio":                    "Sector 2 Time Ratio",
    "sector3_ratio":                    "Sector 3 Time Ratio",
    "power_sensitivity_score":          "Circuit Power Sensitivity",
    "full_throttle_pct":               "Full Throttle Percentage",
    "avg_corner_speed_kmh":            "Avg Corner Speed (km/h)",
    "elevation_change_m":              "Elevation Change (m)",
    "num_corners":                      "Number of Corners",
    "track_temp":                       "Track Temperature (°C)",
    "air_temp":                         "Air Temperature (°C)",
    "humidity":                         "Humidity (%)",
    "driver_skill_score":              "Driver Skill Score",
    "speed_trap":                       "Speed Trap (km/h)",
    "circuit":                          "Circuit",
    "lap_number":                       "Lap Number",
    "grip_temp_interaction":           "Grip × Temperature",
    # One-hot encoded compound columns
    "SOFT":                             "Compound: Soft",
    "MEDIUM":                           "Compound: Medium",
    "HARD":                             "Compound: Hard",
    "INTERMEDIATE":                     "Compound: Intermediate",
    "WET":                              "Compound: Wet",
    "UNKNOWN":                          "Compound: Unknown",
    # One-hot encoded circuit columns
    "Monaco":                           "Circuit: Monaco",
    "Hungaroring":                      "Circuit: Hungaroring",
    "Australia":                        "Circuit: Australia",
    "China":                            "Circuit: China",
    "Bahrain":                          "Circuit: Bahrain",
    "Suzuka":                           "Circuit: Suzuka",
    "Interlagos":                       "Circuit: Interlagos",
    "Silverstone":                      "Circuit: Silverstone",
    "Spa":                              "Circuit: Spa",
    "Jeddah":                           "Circuit: Jeddah",
    "Monza":                            "Circuit: Monza",
}


# ============================================================
# Data Loading
# ============================================================

def load_model_and_metadata(
    model_name: str = "best_model"
) -> tuple:
    """
    Loads the trained model pipeline and its metadata JSON.

    Args:
        model_name: Model filename without extension.

    Returns:
        Tuple of (pipeline, metadata_dict)

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    meta_path = os.path.join(
        MODELS_DIR, f"{model_name}_metadata.json"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Run Phase 5 training first."
        )

    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded: {model_path}")

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return pipeline, metadata


def load_test_data(
    engine,
    feature_columns: list,
    target_column: str,
    max_samples: int = None
) -> tuple:
    """
    Loads the 2025 test set from the Gold table for SHAP computation.

    SHAP computation is expensive, particularly for tree ensembles.
    We optionally subsample to keep computation time reasonable
    while maintaining statistical representativeness.

    Subsampling strategy:
        We sample proportionally across circuits so every circuit
        remains represented in the SHAP analysis.

    Args:
        engine:          SQLAlchemy engine
        feature_columns: Feature column names
        target_column:   Target column name
        max_samples:     Maximum rows to use for SHAP computation.
                         None means use all available test rows.

    Returns:
        Tuple of (X_test, y_test, meta_df)
    """
    logger.info("Loading 2025 test set for SHAP computation...")

    # Avoid duplicate columns in SQL select
    already_selected = {"race_id", "season", "circuit", "driver", "lap_number"}
    feature_cols_for_sql = [
        f for f in feature_columns if f not in already_selected
    ]

    query = f"""
        SELECT
            race_id,
            season,
            circuit,
            driver,
            lap_number,
            {target_column},
            {", ".join(feature_cols_for_sql)}
        FROM gold_modeling_data
        WHERE data_split = 'test'
          AND {target_column} IS NOT NULL
        ORDER BY circuit, driver, lap_number
    """

    df = pd.read_sql(query, engine)

    # Safety net in case duplicates still slip through
    df = df.loc[:, ~df.columns.duplicated()]

    logger.info(
        f"Loaded {len(df)} test rows from {df['circuit'].nunique()} "
        f"circuits and {df['driver'].nunique()} drivers"
    )

    # Fill any remaining nulls
    for col in feature_columns:
        if col not in df.columns:
            logger.error(f"Missing feature column in SHAP data: {col}")
            raise KeyError(f"Missing feature column: {col}")

        if col in ("compound", "circuit"):
            df[col] = df[col].fillna("UNKNOWN")
        else:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                median_val = df[col].median()
                logger.warning(
                    f"Imputing {null_count} nulls in SHAP column "
                    f"{col} with median {median_val:.4f}"
                )
                df[col] = df[col].fillna(median_val)

    # Stratified subsample if max_samples specified
    if max_samples and len(df) > max_samples:
        logger.info(
            f"Subsampling {max_samples} rows from {len(df)} "
            f"using circuit-stratified sampling"
        )

        circuits = df["circuit"].dropna().unique()
        samples_per_circuit = max(1, max_samples // len(circuits))

        sampled = []
        for circuit in circuits:
            circuit_df = df[df["circuit"] == circuit]
            n = min(samples_per_circuit, len(circuit_df))
            sampled.append(
                circuit_df.sample(n=n, random_state=42)
            )

        df = pd.concat(sampled, ignore_index=True)
        logger.info(f"Subsampled to {len(df)} rows")

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    meta = df[["race_id", "circuit", "driver", "lap_number", "season"]].copy()

    return X, y, meta


# ============================================================
# SHAP Computation
# ============================================================

def compute_shap_values(
    pipeline,
    X: pd.DataFrame
) -> tuple:
    """
    Computes SHAP values for the given feature matrix.

    The computation method depends on the model type:

    Tree-based models (XGBoost, LightGBM, Random Forest):
        We use TreeExplainer which uses an exact, efficient algorithm
        specifically designed for tree ensembles. It is orders of
        magnitude faster than the model-agnostic KernelExplainer.

    Linear models (Linear Regression):
        We use LinearExplainer which uses the analytical solution
        for linear models. Also exact and very fast.

    The pipeline contains a preprocessor step before the model.
    We must transform X through the preprocessor before passing
    it to the explainer so the feature values match what the
    model actually received during training.

    Args:
        pipeline: Trained sklearn Pipeline
        X:        Feature matrix in original (pre-preprocessing) form

    Returns:
        Tuple of (shap_values array, X_transformed DataFrame,
                  explainer object)
    """
    logger.info("Computing SHAP values...")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Transform features through the preprocessing step
    X_transformed = preprocessor.transform(X)

    # ColumnTransformer does not preserve names automatically
    # We reconstruct them from the transformer components
    # Get transformed feature names from the ColumnTransformer
    # OneHotEncoder expands categoricals into multiple columns
    # so we must retrieve the actual output column names
    try:
        transformed_feature_names = (
            preprocessor.get_feature_names_out()
        )
        # Clean up prefixes like "numeric__", "compound__", "circuit__"
        cleaned_names = []
        for name in transformed_feature_names:
            # Remove transformer prefixes
            for prefix in [
                "numeric__", "compound__", "circuit__",
                "remainder__"
            ]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            cleaned_names.append(name)
        transformed_feature_names = cleaned_names
    except AttributeError:
        # Fallback: generate names manually
        logger.warning(
            "Could not get feature names from preprocessor. "
            "Generating names manually."
        )
        n_cols = X_transformed.shape[1]
        transformed_feature_names = [
            f"feature_{i}" for i in range(n_cols)
        ]

    # Convert to DataFrame to preserve feature names for plots
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=transformed_feature_names,
        index=X.index
    )

    # Select the appropriate explainer
    model_type = type(model).__name__
    logger.info(f"Model type: {model_type}")

    if model_type in [
        "XGBRegressor",
        "LGBMRegressor",
        "RandomForestRegressor",
        "GradientBoostingRegressor"
    ]:
        logger.info("Using TreeExplainer (exact, efficient)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed_df)

    elif model_type in ["LinearRegression", "Ridge", "Lasso"]:
        logger.info("Using LinearExplainer (exact, analytical)")
        explainer = shap.LinearExplainer(
            model,
            X_transformed_df,
            feature_dependence="independent"
        )
        shap_values = explainer.shap_values(X_transformed_df)

    else:
        # Fallback to KernelExplainer for unknown model types
        # This is slow but model-agnostic
        logger.warning(
            f"Unknown model type {model_type}. "
            f"Using KernelExplainer (slow). "
            f"Consider limiting max_samples."
        )
        background = shap.sample(X_transformed_df, 100)
        explainer = shap.KernelExplainer(
            model.predict, background
        )
        shap_values = explainer.shap_values(X_transformed_df)

    logger.info(
        f"SHAP values computed. Shape: {np.array(shap_values).shape}"
    )

    return shap_values, X_transformed_df, explainer


# ============================================================
# Plot 1 — Summary Bar Plot
# ============================================================

def plot_shap_summary_bar(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    feature_columns: list
) -> str:
    """
    Generates the SHAP summary bar plot showing mean absolute
    feature importance.

    This is the first plot to include in your README because it
    gives an immediate overview of which features the model relies
    on most. Each bar shows the mean absolute SHAP value across
    all predictions — a measure of average impact magnitude.

    Args:
        shap_values:    SHAP values array
        X_transformed:  Transformed feature matrix
        feature_columns: Original feature column names

    Returns:
        Path to saved plot file.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Compute mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        "feature": X_transformed.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=True)

    # Apply human-readable labels
    importance_df["label"] = importance_df["feature"].map(
        FEATURE_LABELS
    ).fillna(importance_df["feature"])

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    bars = ax.barh(
        importance_df["label"],
        importance_df["mean_abs_shap"],
        color=F1_RED,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5
    )

    # Add value labels on bars
    for bar, val in zip(bars, importance_df["mean_abs_shap"]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}s",
            va="center",
            ha="left",
            fontsize=8,
            color=F1_DARK
        )

    ax.set_xlabel(
        "Mean |SHAP Value| (seconds impact on lap delta)",
        fontsize=11
    )
    ax.set_title(
        "Feature Importance — SHAP Values\n"
        "F1 Lap Time Prediction Model",
        fontsize=13,
        fontweight="bold",
        color=F1_DARK
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_summary_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Summary bar plot saved: {path}")
    return path


# ============================================================
# Plot 2 — Beeswarm Plot
# ============================================================

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame
) -> str:
    """
    Generates the SHAP beeswarm plot.

    This is the most information-dense SHAP visualization.
    Each dot is one prediction. The horizontal axis shows
    the SHAP value. The color shows the feature value.

    How to read it:
        Red dots on the right  = high feature value increases lap delta
                                 (makes car slower than median)
        Blue dots on the right = low feature value increases lap delta
        Red dots on the left   = high feature value decreases lap delta
                                 (makes car faster than median)

    For total_car_weight you expect red dots on the right because
    high weight pushes the lap time above median.
    For effective_tire_grip you expect red dots on the left because
    high grip pushes the lap time below median.

    Args:
        shap_values:   SHAP values array
        X_transformed: Transformed feature matrix

    Returns:
        Path to saved plot file.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Apply readable labels to column names
    display_cols = [
        FEATURE_LABELS.get(c, c) for c in X_transformed.columns
    ]
    X_display = X_transformed.copy()
    X_display.columns = display_cols

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("white")

    shap.summary_plot(
        shap_values,
        X_display,
        plot_type="dot",
        show=False,
        color_bar=True,
        plot_size=None,
        alpha=0.6
    )

    ax = plt.gca()
    ax.set_xlabel(
        "SHAP Value (seconds impact on lap delta)",
        fontsize=11
    )
    ax.set_title(
        "SHAP Beeswarm Plot — Feature Impact Distribution\n"
        "F1 Lap Time Prediction Model",
        fontsize=13,
        fontweight="bold",
        color=F1_DARK
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Beeswarm plot saved: {path}")
    return path


# ============================================================
# Plot 3 — Dependence Plots
# ============================================================

def plot_shap_dependence(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    features_to_plot: list = None
) -> list:
    """
    Generates SHAP dependence plots for the most important features.

    A dependence plot shows how a single feature's SHAP value
    changes as the feature value changes. This reveals the
    functional relationship the model has learned between
    a feature and the prediction.

    For this project the most important dependence plots are:
        - total_car_weight: should show positive slope (heavier = slower)
        - effective_tire_grip: should show negative slope (more grip = faster)
        - power_sensitivity_score: reveals circuit-level engine effects

    The color of each dot represents a second feature chosen
    automatically by SHAP as the feature that most interacts
    with the plotted feature. These interaction patterns are
    often highly insightful.

    Args:
        shap_values:      SHAP values array
        X_transformed:    Transformed feature matrix
        features_to_plot: List of feature names to plot.
                          Defaults to top 6 by importance.

    Returns:
        List of paths to saved plot files.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if features_to_plot is None:
        # Default to top 6 features by mean absolute SHAP
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:6]
        features_to_plot = [
            X_transformed.columns[i] for i in top_indices
        ]

    saved_paths = []

    for feature in features_to_plot:
        if feature not in X_transformed.columns:
            logger.warning(
                f"Feature {feature} not found in transformed data"
            )
            continue

        feature_idx = list(X_transformed.columns).index(feature)
        feature_label = FEATURE_LABELS.get(feature, feature)

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("white")

        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_transformed,
            feature_names=[
                FEATURE_LABELS.get(c, c)
                for c in X_transformed.columns
            ],
            ax=ax,
            show=False,
            alpha=0.5,
            dot_size=15
        )

        ax.set_title(
            f"SHAP Dependence — {feature_label}",
            fontsize=12,
            fontweight="bold",
            color=F1_DARK
        )
        ax.set_ylabel(
            f"SHAP Value for {feature_label}\n"
            f"(seconds impact on lap delta)",
            fontsize=10
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        safe_name = feature.replace(" ", "_").replace("/", "_")
        path = os.path.join(
            PLOTS_DIR, f"shap_dependence_{safe_name}.png"
        )
        plt.savefig(
            path, dpi=150, bbox_inches="tight", facecolor="white"
        )
        plt.close()
        saved_paths.append(path)
        logger.info(f"Dependence plot saved: {path}")

    return saved_paths


# ============================================================
# Plot 4 — Waterfall Plot
# ============================================================

def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    X_original: pd.DataFrame,
    meta: pd.DataFrame,
    explainer,
    circuit_filter: str = None,
    sample_index: int = None
) -> str:
    """
    Generates a SHAP waterfall plot for a single prediction.

    The waterfall plot shows how each feature pushes the prediction
    from the baseline (average prediction) to the final value.
    Features pushing the prediction up (toward slower laps) are
    shown in red. Features pushing it down (toward faster laps)
    are shown in blue.

    This is the plot you use in interviews to explain a specific
    prediction in plain English. For example:
        'For this lap at Monza on lap 5, the high car weight due
        to full fuel added 1.2 seconds to the prediction, but the
        fresh soft tire subtracted 0.8 seconds, giving a net
        prediction of +0.4 seconds above the session median.'

    Args:
        shap_values:    SHAP values array
        X_transformed:  Transformed feature matrix
        X_original:     Original untransformed feature matrix
        meta:           Metadata dataframe with circuit column
        explainer:      SHAP explainer object for expected value
        circuit_filter: If provided, select sample from this circuit
        sample_index:   Specific row index to explain.
                        If None, selects a representative sample.

    Returns:
        Path to saved plot file.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Select which prediction to explain
    if sample_index is None:
        if circuit_filter:
            circuit_mask = meta["circuit"] == circuit_filter
            if circuit_mask.sum() == 0:
                logger.warning(
                    f"Circuit {circuit_filter} not found in test data. "
                    f"Using first available sample."
                )
                sample_index = 0
            else:
                # Pick a lap that is close to median for that circuit
                # This gives a more representative example
                circuit_indices = meta[circuit_mask].index
                circuit_shap_sums = np.abs(
                    shap_values[circuit_indices]
                ).sum(axis=1)
                relative_idx = np.argsort(circuit_shap_sums)[
                    len(circuit_shap_sums) // 2
                ]
                sample_index = circuit_indices[relative_idx]
        else:
            # Use the prediction closest to the median SHAP sum
            shap_sums = np.abs(shap_values).sum(axis=1)
            sample_index = int(np.argsort(shap_sums)[len(shap_sums) // 2])

    # Get sample information for the plot title
    sample_meta = meta.iloc[sample_index]
    sample_circuit = sample_meta.get("circuit", "Unknown")
    sample_driver = sample_meta.get("driver", "Unknown")
    sample_lap = sample_meta.get("lap_number", "?")

    logger.info(
        f"Generating waterfall for: "
        f"{sample_circuit} | {sample_driver} | Lap {sample_lap}"
    )

    # Build SHAP Explanation object
    expected_value = (
        explainer.expected_value
        if hasattr(explainer, "expected_value")
        else 0.0
    )

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[0])

    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=expected_value,
        data=X_transformed.iloc[sample_index].values,
        feature_names=[
            FEATURE_LABELS.get(c, c)
            for c in X_transformed.columns
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    shap.waterfall_plot(explanation, show=False, max_display=15)

    plt.title(
        f"SHAP Waterfall — Single Prediction Explanation\n"
        f"Circuit: {sample_circuit} | "
        f"Driver: {sample_driver} | Lap: {sample_lap}",
        fontsize=11,
        fontweight="bold",
        color=F1_DARK
    )

    plt.tight_layout()

    circuit_name = (
        circuit_filter or sample_circuit
    ).replace(" ", "_").lower()
    path = os.path.join(
        PLOTS_DIR, f"shap_waterfall_{circuit_name}.png"
    )
    plt.savefig(
        path, dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    logger.info(f"Waterfall plot saved: {path}")
    return path


# ============================================================
# Plot 5 — Circuit SHAP Heatmap
# ============================================================

def plot_circuit_shap_heatmap(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    meta: pd.DataFrame
) -> str:
    """
    Generates a heatmap of average SHAP values per circuit.

    This is a unique visualization that shows which features
    drive predictions differently across circuits. It directly
    supports the 2026 simulation narrative by showing:

        - power_sensitivity_score matters more at Monza than Monaco
        - total_car_weight matters more at circuits with long runs
        - effective_tire_grip varies by circuit degradation rates

    Reading the heatmap:
        Each row is a circuit. Each column is a feature.
        Red cells mean that feature pushes predictions upward
        (toward slower lap times) at that circuit on average.
        Blue cells mean it pushes predictions downward
        (toward faster lap times).

    Args:
        shap_values:   SHAP values array
        X_transformed: Transformed feature matrix
        meta:          Metadata dataframe with circuit column

    Returns:
        Path to saved plot file.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Build a DataFrame of SHAP values with circuit labels
    shap_df = pd.DataFrame(
        shap_values,
        columns=X_transformed.columns
    )
    shap_df["circuit"] = meta["circuit"].values

    # Compute mean SHAP value per circuit per feature
    circuit_shap = shap_df.groupby("circuit").mean()

    # Apply human-readable labels
    circuit_shap.columns = [
        FEATURE_LABELS.get(c, c) for c in circuit_shap.columns
    ]

    # Select top features by overall variance across circuits
    # These are the features that differ most between circuits
    feature_variance = circuit_shap.var(axis=0).sort_values(
        ascending=False
    )
    top_features = feature_variance.head(10).index.tolist()
    heatmap_data = circuit_shap[top_features]

    fig, ax = plt.subplots(
        figsize=(14, max(6, len(heatmap_data) * 0.6))
    )
    fig.patch.set_facecolor("white")

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Mean SHAP Value (seconds impact on lap delta)"
        },
        ax=ax,
        annot_kws={"size": 8}
    )

    ax.set_title(
        "Circuit-Level SHAP Feature Impact Heatmap\n"
        "Average SHAP contribution per feature per circuit",
        fontsize=13,
        fontweight="bold",
        color=F1_DARK,
        pad=15
    )
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Circuit", fontsize=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_circuit_heatmap.png")
    plt.savefig(
        path, dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    logger.info(f"Circuit SHAP heatmap saved: {path}")
    return path


# ============================================================
# SHAP Summary JSON for Dashboard
# ============================================================

def save_shap_summary_json(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    meta: pd.DataFrame
) -> str:
    """
    Saves a concise SHAP summary as JSON for use in the
    Streamlit dashboard.

    The dashboard needs:
        1. Global feature importance ranking
        2. Per-circuit mean SHAP values for the key features
           that the simulation modifies (weight, power sensitivity)

    This JSON is loaded at dashboard startup rather than
    recomputing SHAP values live which would be too slow
    for an interactive application.

    Args:
        shap_values:   SHAP values array
        X_transformed: Transformed feature matrix
        meta:          Metadata dataframe

    Returns:
        Path to saved JSON file.
    """
    # Global feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = {
        FEATURE_LABELS.get(col, col): round(float(val), 6)
        for col, val in zip(X_transformed.columns, mean_abs_shap)
    }
    feature_importance_sorted = dict(
        sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    # Per-circuit mean SHAP for simulation-relevant features
    simulation_features = [
        "total_car_weight",
        "fuel_weight_estimate",
        "power_sensitivity_score",
        "effective_tire_grip",
        "full_throttle_pct"
    ]

    shap_df = pd.DataFrame(
        shap_values,
        columns=X_transformed.columns
    )
    shap_df["circuit"] = meta["circuit"].values

    circuit_shap_dict = {}
    for circuit in shap_df["circuit"].unique():
        circuit_mask = shap_df["circuit"] == circuit
        circuit_shap_dict[circuit] = {}
        for feat in simulation_features:
            if feat in shap_df.columns:
                mean_val = float(
                    shap_df.loc[circuit_mask, feat].mean()
                )
                circuit_shap_dict[circuit][
                    FEATURE_LABELS.get(feat, feat)
                ] = round(mean_val, 6)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "n_samples": len(shap_values),
        "n_features": len(X_transformed.columns),
        "global_feature_importance": feature_importance_sorted,
        "top_5_features": list(feature_importance_sorted.keys())[:5],
        "circuit_shap_means": circuit_shap_dict
    }

    json_path = os.path.join(MODELS_DIR, "shap_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"SHAP summary JSON saved: {json_path}")
    return json_path


# ============================================================
# Interpretation Report
# ============================================================

def generate_interpretation_report(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    meta: pd.DataFrame
) -> None:
    """
    Generates a written interpretation of the SHAP results.

    This report translates the mathematical SHAP values into
    plain English insights about what the model learned.
    It goes directly into your README methodology section
    and is what you talk through in interviews.

    Args:
        shap_values:   SHAP values array
        X_transformed: Transformed feature matrix
        meta:          Metadata dataframe
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)

    feature_importance = pd.DataFrame({
        "feature": X_transformed.columns,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
        "label": [
            FEATURE_LABELS.get(c, c) for c in X_transformed.columns
        ]
    }).sort_values("mean_abs_shap", ascending=False)

    top_feature = feature_importance.iloc[0]
    second_feature = feature_importance.iloc[1]
    third_feature = feature_importance.iloc[2]

    # Determine directionality for top features
    def direction(mean_shap_val):
        return "increases" if mean_shap_val > 0 else "decreases"

    report_lines = [
        "",
        "=" * 70,
        "SHAP EXPLAINABILITY — MODEL INTERPRETATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "TOP FEATURE INFLUENCES",
        "-" * 70,
        "",
        f"1. {top_feature['label']}",
        f"   Mean |SHAP|: {top_feature['mean_abs_shap']:.4f}s",
        f"   Direction:   On average {direction(top_feature['mean_shap'])} "
        f"lap delta (makes laps "
        f"{'slower' if top_feature['mean_shap'] > 0 else 'faster'} "
        f"relative to session median)",
        "",
        f"2. {second_feature['label']}",
        f"   Mean |SHAP|: {second_feature['mean_abs_shap']:.4f}s",
        f"   Direction:   On average "
        f"{direction(second_feature['mean_shap'])} lap delta",
        "",
        f"3. {third_feature['label']}",
        f"   Mean |SHAP|: {third_feature['mean_abs_shap']:.4f}s",
        f"   Direction:   On average "
        f"{direction(third_feature['mean_shap'])} lap delta",
        "",
        "FULL FEATURE RANKING",
        "-" * 70,
        f"  {'Rank':<6} {'Feature':<45} {'Mean |SHAP|':>12} "
        f"{'Direction':>12}",
        "  " + "-" * 75,
    ]

    for rank, (_, row) in enumerate(feature_importance.iterrows(), 1):
        arrow = "↑ slower" if row["mean_shap"] > 0 else "↓ faster"
        report_lines.append(
            f"  {rank:<6} {row['label']:<45} "
            f"{row['mean_abs_shap']:>12.4f}s "
            f"{arrow:>12}"
        )

    report_lines += [
        "",
        "CIRCUIT-LEVEL INSIGHTS",
        "-" * 70,
    ]

    # Per-circuit analysis of power sensitivity SHAP
    shap_df = pd.DataFrame(
        shap_values,
        columns=X_transformed.columns
    )
    shap_df["circuit"] = meta["circuit"].values

    if "power_sensitivity_score" in shap_df.columns:
        circuit_power_shap = (
            shap_df.groupby("circuit")["power_sensitivity_score"]
            .mean()
            .sort_values(ascending=False)
        )
        report_lines.append(
            "\n  Circuit power sensitivity SHAP contribution "
            "(positive = engine matters more here):"
        )
        for circuit, val in circuit_power_shap.items():
            bar = "█" * int(abs(val) * 50)
            sign = "+" if val >= 0 else "-"
            report_lines.append(
                f"  {circuit:<20} {sign}{abs(val):.4f}s  {bar}"
            )

    report_lines += [
        "",
        "SIMULATION IMPLICATIONS",
        "-" * 70,
        "",
        "Based on SHAP analysis, the 2026 regulation changes will",
        "affect circuits differently because:",
        "",
        "  Weight reduction (-30kg):",
        f"    total_car_weight has mean |SHAP| of "
        f"{feature_importance[feature_importance['feature']=='total_car_weight']['mean_abs_shap'].values[0]:.4f}s",
        "    Circuits with longer straights and higher average",
        "    speeds benefit more from weight reduction.",
        "",
        "  Engine power change (50% electric / 50% combustion):",
        "    power_sensitivity_score drives this effect.",
        "    High-power circuits (Monza, Spa) feel the engine",
        "    rule change most strongly.",
        "    Low-power circuits (Monaco, Hungaroring) benefit",
        "    from the electric torque advantage in slow corners.",
        "",
        "=" * 70,
    ]

    report = "\n".join(report_lines)
    print(report)

    # Save report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )),
        "data", "processed",
        f"shap_interpretation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Interpretation report saved: {report_path}")


# ============================================================
# Main SHAP Pipeline Orchestrator
# ============================================================

def run_shap_pipeline(
    model_name: str = "best_model",
    max_samples: int = 3000,
    circuit_for_waterfall: str = None
) -> None:
    """
    Orchestrates the complete SHAP explainability pipeline.

    Steps:
        1. Load trained model and metadata
        2. Load 2025 test set
        3. Compute SHAP values
        4. Generate all five plot types
        5. Save SHAP summary JSON for dashboard
        6. Generate interpretation report

    Args:
        model_name:            Model file to load (without extension)
        max_samples:           Maximum samples for SHAP computation
        circuit_for_waterfall: Circuit to use for waterfall plot
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("SHAP EXPLAINABILITY PIPELINE STARTING")
    logger.info(
        f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- Step 1: Load model ----
    pipeline, metadata = load_model_and_metadata(model_name)

    feature_columns = metadata.get("feature_columns", [])
    target_column = metadata.get(
        "target_column",
        "lap_time_delta_from_session_median"
    )

    if not feature_columns:
        logger.error(
            "No feature columns in model metadata. "
            "Check best_model_metadata.json exists."
        )
        return

    model_display_name = metadata.get("model_name", model_name)
    logger.info(f"Explaining model: {model_display_name}")

    # ---- Step 2: Load test data ----
    engine = get_sqlalchemy_engine()

    X_test, y_test, meta_test = load_test_data(
        engine,
        feature_columns,
        target_column,
        max_samples=max_samples
    )

    if X_test.empty:
        logger.error(
            "No test data loaded. "
            "Is 2025 data present in the Gold table?"
        )
        return

    logger.info(
        f"SHAP will be computed on {len(X_test)} samples"
    )

    # ---- Step 3: Compute SHAP values ----
    shap_values, X_transformed, explainer = compute_shap_values(
        pipeline, X_test
    )

    # ---- Step 4: Generate plots ----
    logger.info("Generating SHAP visualizations...")

    plot_shap_summary_bar(shap_values, X_transformed, feature_columns)
    plot_shap_beeswarm(shap_values, X_transformed)
    plot_shap_dependence(shap_values, X_transformed)
    plot_shap_waterfall(
        shap_values, X_transformed, X_test,
        meta_test, explainer,
        circuit_filter=circuit_for_waterfall
    )
    plot_circuit_shap_heatmap(shap_values, X_transformed, meta_test)

    # ---- Step 5: Save SHAP summary JSON ----
    save_shap_summary_json(shap_values, X_transformed, meta_test)

    # ---- Step 6: Generate interpretation report ----
    generate_interpretation_report(
        shap_values, X_transformed, meta_test
    )

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 60)
    logger.info(
        f"SHAP PIPELINE COMPLETE. "
        f"All plots saved to {PLOTS_DIR}"
    )
    logger.info(f"Duration: {duration}")
    logger.info("=" * 60)

    # Print plot locations
    print("\nGenerated files:")
    for fname in os.listdir(PLOTS_DIR):
        if fname.startswith("shap_"):
            print(f"  {os.path.join(PLOTS_DIR, fname)}")
    print(f"  {os.path.join(MODELS_DIR, 'shap_summary.json')}")
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="F1 Model Explainability — SHAP Analysis Pipeline"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="best_model",
        help="Model name to explain (default: best_model)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help=(
            "Maximum samples for SHAP computation. "
            "Lower values are faster. Default: 3000"
        )
    )

    parser.add_argument(
        "--circuit",
        type=str,
        default=None,
        help=(
            "Circuit to use for waterfall plot "
            "(e.g., --circuit Monza)"
        )
    )

    args = parser.parse_args()

    run_shap_pipeline(
        model_name=args.model,
        max_samples=args.max_samples,
        circuit_for_waterfall=args.circuit
    )