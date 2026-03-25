# src/models/evaluate_model.py

"""
Model Evaluation Script

Loads a saved model artifact and runs a complete evaluation
report including overall metrics, per-circuit performance,
and residual analysis.

This script is separate from train_model.py because in
production environments training and evaluation often run
at different times on different data.

Usage:
    python src/models/evaluate_model.py
    python src/models/evaluate_model.py --model xgboost
    python src/models/evaluate_model.py --season 2025
"""

import sys
import os
import argparse
import json
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from db_connection import get_sqlalchemy_engine
from logger import get_logger

logger = get_logger("evaluation")

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


def load_model_and_metadata(model_name: str = "best_model"):
    """
    Loads a saved model pipeline and its companion metadata.

    Args:
        model_name: Base filename without extension.
                    Defaults to 'best_model'.

    Returns:
        Tuple of (pipeline, metadata_dict)
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    meta_path = os.path.join(
        MODELS_DIR, f"{model_name}_metadata.json"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run train_model.py first."
        )

    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded from {meta_path}")

    return pipeline, metadata


def load_evaluation_data(
    engine,
    feature_columns: list,
    target_column: str,
    season: int = None
) -> tuple:
    """
    Loads data for evaluation from the Gold table.

    Args:
        engine:          SQLAlchemy engine
        feature_columns: List of feature column names
        target_column:   Target column name
        season:          Optional season filter

    Returns:
        Tuple of (X, y, metadata_df)
    """
    season_filter = f"AND season = {season}" if season else ""

    # Avoid duplicate columns in SQL select
    # These are already selected as metadata above
    already_selected = {"race_id", "season", "circuit", "driver"}
    feature_cols_for_sql = [
        f for f in feature_columns if f not in already_selected
    ]

    query = f"""
        SELECT
            race_id,
            season,
            circuit,
            driver,
            {target_column},
            {", ".join(feature_cols_for_sql)}
        FROM gold_modeling_data
        WHERE {target_column} IS NOT NULL
        {season_filter}
        ORDER BY season, circuit, driver, lap_number
    """

    df = pd.read_sql(query, engine)

    logger.info(
        f"Loaded {len(df)} evaluation rows for "
        f"{df['season'].nunique()} season(s)"
    )

    # Ensure plain Python list
    feature_columns = list(feature_columns)

    # Final safety check: remove duplicate columns if any slipped through
    df = df.loc[:, ~df.columns.duplicated()]

    # Fill nulls carefully
    for col in feature_columns:
        if col not in df.columns:
            logger.error(f"Missing feature column in evaluation data: {col}")
            raise KeyError(f"Missing feature column: {col}")

        if col == "compound":
            df[col] = df[col].fillna("UNKNOWN")
        elif col == "circuit":
            # circuit is categorical, fill with placeholder if ever null
            df[col] = df[col].fillna("UNKNOWN")
        else:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                median_val = df[col].median()
                logger.warning(
                    f"Imputing {null_count} nulls in evaluation column "
                    f"{col} with median {median_val:.4f}"
                )
                df[col] = df[col].fillna(median_val)

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    meta = df[["race_id", "season", "circuit", "driver"]].copy()

    return X, y, meta


def generate_residual_plots(
    actual: np.ndarray,
    predicted: np.ndarray,
    meta: pd.DataFrame,
    model_name: str
) -> None:
    """
    Generates and saves residual analysis plots.

    Plots produced:
        1. Actual vs Predicted scatter plot
        2. Residual distribution histogram
        3. Residuals by circuit box plot

    These plots are saved to data/processed/plots/ and
    are referenced in your GitHub README.

    Args:
        actual:     Array of actual target values
        predicted:  Array of predicted values
        meta:       Metadata dataframe with circuit column
        model_name: Name string for plot titles
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    residuals = predicted - actual

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Model Evaluation — {model_name}",
        fontsize=14,
        fontweight="bold"
    )

    # Plot 1 — Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(actual, predicted, alpha=0.3, s=10, color="#E8002D")
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val],
        "k--", linewidth=1.5, label="Perfect prediction"
    )
    ax1.set_xlabel("Actual Lap Delta (s)")
    ax1.set_ylabel("Predicted Lap Delta (s)")
    ax1.set_title("Actual vs Predicted")
    ax1.legend()

    # Plot 2 — Residual distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=60, color="#E8002D", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax2.axvline(
        residuals.mean(), color="blue",
        linestyle="--", linewidth=1, label=f"Mean: {residuals.mean():.4f}s"
    )
    ax2.set_xlabel("Residual (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend()

    # Plot 3 — Residuals by circuit
    ax3 = axes[2]
    residual_df = pd.DataFrame({
        "circuit": meta["circuit"].values,
        "residual": residuals
    })
    circuit_order = (
        residual_df.groupby("circuit")["residual"]
        .median()
        .sort_values()
        .index
    )
    residual_df["circuit"] = pd.Categorical(
        residual_df["circuit"],
        categories=circuit_order,
        ordered=True
    )
    residual_df_sorted = residual_df.sort_values("circuit")

    ax3.boxplot(
        [
            residual_df_sorted[
                residual_df_sorted["circuit"] == c
            ]["residual"].values
            for c in circuit_order
        ],
        labels=circuit_order,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#E8002D", alpha=0.7)
    )
    ax3.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax3.set_xlabel("Residual (s)")
    ax3.set_title("Residuals by Circuit")

    plt.tight_layout()

    plot_path = os.path.join(
        PLOTS_DIR,
        f"residuals_{model_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Residual plots saved to {plot_path}")


def run_evaluation(
    model_name: str = "best_model",
    season: int = None
) -> None:
    """
    Runs a complete model evaluation and generates all outputs.

    Args:
        model_name: Model filename without extension
        season:     Optional season to evaluate on
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION STARTING")
    logger.info("=" * 60)

    engine = get_sqlalchemy_engine()

    # Load model
    pipeline, metadata = load_model_and_metadata(model_name)

    feature_columns = metadata.get("feature_columns", [])
    target_column = metadata.get(
        "target_column",
        "lap_time_delta_from_session_median"
    )

    if not feature_columns:
        logger.error(
            "No feature columns found in metadata. "
            "The model metadata JSON may be missing."
        )
        return

    # Load evaluation data
    eval_season = season or metadata.get("test_season", 2025)
    logger.info(f"Evaluating on season: {eval_season}")

    X, y, meta = load_evaluation_data(
        engine, feature_columns, target_column, season=eval_season
    )

    if X.empty:
        logger.error(f"No data found for season {eval_season}")
        return

    # Generate predictions
    predictions = pipeline.predict(X)
    residuals = predictions - y.values

    # Overall metrics
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae  = np.mean(np.abs(residuals))
    r2   = 1 - (np.sum(residuals**2) / np.sum((y - y.mean())**2))

    target_rmse = 0.90

    print("\n" + "=" * 65)
    print(f"EVALUATION REPORT — {metadata.get('model_name', model_name)}")
    print(f"Evaluation Season: {eval_season}  |  Samples: {len(y)}")
    print("=" * 65)
    print(f"  RMSE:                 {rmse:.4f} seconds  (target < {target_rmse}s)")
    print(f"  MAE:                  {mae:.4f} seconds")
    print(f"  R2:                   {r2:.4f}")
    print(f"  Bias (mean residual): {residuals.mean():.4f} seconds")

    if rmse < target_rmse:
        print(f"\n  ✓ Target met: RMSE {rmse:.4f}s is below the {target_rmse}s threshold")
    else:
        print(f"\n  ✗ Target not met: RMSE {rmse:.4f}s exceeds the {target_rmse}s threshold")

    # Per-circuit breakdown
    results_df = meta.copy()
    results_df["actual"]    = y.values
    results_df["predicted"] = predictions
    results_df["abs_error"] = np.abs(residuals)
    results_df["residual"]  = residuals

    print("\nPer-Circuit Performance (sorted best to worst):")
    print(f"  {'Circuit':<20} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'N':>6}")
    print("  " + "-" * 54)

    circuit_perf = (
        results_df.groupby("circuit")
        .agg(
            rmse=("residual", lambda x: np.sqrt((x**2).mean())),
            mae=("abs_error", "mean"),
            bias=("residual", "mean"),
            n=("actual", "count")
        )
        .sort_values("rmse")
    )

    for circuit, row in circuit_perf.iterrows():
        flag = "  <- known outlier circuit" if row["rmse"] > 1.2 else ""
        print(
            f"  {circuit:<20} "
            f"{row['rmse']:>8.4f} "
            f"{row['mae']:>8.4f} "
            f"{row['bias']:>8.4f} "
            f"{int(row['n']):>6}"
            f"{flag}"
        )

    # Per-season breakdown
    print("\nPer-Season Performance:")
    print(f"  {'Season':<10} {'RMSE':>8} {'MAE':>8} {'N':>6}")
    print("  " + "-" * 36)

    season_perf = (
        results_df.groupby("season")
        .agg(
            rmse=("residual", lambda x: np.sqrt((x**2).mean())),
            mae=("abs_error", "mean"),
            n=("actual", "count")
        )
        .sort_values("season")
    )

    for season_val, row in season_perf.iterrows():
        print(
            f"  {season_val:<10} "
            f"{row['rmse']:>8.4f} "
            f"{row['mae']:>8.4f} "
            f"{int(row['n']):>6}"
        )

    # Residual distribution summary
    print(f"\nResidual Distribution:")
    print(f"  Mean:            {residuals.mean():>8.4f}s  (should be near 0)")
    print(f"  Std:             {residuals.std():>8.4f}s")
    print(f"  Min:             {residuals.min():>8.4f}s")
    print(f"  Max:             {residuals.max():>8.4f}s")
    within_half = (np.abs(residuals) < 0.5).mean() * 100
    within_one  = (np.abs(residuals) < 1.0).mean() * 100
    print(f"  Within +/-0.5s:  {within_half:.1f}% of predictions")
    print(f"  Within +/-1.0s:  {within_one:.1f}% of predictions")

    print("=" * 65)

    # Generate residual plots
    generate_residual_plots(
        y.values, predictions, meta,
        metadata.get("model_name", model_name)
    )

    logger.info("Evaluation complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="F1 Model Evaluation Script"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="best_model",
        # Linear Regression removed from active pipeline
        choices=["best_model", "random_forest", "xgboost", "lightgbm"],
        help="Model name to evaluate (default: best_model)"
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to evaluate on (default: 2025)"
    )

    args = parser.parse_args()
    run_evaluation(model_name=args.model, season=args.season)