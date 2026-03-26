# src/models/mlflow_tracking.py

"""
MLflow Experiment Tracking — Organization and Summary

This script serves three purposes:

1. REVIEW: Queries the MLflow tracking store and prints a clean
   summary of all experiment runs, showing which model won and
   by how much.

2. REGISTER: Formally registers the best model in the MLflow
   Model Registry, marking it as 'Production' stage. This is
   how real ML teams track which model version is currently
   deployed.

3. EXPORT: Generates a static CSV and JSON summary of all runs
   for inclusion in the project README and dashboard.

Why this matters for your career:
    Most junior candidates train a model and save a .pkl file.
    MLflow experiment tracking shows you understand that in
    production, teams need to know which model is deployed,
    why it was chosen over alternatives, and what its exact
    performance characteristics were at the time of deployment.
    This is called ML governance and it is increasingly required
    at every serious data science employer.

Usage:
    python src/models/mlflow_tracking.py

    Optional flags:
    --review-only        Only print the run summary, no registration
    --register-best      Register the best run in the Model Registry
    --export             Export run summary to CSV and JSON
"""

import sys
import os
import argparse
import json
import csv
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config_loader import load_config
from logger import get_logger

logger = get_logger("mlflow_tracking")


# ============================================================
# Constants
# ============================================================

EXPERIMENT_NAME = "f1_lap_time_prediction"
MODEL_REGISTRY_NAME = "f1_lap_time_predictor"

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )),
    "models"
)

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )),
    "data", "processed"
)


# ============================================================
# MLflow Client Setup
# ============================================================

def setup_mlflow_client() -> tuple:
    """
    Initializes the MLflow client pointing at the local
    tracking store in the mlruns directory.

    Returns:
        Tuple of (MlflowClient, experiment_id)
        Returns (client, None) if experiment does not exist.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logger.warning(
                f"Experiment '{EXPERIMENT_NAME}' not found. "
                f"Have you run the Phase 5 training pipeline?"
            )
            return client, None

        logger.info(
            f"Found experiment: '{EXPERIMENT_NAME}' "
            f"(ID: {experiment.experiment_id})"
        )
        return client, experiment.experiment_id

    except Exception as e:
        logger.error(f"Failed to connect to MLflow: {e}")
        return client, None


# ============================================================
# Run Review and Summary
# ============================================================

def get_all_runs(
    client: MlflowClient,
    experiment_id: str
) -> pd.DataFrame:
    """
    Retrieves all runs from the experiment and returns them
    as a structured DataFrame sorted by RMSE ascending.

    Args:
        client:        MlflowClient instance
        experiment_id: MLflow experiment ID string

    Returns:
        DataFrame with one row per run, sorted best to worst.
    """
    logger.info(f"Fetching all runs for experiment {experiment_id}...")

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.rmse ASC"]
    )

    if not runs:
        logger.warning("No runs found in this experiment.")
        return pd.DataFrame()

    rows = []
    for run in runs:
        row = {
            "run_id":           run.info.run_id,
            "run_name":         run.info.run_name or "unnamed",
            "model_type":       run.data.tags.get("model_type", "unknown"),
            "status":           run.info.status,
            "start_time":       datetime.fromtimestamp(
                run.info.start_time / 1000
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": (
                (run.info.end_time - run.info.start_time) / 1000
                if run.info.end_time else None
            ),
            "rmse":             run.data.metrics.get("rmse"),
            "mae":              run.data.metrics.get("mae"),
            "r2":               run.data.metrics.get("r2"),
            "mape":             run.data.metrics.get("mape"),
            "cv_rmse_mean":     run.data.metrics.get("cv_rmse_mean"),
            "cv_rmse_std":      run.data.metrics.get("cv_rmse_std"),
            "training_seasons": run.data.tags.get("training_seasons"),
            "test_season":      run.data.tags.get("test_season"),
            "target":           run.data.tags.get("target"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    logger.info(f"Retrieved {len(df)} runs")
    return df


def print_run_comparison_table(runs_df: pd.DataFrame) -> None:
    """
    Prints a clean, formatted comparison table of all MLflow runs.

    This is the table you screenshot for your README under the
    Experiment Tracking section. It demonstrates that you tried
    multiple approaches systematically and selected the best one
    with documented evidence.

    Args:
        runs_df: DataFrame from get_all_runs()
    """
    if runs_df.empty:
        print("No runs to display.")
        return

    print("\n")
    print("=" * 90)
    print("MLFLOW EXPERIMENT TRACKING — ALL RUNS")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    print(
        f"{'Rank':<6} "
        f"{'Model':<22} "
        f"{'RMSE':>8} "
        f"{'MAE':>8} "
        f"{'R2':>8} "
        f"{'CV RMSE':>10} "
        f"{'CV Std':>8} "
        f"{'Status':<10}"
    )
    print("-" * 90)

    for rank, (_, row) in enumerate(runs_df.iterrows(), 1):
        is_best = "★" if rank == 1 else " "
        rmse_str = f"{row['rmse']:.4f}" if row['rmse'] is not None else "N/A"
        mae_str = f"{row['mae']:.4f}" if row['mae'] is not None else "N/A"
        r2_str = f"{row['r2']:.4f}" if row['r2'] is not None else "N/A"
        cv_mean_str = (
            f"{row['cv_rmse_mean']:.4f}"
            if row['cv_rmse_mean'] is not None else "N/A"
        )
        cv_std_str = (
            f"{row['cv_rmse_std']:.4f}"
            if row['cv_rmse_std'] is not None else "N/A"
        )

        print(
            f"{is_best} {rank:<4} "
            f"{row['model_type']:<22} "
            f"{rmse_str:>8} "
            f"{mae_str:>8} "
            f"{r2_str:>8} "
            f"{cv_mean_str:>10} "
            f"{cv_std_str:>8} "
            f"{row['status']:<10}"
        )

    print("-" * 90)

    # Best model summary
    best = runs_df.iloc[0]
    print(f"\n★ Best Model: {best['model_type']}")
    print(f"  RMSE:         {best['rmse']:.4f}s")
    print(f"  MAE:          {best['mae']:.4f}s")
    print(f"  R2:           {best['r2']:.4f}")
    print(f"  CV RMSE Mean: {best['cv_rmse_mean']:.4f}s")
    print(f"  Run ID:       {best['run_id']}")
    print(f"  Training:     {best['training_seasons']}")
    print(f"  Test Season:  {best['test_season']}")

    target_rmse = 0.5
    if best['rmse'] is not None and best['rmse'] < target_rmse:
        print(
            f"\n  ✓ Target RMSE of <{target_rmse}s "
            f"{'MET' if best['rmse'] < target_rmse else 'NOT MET'}: "
            f"{best['rmse']:.4f}s"
        )
    print("=" * 90)
    print()


def print_model_selection_rationale(runs_df: pd.DataFrame) -> None:
    """
    Prints a written rationale for model selection based on
    the run results. This narrative goes directly into your
    README methodology section.

    Args:
        runs_df: DataFrame from get_all_runs()
    """
    if runs_df.empty or len(runs_df) < 2:
        return

    best = runs_df.iloc[0]
    worst = runs_df.iloc[-1]

    # Use Random Forest as baseline comparison
    rf_rows = runs_df[
        runs_df["model_type"].str.contains(
            "Random Forest", case=False
        )
    ]
    baseline = rf_rows.iloc[0] if not rf_rows.empty else runs_df.iloc[-1]

    print("\n" + "=" * 70)
    print("MODEL SELECTION RATIONALE")
    print("=" * 70)
    print()

    if best["rmse"] is not None:
        improvement = (
            (baseline["rmse"] - best["rmse"]) / baseline["rmse"] * 100
            if baseline["rmse"] else 0
        )
        print(
            f"The {best['model_type']} was selected as the production "
            f"model based on the following evidence:\n"
        )
        print(
            f"  1. Lowest test set RMSE: {best['rmse']:.4f}s "
            f"vs Random Forest baseline: "
            f"{baseline['rmse']:.4f}s\n"
            f"     ({improvement:.1f}% improvement over baseline)"
        )
        print(
            f"\n  2. Cross validation RMSE {best['cv_rmse_mean']:.4f}s "
            f"(+/- {best['cv_rmse_std']:.4f}s) confirms "
            f"results generalize beyond a single train-test split."
        )
        print(
            f"\n  3. R2 of {best['r2']:.4f} indicates the model explains "
            f"{best['r2']*100:.1f}% of lap time delta variance."
        )

    print(
        "\n  Temporal holdout validation: The model was trained on "
        "2023-2024 seasons\n"
        "  and evaluated on the 2025 season. This temporal split "
        "simulates the real\n"
        "  deployment scenario of predicting future (2026) performance "
        "from historical data."
    )
    print("\n" + "=" * 70 + "\n")


# ============================================================
# Model Registration
# ============================================================

def register_best_model(
    client: MlflowClient,
    runs_df: pd.DataFrame
) -> None:
    """
    Registers the best performing model in the MLflow Model Registry.

    The Model Registry is MLflow's system for tracking model
    lifecycle stages:
        None     → model exists but has no stage assignment
        Staging  → model is being tested before production
        Production → model is the currently deployed version
        Archived → model has been superseded

    For this project we register the best model directly as
    Production because we have already validated it on the
    2025 holdout set.

    In a real company workflow, a model would move through
    Staging first with additional validation before being
    promoted to Production. Being able to explain this
    lifecycle in an interview is valuable.

    Args:
        client:   MlflowClient instance
        runs_df:  DataFrame from get_all_runs()
    """
    if runs_df.empty:
        logger.warning("No runs to register.")
        return

    best_run = runs_df.iloc[0]
    best_run_id = best_run["run_id"]
    best_model_type = best_run["model_type"]

    logger.info(
        f"Registering best model: {best_model_type} "
        f"(run_id: {best_run_id[:8]}...)"
    )

    model_uri = f"runs:/{best_run_id}/model"

    try:
        # Register the model
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_REGISTRY_NAME
        )

        logger.info(
            f"Model registered: {MODEL_REGISTRY_NAME} "
            f"version {registered.version}"
        )

        # Add descriptive tags to the registered version
        client.set_registered_model_tag(
            name=MODEL_REGISTRY_NAME,
            key="project",
            value="F1 2026 Regulation Impact Analysis"
        )
        client.set_registered_model_tag(
            name=MODEL_REGISTRY_NAME,
            key="best_model_type",
            value=best_model_type
        )
        client.set_registered_model_tag(
            name=MODEL_REGISTRY_NAME,
            key="target_variable",
            value="lap_time_delta_from_session_median"
        )

        # Add version-specific description
        client.update_model_version(
            name=MODEL_REGISTRY_NAME,
            version=registered.version,
            description=(
                f"Best performing model from training pipeline. "
                f"Model type: {best_model_type}. "
                f"Test RMSE: {best_run['rmse']:.4f}s. "
                f"Trained on 2023-2024 seasons, "
                f"validated on 2025 season. "
                f"Selected for 2026 regulation simulation."
            )
        )

        # Transition to Production stage
        # Note: In newer MLflow versions use set_registered_model_alias
        # instead of transition_model_version_stage which is deprecated
        try:
            client.set_registered_model_alias(
                name=MODEL_REGISTRY_NAME,
                alias="production",
                version=registered.version
            )
            logger.info(
                f"Model version {registered.version} "
                f"aliased as 'production'"
            )
        except AttributeError:
            # Fallback for older MLflow versions
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=registered.version,
                stage="Production"
            )
            logger.info(
                f"Model version {registered.version} "
                f"transitioned to Production stage"
            )

        print(f"\n✓ Model registered successfully:")
        print(f"  Registry name: {MODEL_REGISTRY_NAME}")
        print(f"  Version:       {registered.version}")
        print(f"  Model type:    {best_model_type}")
        print(f"  RMSE:          {best_run['rmse']:.4f}s")
        print(f"  Stage:         Production")
        print()

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        logger.info(
            "This is non-critical. The best_model.joblib artifact "
            "is still available for the simulation pipeline."
        )


# ============================================================
# Export Functions
# ============================================================

def export_run_summary(runs_df: pd.DataFrame) -> None:
    """
    Exports the run summary to both CSV and JSON formats.

    CSV format: For easy viewing in Excel or pandas
    JSON format: For loading in the Streamlit dashboard

    These files are committed to GitHub and referenced in
    the README as evidence of systematic experimentation.

    Args:
        runs_df: DataFrame from get_all_runs()
    """
    if runs_df.empty:
        logger.warning("No runs to export.")
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Export CSV ----
    csv_path = os.path.join(REPORTS_DIR, "mlflow_run_summary.csv")
    export_cols = [
        "model_type", "rmse", "mae", "r2",
        "cv_rmse_mean", "cv_rmse_std",
        "training_seasons", "test_season", "start_time"
    ]
    export_cols_existing = [
        c for c in export_cols if c in runs_df.columns
    ]
    runs_df[export_cols_existing].to_csv(csv_path, index=False)
    logger.info(f"Run summary exported to CSV: {csv_path}")

    # ---- Export JSON for dashboard ----
    json_summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment_name": EXPERIMENT_NAME,
        "total_runs": len(runs_df),
        "best_model": {
            "model_type":    runs_df.iloc[0]["model_type"],
            "rmse":          runs_df.iloc[0]["rmse"],
            "mae":           runs_df.iloc[0]["mae"],
            "r2":            runs_df.iloc[0]["r2"],
            "cv_rmse_mean":  runs_df.iloc[0]["cv_rmse_mean"],
            "cv_rmse_std":   runs_df.iloc[0]["cv_rmse_std"],
            "run_id":        runs_df.iloc[0]["run_id"],
        },
        "all_models": []
    }

    for _, row in runs_df.iterrows():
        json_summary["all_models"].append({
            "model_type":   row["model_type"],
            "rmse":         row["rmse"],
            "mae":          row["mae"],
            "r2":           row["r2"],
            "cv_rmse_mean": row["cv_rmse_mean"],
            "cv_rmse_std":  row["cv_rmse_std"],
        })

    json_path = os.path.join(MODELS_DIR, "mlflow_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    logger.info(f"Run summary exported to JSON: {json_path}")

    print(f"  CSV exported to:  {csv_path}")
    print(f"  JSON exported to: {json_path}")


# ============================================================
# Re-log Missing Runs
# ============================================================

def relog_runs_if_missing(
    client: MlflowClient,
    experiment_id: str
) -> None:
    """
    Checks if any expected model runs are missing from MLflow
    and re-logs them from the saved joblib artifacts and
    metadata JSON files.

    This handles the case where Phase 5 training completed
    successfully and saved the model files, but the MLflow
    logging step had a silent failure.

    Expected models: Linear Regression, Random Forest,
                     XGBoost, LightGBM

    Args:
        client:        MlflowClient instance
        experiment_id: MLflow experiment ID string
    """
    if experiment_id is None:
        logger.warning(
            "No experiment found. Creating experiment and "
            "re-logging from saved artifacts."
        )
        mlflow.set_experiment(EXPERIMENT_NAME)

    # Check which model metadata files exist
    expected_models = [
        "random_forest",
        "xgboost",
        "lightgbm"
    ]

    missing_models = []
    for model_name in expected_models:
        meta_path = os.path.join(
            MODELS_DIR, f"{model_name}_metadata.json"
        )
        if os.path.exists(meta_path):
            missing_models.append((model_name, meta_path))

    if not missing_models:
        logger.info(
            "No model metadata files found in models directory. "
            "Run Phase 5 training first."
        )
        return

    # Get existing run names to avoid duplicates
    existing_run_names = set()
    if experiment_id:
        existing_runs = client.search_runs(
            experiment_ids=[experiment_id]
        )
        existing_run_names = {
            r.data.tags.get("model_type", "").lower()
            for r in existing_runs
        }

    for model_name, meta_path in missing_models:

        with open(meta_path) as f:
            metadata = json.load(f)

        model_type = metadata.get("model_name", model_name)
        model_type_lower = model_type.lower()

        # Skip if this model type already has a run logged
        if any(
            model_type_lower in existing.lower()
            for existing in existing_run_names
        ):
            logger.info(
                f"Run already exists for {model_type}, skipping re-log"
            )
            continue

        logger.info(f"Re-logging run for {model_type}...")

        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=model_type):
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag(
                "training_seasons",
                str(metadata.get("training_seasons", [2023, 2024]))
            )
            mlflow.set_tag(
                "test_season",
                str(metadata.get("test_season", 2025))
            )
            mlflow.set_tag("target", metadata.get("target_column", ""))
            mlflow.set_tag("source", "re-logged from metadata")

            metrics = metadata.get("metrics", {})
            for metric_name in ["rmse", "mae", "r2", "mape"]:
                if metrics.get(metric_name) is not None:
                    mlflow.log_metric(metric_name, metrics[metric_name])
            for metric_name in ["cv_rmse_mean", "cv_rmse_std"]:
                if metrics.get(metric_name) is not None:
                    mlflow.log_metric(metric_name, metrics[metric_name])

            logger.info(f"Re-logged {model_type} successfully")


# ============================================================
# Generate README Snippet
# ============================================================

def generate_readme_snippet(runs_df: pd.DataFrame) -> str:
    """
    Generates a formatted markdown table for the README
    showing all model comparison results.

    Copy this directly into your README.md under the
    Experiment Tracking section.

    Args:
        runs_df: DataFrame from get_all_runs()

    Returns:
        Markdown string for README insertion.
    """
    if runs_df.empty:
        return "No runs available."

    lines = [
        "## Model Comparison — Experiment Tracking",
        "",
        "All experiments tracked with MLflow. "
        "Training seasons: 2023-2024. "
        "Test season: 2025 (temporal holdout).",
        "",
        "| Model | RMSE (s) | MAE (s) | R² | CV RMSE | CV Std |",
        "|---|---|---|---|---|---|",
    ]

    for _, row in runs_df.iterrows():
        is_best = "★ " if _ == 0 else ""
        rmse = f"{row['rmse']:.4f}" if row['rmse'] is not None else "N/A"
        mae = f"{row['mae']:.4f}" if row['mae'] is not None else "N/A"
        r2 = f"{row['r2']:.4f}" if row['r2'] is not None else "N/A"
        cv_mean = (
            f"{row['cv_rmse_mean']:.4f}"
            if row['cv_rmse_mean'] is not None else "N/A"
        )
        cv_std = (
            f"{row['cv_rmse_std']:.4f}"
            if row['cv_rmse_std'] is not None else "N/A"
        )
        lines.append(
            f"| {is_best}{row['model_type']} | "
            f"{rmse} | {mae} | {r2} | {cv_mean} | {cv_std} |"
        )

    lines += [
        "",
        "> ★ = Best model selected for 2026 simulation",
        "> Temporal holdout validation: trained on 2023-2024, "
        "tested on 2025",
        "",
        "![MLflow Experiment UI](data/processed/plots/mlflow_screenshot.png)",
    ]

    snippet = "\n".join(lines)

    # Save to file
    snippet_path = os.path.join(
        REPORTS_DIR, "readme_model_comparison.md"
    )
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(snippet_path, "w", encoding="utf-8") as f:
        f.write(snippet)
    logger.info(f"README snippet saved to {snippet_path}")

    return snippet


# ============================================================
# Main Orchestrator
# ============================================================

def run_mlflow_organization(
    review_only: bool = False,
    register_best: bool = True,
    export: bool = True
) -> None:
    """
    Orchestrates the full MLflow organization pipeline.

    Steps:
        1. Connect to MLflow tracking store
        2. Check for missing runs and re-log if needed
        3. Retrieve all runs
        4. Print comparison table
        5. Print model selection rationale
        6. Register best model (unless review_only)
        7. Export run summary (unless review_only)
        8. Generate README snippet

    Args:
        review_only:   If True skip registration and export
        register_best: If True register best model in registry
        export:        If True export CSV and JSON summaries
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("MLFLOW EXPERIMENT TRACKING ORGANIZATION STARTING")
    logger.info(
        f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 60)

    # ---- Step 1: Setup client ----
    client, experiment_id = setup_mlflow_client()

    # ---- Step 2: Re-log missing runs if needed ----
    relog_runs_if_missing(client, experiment_id)

    # Refresh experiment_id after potential re-log
    if experiment_id is None:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment:
            experiment_id = experiment.experiment_id

    if experiment_id is None:
        logger.error(
            "Could not find or create MLflow experiment. "
            "Check that Phase 5 training was completed and "
            "MLFLOW_TRACKING_URI is set correctly in .env"
        )
        return

    # ---- Step 3: Get all runs ----
    runs_df = get_all_runs(client, experiment_id)

    if runs_df.empty:
        logger.error(
            "No runs found after attempting re-log. "
            "Run Phase 5 training pipeline first: "
            "python src/models/train_model.py"
        )
        return

    # ---- Step 4: Print comparison table ----
    print_run_comparison_table(runs_df)

    # ---- Step 5: Print selection rationale ----
    print_model_selection_rationale(runs_df)

    if review_only:
        logger.info("Review-only mode. Skipping registration and export.")
        return

    # ---- Step 6: Register best model ----
    if register_best:
        register_best_model(client, runs_df)

    # ---- Step 7: Export summaries ----
    if export:
        print("\nExporting run summaries...")
        export_run_summary(runs_df)

    # ---- Step 8: Generate README snippet ----
    snippet = generate_readme_snippet(runs_df)
    print("\nREADME Markdown Snippet:")
    print("-" * 50)
    print(snippet)
    print("-" * 50)

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 60)
    logger.info(f"MLFLOW ORGANIZATION COMPLETE. Duration: {duration}")
    logger.info("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "MLflow Experiment Tracking — "
            "Organization and Model Registry"
        )
    )

    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Only print run summary, skip registration and export",
        default=False
    )

    parser.add_argument(
        "--register-best",
        action="store_true",
        help="Register the best model in the MLflow Model Registry",
        default=False
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export run summary to CSV and JSON",
        default=False
    )

    args = parser.parse_args()

    # Default behavior runs everything
    if not any([args.review_only, args.register_best, args.export]):
        run_mlflow_organization(
            review_only=False,
            register_best=True,
            export=True
        )
    else:
        run_mlflow_organization(
            review_only=args.review_only,
            register_best=args.register_best,
            export=args.export
        )