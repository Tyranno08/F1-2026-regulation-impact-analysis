"""
2026 F1 Regulation Simulation Engine

Refined version:
- Applies all 3 regulation changes
- Produces separate decomposition:
    weight / combustion / electric
- Validates against real 2026 data
- Includes fallback simulation for circuits present in 2026 validation
  but absent from 2025 baseline simulation set (e.g. Australia)
"""

import os
import sys
import json
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from db_connection import get_sqlalchemy_engine, get_mysql_connection
from config_loader import load_config
from logger import get_logger

logger = get_logger("simulation")


# ============================================================
# Paths / Constants
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# Model and Features
# ============================================================

def load_model_and_features():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    feature_path = os.path.join(MODELS_DIR, "feature_columns.json")
    meta_path = os.path.join(MODELS_DIR, "best_model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found: {model_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature column file not found: {feature_path}")

    pipeline = joblib.load(model_path)

    with open(feature_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    logger.info(f"Model loaded: {model_path}")
    logger.info(f"Best model: {metadata.get('model_name', 'Unknown')}")
    logger.info(f"Feature columns loaded: {len(feature_columns)} features")

    return pipeline, feature_columns, metadata


# ============================================================
# Data Loading
# ============================================================

def load_2025_baseline(engine, feature_columns: list) -> pd.DataFrame:
    """
    Loads 2025 baseline laps and ensures no duplicate columns.
    """
    logger.info("Loading 2025 baseline data from Gold table...")

    target_col = "lap_time_delta_from_session_median"

    metadata_cols = ["race_id", "season", "circuit", "driver", "team", "lap_number", target_col]

    selected_cols = metadata_cols.copy()
    for col in feature_columns:
        if col not in selected_cols:
            selected_cols.append(col)

    query = f"""
        SELECT {", ".join(selected_cols)}
        FROM gold_modeling_data
        WHERE season = 2025
          AND {target_col} IS NOT NULL
        ORDER BY circuit, driver, lap_number
    """

    df = pd.read_sql(query, engine)

    for col in feature_columns:
        if col in df.columns:
            if df[col].isna().any(axis=None):
                if col == "compound":
                    df[col] = df[col].fillna("UNKNOWN")
                elif df[col].dtype == "O":
                    df[col] = df[col].fillna("UNKNOWN")
                else:
                    df[col] = df[col].fillna(df[col].median())

    logger.info(f"Loaded {len(df)} baseline laps from 2025.")
    return df


def load_2026_real_data(engine, feature_columns):
    logger.info("Loading real 2026 validation data...")

    target_col = "lap_time_delta_from_session_median"
    selected_base = ["race_id", "season", "circuit", "driver", "team", "lap_number", target_col]
    feature_cols_for_sql = [c for c in feature_columns if c not in selected_base]

    query = f"""
        SELECT
            race_id,
            season,
            circuit,
            driver,
            team,
            lap_number,
            {target_col},
            {", ".join(feature_cols_for_sql)}
        FROM gold_modeling_data
        WHERE season = 2026
          AND data_split = 'validation'
          AND {target_col} IS NOT NULL
        ORDER BY circuit, driver, lap_number
    """

    df = pd.read_sql(query, engine)
    df = df.loc[:, ~df.columns.duplicated()]

    logger.info(
        f"Loaded {len(df)} real 2026 laps: "
        f"{df['circuit'].nunique()} circuits "
        f"({', '.join(sorted(df['circuit'].unique())) if len(df) else 'none'})"
    )

    return df


# ============================================================
# Feature Safety
# ============================================================

def ensure_features(df, feature_columns):
    df = df.copy()

    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f"Missing feature in simulation input: {col}. Filling with 0.")
            df[col] = 0

    for col in feature_columns:
        if col == "compound":
            df[col] = df[col].fillna("UNKNOWN")
        else:
            if df[col].dtype == "O":
                df[col] = df[col].fillna("UNKNOWN")
            else:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())

    return df[feature_columns]


# ============================================================
# Transformations
# ============================================================

def apply_weight_transformation(df, config):
    df = df.copy()

    sim_cfg = config.get("simulation_2026", {})
    weight_reduction = sim_cfg.get("weight_reduction_kg", 20)

    weight_change = 0.0
    fuel_change = 0.0

    if "total_car_weight" in df.columns:
        old_weight = df["total_car_weight"].copy()
        df["total_car_weight"] = (df["total_car_weight"] - weight_reduction).clip(lower=700)
        weight_change = (df["total_car_weight"] - old_weight).mean()

    if "fuel_weight_estimate" in df.columns:
        old_fuel = df["fuel_weight_estimate"].copy()
        df["fuel_weight_estimate"] = (df["fuel_weight_estimate"] - weight_reduction).clip(lower=0)
        fuel_change = (df["fuel_weight_estimate"] - old_fuel).mean()

    logger.info(
        f"Weight transformation applied. Mean weight change: {weight_change:.1f}kg "
        f"| Mean fuel change: {fuel_change:.1f}kg"
    )
    return df


def apply_combustion_transformation(df, config):
    df = df.copy()

    sim_cfg = config.get("simulation_2026", {})
    combustion_penalty_factor = sim_cfg.get("combustion_power_penalty_factor", 0.085)

    if "power_sensitivity_score" not in df.columns:
        logger.warning("power_sensitivity_score missing - combustion transformation skipped")
        return df

    power_sens = df["power_sensitivity_score"].fillna(0.5)

    old_full_throttle = df["full_throttle_pct"].copy() if "full_throttle_pct" in df.columns else None
    old_speed_trap = df["speed_trap"].copy() if "speed_trap" in df.columns else None

    # Use actual throttle profile too, not only power sensitivity
    if "full_throttle_pct" in df.columns:
        throttle_profile = df["full_throttle_pct"].fillna(df["full_throttle_pct"].median())
    else:
        throttle_profile = pd.Series(0.6, index=df.index)

    # Moderated penalty: depends on both circuit power sensitivity and throttle exposure
    combined_exposure = (0.6 * power_sens) + (0.4 * throttle_profile)

    # cap to avoid over-penalizing fallback circuits like Australia
    combined_exposure = combined_exposure.clip(lower=0.2, upper=0.85)

    if "full_throttle_pct" in df.columns:
        delta = combustion_penalty_factor * combined_exposure
        df["full_throttle_pct"] = (df["full_throttle_pct"] * (1 - delta)).clip(lower=0, upper=1)

    if "speed_trap" in df.columns:
        speed_delta = combustion_penalty_factor * 0.35 * combined_exposure
        df["speed_trap"] = (df["speed_trap"] * (1 - speed_delta)).clip(lower=100)

    msg_parts = ["Combustion penalty applied."]
    if old_full_throttle is not None:
        msg_parts.append(
            f"Mean full_throttle_pct change: {(df['full_throttle_pct'] - old_full_throttle).mean():.4f}"
        )
    if old_speed_trap is not None:
        msg_parts.append(
            f"Mean speed_trap change: {(df['speed_trap'] - old_speed_trap).mean():.4f}"
        )

    logger.info(" ".join(msg_parts))
    return df


def apply_electric_transformation(df, config):
    df = df.copy()

    sim_cfg = config.get("simulation_2026", {})
    electric_gain_factor = sim_cfg.get("electric_torque_gain_factor", 0.03)

    if "power_sensitivity_score" not in df.columns:
        logger.warning("power_sensitivity_score missing - electric transformation skipped")
        return df

    power_sens = df["power_sensitivity_score"].fillna(0.5)
    low_power_benefit = 1 - power_sens

    old_corner_speed = df["avg_corner_speed_kmh"].copy() if "avg_corner_speed_kmh" in df.columns else None
    old_grip = df["effective_tire_grip"].copy() if "effective_tire_grip" in df.columns else None

    if "avg_corner_speed_kmh" in df.columns:
        corner_gain = electric_gain_factor * low_power_benefit
        df["avg_corner_speed_kmh"] = df["avg_corner_speed_kmh"] * (1 + corner_gain)

    if "effective_tire_grip" in df.columns:
        grip_gain = electric_gain_factor * 0.25 * low_power_benefit
        df["effective_tire_grip"] = (df["effective_tire_grip"] * (1 + grip_gain)).clip(upper=1.0)

    msg_parts = ["Electric torque benefit applied."]
    if old_corner_speed is not None:
        msg_parts.append(
            f"Mean corner speed change: {(df['avg_corner_speed_kmh'] - old_corner_speed).mean():.4f} km/h"
        )
    if old_grip is not None:
        msg_parts.append(
            f"Mean effective_tire_grip change: {(df['effective_tire_grip'] - old_grip).mean():.4f}"
        )

    logger.info(" ".join(msg_parts))
    return df


def apply_all_2026_transformations(df, config):
    df_weight = apply_weight_transformation(df, config)
    df_weight_combustion = apply_combustion_transformation(df_weight, config)
    df_full = apply_electric_transformation(df_weight_combustion, config)
    return df_weight, df_weight_combustion, df_full


# ============================================================
# Prediction Core
# ============================================================

def run_predictions(
    pipeline,
    df_baseline,
    df_weight,
    df_weight_combustion,
    df_full,
    feature_columns
):
    logger.info("Generating simulation predictions...")

    X_base = ensure_features(df_baseline, feature_columns)
    X_weight = ensure_features(df_weight, feature_columns)
    X_wc = ensure_features(df_weight_combustion, feature_columns)
    X_full = ensure_features(df_full, feature_columns)

    pred_base = pipeline.predict(X_base)
    pred_weight = pipeline.predict(X_weight)
    pred_wc = pipeline.predict(X_wc)
    pred_full = pipeline.predict(X_full)

    results = df_baseline[
        ["race_id", "season", "circuit", "driver", "team", "lap_number", "lap_time_delta_from_session_median"]
    ].copy()

    results["predicted_baseline_delta"] = pred_base
    results["predicted_weight_delta"] = pred_weight
    results["predicted_weight_combustion_delta"] = pred_wc
    results["predicted_2026_delta"] = pred_full

    results["weight_effect"] = results["predicted_weight_delta"] - results["predicted_baseline_delta"]
    results["combustion_effect"] = results["predicted_weight_combustion_delta"] - results["predicted_weight_delta"]
    results["electric_effect"] = results["predicted_2026_delta"] - results["predicted_weight_combustion_delta"]
    results["lap_time_change_seconds"] = results["predicted_2026_delta"] - results["predicted_baseline_delta"]

    logger.info(
        f"Predictions generated for {len(results)} laps. "
        f"Mean lap time change: {results['lap_time_change_seconds'].mean():.4f}s"
    )

    return results


# ============================================================
# Aggregation
# ============================================================

def aggregate_circuit_results(results):
    logger.info("Aggregating results to circuit level...")

    circuit_agg = (
        results.groupby("circuit")
        .agg(
            mean_lap_change=("lap_time_change_seconds", "mean"),
            median_lap_change=("lap_time_change_seconds", "median"),
            std_lap_change=("lap_time_change_seconds", "std"),
            n_laps=("lap_time_change_seconds", "count"),
            n_drivers=("driver", "nunique"),
            weight_effect=("weight_effect", "mean"),
            combustion_effect=("combustion_effect", "mean"),
            electric_effect=("electric_effect", "mean"),
        )
        .round(4)
        .reset_index()
    )

    circuit_agg["power_plus_electric_effect"] = (
        circuit_agg["combustion_effect"] + circuit_agg["electric_effect"]
    ).round(4)

    def impact_label(delta):
        if delta < -0.5:
            return "Significantly Faster"
        elif delta < -0.1:
            return "Faster"
        elif delta <= 0.1:
            return "Neutral"
        elif delta <= 0.5:
            return "Slower"
        return "Significantly Slower"

    circuit_agg["impact_label"] = circuit_agg["mean_lap_change"].apply(impact_label)
    circuit_agg = circuit_agg.sort_values("mean_lap_change", ascending=True).reset_index(drop=True)

    logger.info("Effect decomposition complete:")
    for _, row in circuit_agg.iterrows():
        logger.info(
            f"{row['circuit']} Total: {row['mean_lap_change']:+.4f}s  "
            f"Weight: {row['weight_effect']:+.4f}s  "
            f"Combustion: {row['combustion_effect']:+.4f}s  "
            f"Electric: {row['electric_effect']:+.4f}s"
        )

    return circuit_agg


# ============================================================
# Fallback Simulation for Missing 2026 Circuits
# ============================================================

def simulate_missing_validation_circuit(
    pipeline,
    circuit_name,
    df_2026_real,
    feature_columns,
    config
):
    real_circuit = df_2026_real[df_2026_real["circuit"] == circuit_name].copy()

    if real_circuit.empty:
        return None

    logger.info(
        f"Running fallback simulation for {circuit_name} using available 2026 real feature rows "
        f"as proxy baseline."
    )

    # Treat real 2026 rows as proxy baseline just to get a comparable scenario estimate
    df_base = real_circuit.copy()
    df_weight, df_weight_combustion, df_full = apply_all_2026_transformations(df_base, config)

    fallback_results = run_predictions(
        pipeline,
        df_base,
        df_weight,
        df_weight_combustion,
        df_full,
        feature_columns
    )

    fallback_agg = aggregate_circuit_results(fallback_results)
    sim_row = fallback_agg[fallback_agg["circuit"] == circuit_name]

    if sim_row.empty:
        return None

    sim_result = sim_row.iloc[0].to_dict()

    # Fallback circuits use proxy baseline logic, so dampen the estimated effect
    # to avoid overstating confidence relative to true 2025-baseline simulations.
    fallback_damping = config.get("simulation_2026", {}).get("fallback_damping_factor", 0.5)

    sim_result["mean_lap_change"] *= fallback_damping
    sim_result["median_lap_change"] *= fallback_damping
    sim_result["weight_effect"] *= fallback_damping
    sim_result["combustion_effect"] *= fallback_damping
    sim_result["electric_effect"] *= fallback_damping
    sim_result["power_plus_electric_effect"] *= fallback_damping

    logger.info(
        f"Applied fallback damping factor {fallback_damping:.2f} to {circuit_name} "
        f"because simulation used proxy 2026 baseline rows."
    )

    return sim_result


# ============================================================
# Validation
# ============================================================

def run_validation_against_real_2026(
    pipeline,
    df_2026_real,
    circuit_agg,
    feature_columns,
    config
):
    if df_2026_real.empty:
        logger.warning("No real 2026 data available. Validation skipped.")
        return pd.DataFrame()

    logger.info("=" * 60)
    logger.info("PHASE 8B — VALIDATION AGAINST REAL 2026 DATA")
    logger.info("=" * 60)

    val_rows = []

    for circuit in sorted(df_2026_real["circuit"].unique()):
        sim_row = circuit_agg[circuit_agg["circuit"] == circuit]
        real_circuit = df_2026_real[df_2026_real["circuit"] == circuit]

        fallback_used = False

        if sim_row.empty:
            logger.warning(
                f"No simulation result found for {circuit}. "
                f"Attempting fallback simulation using available 2026 real rows."
            )
            fallback = simulate_missing_validation_circuit(
                pipeline,
                circuit,
                df_2026_real,
                feature_columns,
                config
            )
            if fallback is not None:
                sim_row = pd.DataFrame([fallback])
                fallback_used = True
                logger.info(f"Fallback simulation generated for {circuit}.")
            else:
                logger.warning(f"Fallback simulation also failed for {circuit}. Skipping.")
                continue

        actual_2026_delta = float(real_circuit["lap_time_delta_from_session_median"].median())
        simulated_change = float(sim_row["mean_lap_change"].iloc[0])
        error = simulated_change - actual_2026_delta

        logger.info(
            f"{circuit}: Actual 2026 delta={actual_2026_delta:+.4f}s | "
            f"Simulated change={simulated_change:+.4f}s | "
            f"Simulation error={error:+.4f}s"
            + (" | fallback_used=True" if fallback_used else "")
        )

        val_rows.append({
            "circuit": circuit,
            "actual_2026_delta": round(actual_2026_delta, 4),
            "simulated_change": round(simulated_change, 4),
            "simulation_error_seconds": round(error, 4),
            "fallback_used": fallback_used,
            "n_real_laps": int(len(real_circuit))
        })

    return pd.DataFrame(val_rows)


# ============================================================
# Plots
# ============================================================

def plot_circuit_impact_bar(circuit_agg):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")

    colors = ["#0067FF" if x < 0 else "#E8002D" for x in circuit_agg["mean_lap_change"]]

    ax.barh(
        circuit_agg["circuit"],
        circuit_agg["mean_lap_change"],
        xerr=circuit_agg["std_lap_change"],
        color=colors,
        alpha=0.85,
        edgecolor="white"
    )

    ax.axvline(0, color="black", linewidth=1.3)
    ax.set_xlabel("Predicted Lap Time Change (seconds)")
    ax.set_title("2026 Regulation Impact by Circuit")
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "simulation_circuit_impact.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Circuit impact bar chart saved: {path}")


def plot_effect_decomposition(circuit_agg):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")

    y = np.arange(len(circuit_agg))

    ax.barh(y - 0.25, circuit_agg["weight_effect"], height=0.22, label="Weight", color="#0067FF")
    ax.barh(y, circuit_agg["combustion_effect"], height=0.22, label="Combustion", color="#E8002D")
    ax.barh(y + 0.25, circuit_agg["electric_effect"], height=0.22, label="Electric", color="#00A19B")

    ax.set_yticks(y)
    ax.set_yticklabels(circuit_agg["circuit"])
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Predicted Contribution (seconds)")
    ax.set_title("2026 Regulation Effect Decomposition by Circuit")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "simulation_effect_decomposition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Effect decomposition chart saved: {path}")


def plot_validation_comparison(validation_df):
    if validation_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(validation_df))
    width = 0.35

    ax.bar(x - width / 2, validation_df["simulated_change"], width, label="Simulated")
    ax.bar(x + width / 2, validation_df["actual_2026_delta"], width, label="Actual 2026")

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(validation_df["circuit"])
    ax.set_ylabel("Lap Time Delta Change (seconds)")
    ax.set_title("Phase 8B — Simulation vs Real 2026")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "simulation_validation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Validation comparison plot saved: {path}")


# ============================================================
# Exports
# ============================================================

def export_simulation_results(circuit_agg, validation_df):
    results = {
        "generated_at": datetime.now().isoformat(),
        "baseline_season": 2025,
        "simulation_season": 2026,
        "circuits": circuit_agg.to_dict(orient="records"),
        "validation": validation_df.to_dict(orient="records"),
        "summary": {
            "avg_impact": float(circuit_agg["mean_lap_change"].mean()),
            "max_impact": float(circuit_agg["mean_lap_change"].max()),
            "min_impact": float(circuit_agg["mean_lap_change"].min()),
            "circuit_count": len(circuit_agg)
        }
    }

    output_path = os.path.join(MODELS_DIR, "simulation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Simulation results JSON saved: {output_path}")

    csv_path = os.path.join(PROCESSED_DIR, "simulation_2026_results.csv")
    circuit_agg.to_csv(csv_path, index=False)
    logger.info(f"Simulation results CSV saved: {csv_path}")


def generate_simulation_report(circuit_agg, validation_df):
    report_lines = [
        "=" * 70,
        "2026 F1 REGULATION SIMULATION — RESULTS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Baseline: 2025 F1 Season",
        "=" * 70,
        "",
        "REGULATION CHANGES MODELED",
        "1. Car weight reduction",
        "2. Combustion power reduction",
        "3. Electric torque benefit",
        "",
        "CIRCUIT-LEVEL PREDICTIONS",
        f"{'Circuit':<18}{'Change':>10}{'Uncertainty':>14}{'Weight':>10}{'Comb.':>10}{'Electric':>10}",
        "-" * 72
    ]

    for _, row in circuit_agg.iterrows():
        report_lines.append(
            f"{row['circuit']:<18}"
            f"{row['mean_lap_change']:>+10.4f}"
            f"{row['std_lap_change']:>14.4f}"
            f"{row['weight_effect']:>+10.4f}"
            f"{row['combustion_effect']:>+10.4f}"
            f"{row['electric_effect']:>+10.4f}"
        )

    if not validation_df.empty:
        report_lines += [
            "",
            "VALIDATION AGAINST REAL 2026",
            f"{'Circuit':<18}{'Actual':>10}{'Simulated':>12}{'Error':>10}{'Fallback':>12}",
            "-" * 62
        ]
        for _, row in validation_df.iterrows():
            report_lines.append(
                f"{row['circuit']:<18}"
                f"{row['actual_2026_delta']:>+10.4f}"
                f"{row['simulated_change']:>+12.4f}"
                f"{row['simulation_error_seconds']:>+10.4f}"
                f"{str(row['fallback_used']):>12}"
            )

    report = "\n".join(report_lines)
    print("\n" + report)

    report_path = os.path.join(
        PROCESSED_DIR,
        f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Simulation report saved: {report_path}")


# ============================================================
# Database Write
# ============================================================

def write_simulation_results(circuit_agg, conn, force=False):
    cursor = conn.cursor()

    if force:
        cursor.execute("DELETE FROM simulation_results")
        conn.commit()
        logger.info("Existing simulation results cleared")

    insert_sql = """
        INSERT INTO simulation_results (
            circuit,
            season_reference,
            predicted_2026_delta,
            lap_time_change_seconds,
            weight_effect_seconds,
            combustion_effect_seconds,
            electric_effect_seconds
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    written = 0
    for _, row in circuit_agg.iterrows():
        cursor.execute(
            insert_sql,
            (
                row["circuit"],
                2025,
                float(row["mean_lap_change"]),
                float(row["mean_lap_change"]),
                float(row["weight_effect"]),
                float(row["combustion_effect"]),
                float(row["electric_effect"]),
            )
        )
        written += 1

    conn.commit()
    cursor.close()
    logger.info(f"Wrote {written} simulation results to database")
    return written


# ============================================================
# Main Pipeline
# ============================================================

def run_simulation_pipeline(force=False):
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("2026 SIMULATION ENGINE STARTING")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    config = load_config()
    engine = get_sqlalchemy_engine()
    conn = get_mysql_connection()

    pipeline, feature_columns, metadata = load_model_and_features()

    df_2025 = load_2025_baseline(engine, feature_columns)
    df_2026_real = load_2026_real_data(engine, feature_columns)

    df_weight, df_weight_combustion, df_full = apply_all_2026_transformations(df_2025, config)

    results = run_predictions(
        pipeline,
        df_2025,
        df_weight,
        df_weight_combustion,
        df_full,
        feature_columns
    )

    circuit_agg = aggregate_circuit_results(results)

    logger.info("Generating simulation visualizations...")
    plot_circuit_impact_bar(circuit_agg)
    plot_effect_decomposition(circuit_agg)

    write_simulation_results(circuit_agg, conn, force=force)

    validation_df = run_validation_against_real_2026(
        pipeline,
        df_2026_real,
        circuit_agg,
        feature_columns,
        config
    )

    plot_validation_comparison(validation_df)

    export_simulation_results(circuit_agg, validation_df)
    generate_simulation_report(circuit_agg, validation_df)

    conn.close()

    end_time = datetime.now()
    logger.info("=" * 60)
    logger.info("SIMULATION ENGINE COMPLETE")
    logger.info(f"Duration: {end_time - start_time}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 F1 regulation simulation")
    parser.add_argument("--force", action="store_true", help="Clear and rewrite simulation results")
    args = parser.parse_args()

    run_simulation_pipeline(force=args.force)