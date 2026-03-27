"""
Dashboard Data Layer

Handles all data loading and preprocessing for the Streamlit
dashboard. Functions use Streamlit caching so data is loaded once
and served efficiently across interactions.
"""

import os
import sys
import json

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for path in [PROJECT_ROOT, SRC_DIR, APP_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from db_connection import get_sqlalchemy_engine
from logger import get_logger

logger = get_logger("dashboard")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "plots")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


@st.cache_data(ttl=3600, show_spinner=False)
def load_simulation_results() -> pd.DataFrame:
    """
    Loads circuit-level simulation results.
    Normalizes schema for dashboard use.
    """
    try:
        engine = get_sqlalchemy_engine()
        query = """
            SELECT
                s.circuit,
                s.season_reference,
                ROUND(s.lap_time_change_seconds, 4) AS lap_time_change_seconds,
                ROUND(s.weight_effect_seconds, 4) AS weight_effect_seconds,
                ROUND(s.combustion_effect_seconds, 4) AS combustion_effect_seconds,
                ROUND(s.electric_effect_seconds, 4) AS electric_effect_seconds
            FROM simulation_results s
            ORDER BY s.lap_time_change_seconds ASC
        """
        df = pd.read_sql(query, engine)
    except Exception as e:
        logger.warning(f"Database load failed for simulation_results: {e}")
        df = pd.DataFrame()

    if df.empty:
        csv_path = os.path.join(REPORTS_DIR, "simulation_2026_results.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df = df.rename(columns={
                    "mean_lap_change": "lap_time_change_seconds",
                    "weight_effect": "weight_effect_seconds",
                    "combustion_effect": "combustion_effect_seconds",
                    "electric_effect": "electric_effect_seconds",
                })
            except Exception as e:
                logger.warning(f"CSV fallback failed for simulation results: {e}")
                return pd.DataFrame()

    # ensure required cols exist
    required_cols = [
        "circuit",
        "lap_time_change_seconds",
        "weight_effect_seconds",
        "combustion_effect_seconds",
        "electric_effect_seconds",
        "power_sensitivity_score",
        "full_throttle_pct",
        "avg_corner_speed_kmh",
        "num_corners",
        "track_length_km",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # fill metadata defaults for safe UI rendering
    numeric_defaults = {
        "power_sensitivity_score": 0.5,
        "full_throttle_pct": 0.5,
        "avg_corner_speed_kmh": 180.0,
        "num_corners": 15,
        "track_length_km": 5.0,
        "weight_effect_seconds": 0.0,
        "combustion_effect_seconds": 0.0,
        "electric_effect_seconds": 0.0,
    }
    for col, val in numeric_defaults.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)

    df["lap_time_change_seconds"] = pd.to_numeric(df["lap_time_change_seconds"], errors="coerce")
    df = df.dropna(subset=["lap_time_change_seconds"]).sort_values("lap_time_change_seconds").reset_index(drop=True)

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_validation_results() -> pd.DataFrame:
    """
    Loads Phase 8B validation results from simulation_results.json
    since that is the most reliable artifact in your current pipeline.
    """
    json_path = os.path.join(MODELS_DIR, "simulation_results.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            validation = data.get("validation", [])
            if validation:
                df = pd.DataFrame(validation)

                rename_map = {
                    "actual_2026_delta": "actual_2026",
                    "simulation_error_seconds": "prediction_error",
                }
                df = df.rename(columns=rename_map)

                for col in ["circuit", "simulated_change", "actual_2026", "prediction_error", "fallback_used"]:
                    if col not in df.columns:
                        df[col] = pd.NA

                if "prediction_error" not in df.columns and {"simulated_change", "actual_2026"}.issubset(df.columns):
                    df["prediction_error"] = df["simulated_change"] - df["actual_2026"]

                if "prediction_error_pct" not in df.columns:
                    df["prediction_error_pct"] = pd.NA

                return df

        except Exception as e:
            logger.warning(f"Validation JSON load failed: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_gold_data_summary() -> pd.DataFrame:
    try:
        engine = get_sqlalchemy_engine()
        query = """
            SELECT
                circuit,
                season,
                data_split,
                COUNT(*) AS n_laps,
                COUNT(DISTINCT driver) AS n_drivers,
                ROUND(AVG(lap_time_delta_from_session_median), 4) AS mean_delta,
                ROUND(STD(lap_time_delta_from_session_median), 4) AS std_delta,
                ROUND(AVG(total_car_weight), 1) AS avg_car_weight,
                ROUND(AVG(effective_tire_grip), 4) AS avg_tire_grip
            FROM gold_modeling_data
            WHERE lap_time_delta_from_session_median IS NOT NULL
            GROUP BY circuit, season, data_split
            ORDER BY season, circuit
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.warning(f"Gold data summary load failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_driver_simulation_data() -> pd.DataFrame:
    try:
        engine = get_sqlalchemy_engine()
        query = """
            SELECT
                g.driver,
                g.team,
                g.circuit,
                g.season,
                ROUND(AVG(g.driver_skill_score), 4) AS skill_score,
                ROUND(AVG(g.total_car_weight), 1) AS avg_weight,
                ROUND(AVG(g.effective_tire_grip), 4) AS avg_grip,
                COUNT(*) AS n_laps
            FROM gold_modeling_data g
            WHERE g.season = 2025
              AND g.data_split = 'test'
            GROUP BY g.driver, g.team, g.circuit, g.season
            ORDER BY g.driver, g.circuit
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.warning(f"Driver data load failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_shap_summary() -> dict:
    json_path = os.path.join(MODELS_DIR, "shap_summary.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load SHAP summary: {e}")
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_mlflow_summary() -> dict:
    json_path = os.path.join(MODELS_DIR, "mlflow_summary.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load MLflow summary: {e}")
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_model_metadata() -> dict:
    json_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model metadata: {e}")
    return {}


def get_plot_path(filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    return path if os.path.exists(path) else ""


def get_impact_color(value: float) -> str:
    if value < -0.1:
        return "#0067FF"
    elif value > 0.1:
        return "#E8002D"
    return "#808080"


def get_impact_label(value: float) -> str:
    if value < -0.5:
        return "Significantly Faster"
    elif value < -0.1:
        return "Faster"
    elif value <= 0.1:
        return "Neutral"
    elif value <= 0.5:
        return "Slower"
    return "Significantly Slower"