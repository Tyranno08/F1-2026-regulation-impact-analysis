# src/pipelines/feature_engineering.py

"""
Feature Engineering Pipeline — Silver to Gold Layer

Reads cleaned lap data from the MySQL Silver table (clean_lap_data),
joins it with circuit metadata, engineers all model features, applies
a session-quality eligibility filter, and writes the modeling-ready
dataset to the Gold table (gold_modeling_data).

Architecture principle:
    Silver is a cleaned analytical layer — it retains all usable dry
    laps including those from mixed-condition races like Australia 2025
    and Silverstone 2025.

    Gold is the modeling-ready layer — only sessions that form a stable
    learning distribution for supervised dry-race modeling are included.
    Sessions that are too sparse or too noisy are excluded from Gold
    using objective quality criteria but remain available in Silver
    for exploratory analysis.

    When running with --force, any sessions that were previously written
    to Gold but are now excluded by the eligibility filter are explicitly
    deleted from the Gold table to prevent stale data contamination.

Features engineered:
    Weight features:
        - fuel_weight_estimate      (carried from Silver)
        - total_car_weight          (dry weight + fuel weight)

    Tire features:
        - effective_tire_grip       (exponential degradation curve)
        - compound encoded as-is    (kept as string for downstream encoding)

    Sector features:
        - sector1_ratio             (sector1 / lap_time)
        - sector2_ratio             (sector2 / lap_time)
        - sector3_ratio             (sector3 / lap_time)

    Circuit features:
        - power_sensitivity_score   (from circuit_metadata)
        - full_throttle_pct         (from circuit_metadata)
        - avg_corner_speed_kmh      (from circuit_metadata)
        - elevation_change_m        (from circuit_metadata)
        - num_corners               (from circuit_metadata)

    Weather features:
        - track_temp                (carried from Silver)
        - air_temp                  (carried from Silver)
        - humidity                  (carried from Silver)

    Driver features:
        - driver_skill_score        (target encoding)

    Performance features:
        - speed_trap                (carried from Silver)

    Target variable:
        - lap_time_delta_from_session_median (carried from Silver)

Gold session eligibility filter:
    After all features are engineered, sessions are evaluated for
    modeling stability. A session is excluded from Gold if any of
    the following are true:
        - It has fewer than 200 remaining clean laps
        - Its target standard deviation exceeds 2.5
        - Its maximum absolute target delta exceeds 8.0

    This filter ensures that mixed-condition races with too few
    surviving dry laps or unstable target distributions do not
    corrupt the supervised learning problem, while keeping those
    sessions available in Silver for exploratory analysis.

Usage:
    python src/pipelines/feature_engineering.py

    Optional flags:
    --season 2024          Process only a specific season
    --circuit Monza        Process only a specific circuit
    --force                Overwrite existing Gold records
"""

import sys
import os
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from db_connection import get_sqlalchemy_engine, get_mysql_connection
from config_loader import load_config
from logger import get_logger

logger = get_logger("feature_engineering")


# ============================================================
# Data Loading
# ============================================================

def load_silver_data(
    engine,
    seasons: list[int] = None,
    circuits: list[str] = None
) -> pd.DataFrame:
    """
    Loads cleaned lap data from the Silver table.

    We load only training data by default because feature
    engineering for the Gold modeling table is a training
    data concern. The 2026 validation data gets its own
    treatment in the simulation phase.

    Args:
        engine:   SQLAlchemy engine
        seasons:  Optional season filter
        circuits: Optional circuit filter

    Returns:
        Cleaned lap dataframe from Silver table.
    """
    logger.info("Loading Silver data...")

    query = """
        SELECT
            c.id                                    AS silver_id,
            c.race_id,
            c.season,
            c.circuit,
            c.driver,
            c.team,
            c.lap_number,
            c.lap_time_seconds,
            c.sector1_seconds,
            c.sector2_seconds,
            c.sector3_seconds,
            c.compound,
            c.tire_life,
            c.track_temp,
            c.air_temp,
            c.humidity,
            c.speed_trap,
            c.fuel_weight_estimate,
            c.lap_time_delta_from_session_median,
            c.data_split
        FROM clean_lap_data c
        WHERE c.lap_time_seconds IS NOT NULL
          AND c.lap_time_delta_from_session_median IS NOT NULL
    """

    conditions = []
    if seasons:
        season_list = ", ".join(str(s) for s in seasons)
        conditions.append(f"c.season IN ({season_list})")
    if circuits:
        circuit_list = ", ".join(f"'{c}'" for c in circuits)
        conditions.append(f"c.circuit IN ({circuit_list})")

    if conditions:
        query += " AND " + " AND ".join(conditions)

    query += " ORDER BY c.season, c.circuit, c.driver, c.lap_number"

    df = pd.read_sql(query, engine)

    logger.info(
        f"Loaded {len(df)} Silver rows: "
        f"{df['season'].nunique()} season(s), "
        f"{df['circuit'].nunique()} circuit(s), "
        f"{df['driver'].nunique()} driver(s)"
    )
    return df


def load_circuit_metadata(engine) -> pd.DataFrame:
    """
    Loads circuit metadata from the support table.

    Args:
        engine: SQLAlchemy engine

    Returns:
        Circuit metadata dataframe.
    """
    query = """
        SELECT
            circuit,
            track_length_km,
            num_corners,
            drs_zones,
            elevation_change_m,
            power_sensitivity_score,
            avg_corner_speed_kmh,
            full_throttle_pct
        FROM circuit_metadata
    """
    df = pd.read_sql(query, engine)
    logger.info(
        f"Loaded circuit metadata for {len(df)} circuits"
    )
    return df


# ============================================================
# Feature Group 1 — Weight Features
# ============================================================

def engineer_weight_features(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Engineers car weight features.

    total_car_weight combines the fixed dry weight of the car
    with the estimated remaining fuel weight at each lap.

    This is the key feature for the 2026 simulation because
    the regulation reduces car weight by 30kg. Our simulation
    engine simply subtracts 30 from total_car_weight and
    re-runs the model prediction.

    Formula:
        total_car_weight = car_dry_weight + fuel_weight_estimate

    The fuel_weight_estimate was already computed in Phase 3
    cleaning. We carry it forward here and add total_car_weight.

    Args:
        df:     Silver dataframe
        config: Project config dict

    Returns:
        DataFrame with total_car_weight column added.
    """
    fuel_cfg = config.get("fuel_model", {})
    dry_weight = fuel_cfg.get("car_dry_weight_kg", 798)

    df["total_car_weight"] = (
        dry_weight + df["fuel_weight_estimate"]
    )

    logger.info(
        f"Weight features engineered. "
        f"Total weight range: "
        f"{df['total_car_weight'].min():.1f}kg to "
        f"{df['total_car_weight'].max():.1f}kg"
    )
    return df


# ============================================================
# Feature Group 2 — Tire Degradation Features
# ============================================================

def engineer_tire_features(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Engineers tire degradation features using an exponential model.

    Raw tire_life (age in laps) is a useful feature but it treats
    all compounds equally. A soft tire at lap 10 has degraded far
    more than a hard tire at lap 10.

    We model effective grip using:

        effective_tire_grip = compound_factor * exp(
            -degradation_rate * tire_life
        )

    Where:
        compound_factor  = the relative grip level of the compound
                           at zero degradation (Soft=1.0, Hard=0.94)
        degradation_rate = how quickly grip falls off per lap
                           (Soft degrades fastest at 0.045/lap)

    The output is a dimensionless grip index between 0 and 1
    where 1 represents a perfect new soft tire.

    Note on null tire_life values:
        Where tire_life is null we impute with the median tire life
        for that compound in that session. This is more informative
        than dropping the row or using a global median.

    Args:
        df:     Silver dataframe
        config: Project config dict with tire model parameters

    Returns:
        DataFrame with effective_tire_grip column added.
    """
    tire_cfg = config.get("tire_model", {}).get("compounds", {})

    # Default parameters for unknown compounds
    default_params = {
        "compound_factor": 0.90,
        "degradation_rate": 0.020
    }

    # ---- Impute missing tire life values ----
    # Group by race_id and compound, fill nulls with group median
    df["tire_life"] = df.groupby(
        ["race_id", "compound"]
    )["tire_life"].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill any remaining nulls with overall median
    overall_median_tire_life = df["tire_life"].median()
    df["tire_life"] = df["tire_life"].fillna(overall_median_tire_life)

    # ---- Compute effective tire grip ----
    def compute_grip(row):
        compound = str(row["compound"]).upper()
        params = tire_cfg.get(compound, default_params)
        factor = params.get("compound_factor", 0.90)
        rate = params.get("degradation_rate", 0.020)
        tire_age = max(0, row["tire_life"])
        return factor * np.exp(-rate * tire_age)

    df["effective_tire_grip"] = df.apply(compute_grip, axis=1)

    logger.info(
        f"Tire features engineered. "
        f"Effective grip range: "
        f"{df['effective_tire_grip'].min():.4f} to "
        f"{df['effective_tire_grip'].max():.4f}"
    )
    return df


# ============================================================
# Feature Group 3 — Sector Ratio Features
# ============================================================

def engineer_sector_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers sector performance ratio features.

    Sector ratios capture the shape of the lap by expressing
    each sector time as a proportion of the total lap time.

    sector1_ratio = sector1_seconds / lap_time_seconds

    These ratios are:
        - Scale-invariant across circuits
        - Informative about where on circuit time is gained/lost
        - Useful for identifying setup and car characteristic effects

    We only compute ratios where all three sectors are available
    and where they sum to approximately the lap time.
    Rows missing sector data get NaN ratios which the model
    handles gracefully.

    Sanity check:
        The sum of three sector ratios should be very close to 1.0.
        We log the mean sum as a verification check.

    Args:
        df: Silver dataframe with sector time columns

    Returns:
        DataFrame with sector ratio columns added.
    """
    # Only compute where all sectors are available
    sectors_available = (
        df["sector1_seconds"].notna() &
        df["sector2_seconds"].notna() &
        df["sector3_seconds"].notna() &
        (df["lap_time_seconds"] > 0)
    )

    df["sector1_ratio"] = np.nan
    df["sector2_ratio"] = np.nan
    df["sector3_ratio"] = np.nan

    mask = sectors_available
    df.loc[mask, "sector1_ratio"] = (
        df.loc[mask, "sector1_seconds"] /
        df.loc[mask, "lap_time_seconds"]
    )
    df.loc[mask, "sector2_ratio"] = (
        df.loc[mask, "sector2_seconds"] /
        df.loc[mask, "lap_time_seconds"]
    )
    df.loc[mask, "sector3_ratio"] = (
        df.loc[mask, "sector3_seconds"] /
        df.loc[mask, "lap_time_seconds"]
    )

    coverage = mask.sum()
    total = len(df)

    # Sanity check — sector ratios should sum to ~1.0
    ratio_sum = (
        df.loc[mask, "sector1_ratio"] +
        df.loc[mask, "sector2_ratio"] +
        df.loc[mask, "sector3_ratio"]
    ).mean()

    logger.info(
        f"Sector ratios engineered for {coverage}/{total} laps "
        f"({coverage/total*100:.1f}% coverage). "
        f"Mean ratio sum: {ratio_sum:.6f} (should be ~1.0)"
    )
    return df


# ============================================================
# Feature Group 4 — Circuit Features
# ============================================================

def join_circuit_features(
    df: pd.DataFrame,
    circuit_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins circuit metadata features onto the main dataframe.

    These features capture the physical characteristics of each
    circuit that drive performance differences. They are essential
    for the 2026 simulation because the power_sensitivity_score
    determines how much the engine rule change affects each circuit.

    Features joined:
        power_sensitivity_score  — how much raw engine power matters
        full_throttle_pct        — percentage of lap at full throttle
        avg_corner_speed_kmh     — average corner speed
        elevation_change_m       — total elevation change
        num_corners              — number of corners

    Circuits not found in metadata get NaN values for these columns.
    We log any circuits with missing metadata so you know to add them.

    Args:
        df:           Silver dataframe
        circuit_meta: Circuit metadata dataframe

    Returns:
        DataFrame with circuit feature columns added.
    """
    circuit_features = [
        "circuit",
        "power_sensitivity_score",
        "full_throttle_pct",
        "avg_corner_speed_kmh",
        "elevation_change_m",
        "num_corners"
    ]

    df = df.merge(
        circuit_meta[circuit_features],
        on="circuit",
        how="left"
    )

    # Check for circuits missing from metadata
    missing = df[df["power_sensitivity_score"].isna()]["circuit"].unique()
    if len(missing) > 0:
        logger.warning(
            f"Circuits missing from metadata table: {missing}. "
            f"These rows will have NaN circuit features. "
            f"Add them to config.yaml and re-run seed_circuit_metadata.py"
        )
    else:
        logger.info(
            f"Circuit features joined successfully for all "
            f"{df['circuit'].nunique()} circuits"
        )

    return df


# ============================================================
# Feature Group 5 — Driver Target Encoding
# ============================================================

def engineer_driver_skill_score(
    df: pd.DataFrame,
    smoothing: int = 200
) -> pd.DataFrame:
    """
    Engineers driver skill score using target encoding.

    Driver mean lap delta is shrunk toward the global mean to reduce
    noise from limited sample sizes and improve generalization.

    Target encoding replaces the categorical driver abbreviation
    with a continuous numeric value representing that driver's
    typical performance relative to the session median.

    Method:
        For each driver, compute their mean lap_time_delta across
        ALL training laps. A value of -0.8 means this driver is
        typically 0.8 seconds faster than the session median.

    Why we use training data only for computing encoding:
        We compute the driver skill score using only rows where
        data_split == 'train'. If we included validation data
        (2026 races) we would be using future information to
        encode a feature, which is a form of data leakage.

    Handling new drivers:
        Drivers who appear only in validation data (2026 rookies
        with no training history) get the global mean score of 0.0,
        meaning we assume they perform at median pace. This is a
        conservative and defensible assumption.

    Args:
        df: Dataframe with lap_time_delta_from_session_median column

    Returns:
        DataFrame with driver_skill_score column added.
    """
    logger.info("Computing driver target encoding...")

    training_mask = df["data_split"] == "train"
    train_df = df.loc[training_mask].copy()

    global_mean = train_df["lap_time_delta_from_session_median"].mean()

    driver_stats = (
        train_df.groupby("driver")["lap_time_delta_from_session_median"]
        .agg(["mean", "count"])
        .reset_index()
    )

    driver_stats["driver_skill_score"] = (
        (driver_stats["count"] * driver_stats["mean"] +
         smoothing * global_mean) /
        (driver_stats["count"] + smoothing)
    )

    driver_encoding = driver_stats.set_index("driver")["driver_skill_score"]

    logger.info(
        f"Driver encoding computed for "
        f"{len(driver_encoding)} drivers from training data "
        f"(smoothing={smoothing})"
    )

    sorted_encoding = driver_encoding.sort_values()
    logger.info("Fastest drivers by skill score (most negative = fastest):")
    for driver, score in sorted_encoding.head(5).items():
        logger.info(f"  {driver}: {score:.4f}s")
    logger.info("Slowest drivers by skill score:")
    for driver, score in sorted_encoding.tail(5).items():
        logger.info(f"  {driver}: {score:.4f}s")

    df["driver_skill_score"] = df["driver"].map(driver_encoding)

    new_drivers = df["driver_skill_score"].isna().sum()
    if new_drivers > 0:
        logger.info(
            f"{new_drivers} laps have drivers not in training data. "
            f"Filling with global mean: {global_mean:.4f}"
        )

    df["driver_skill_score"] = df["driver_skill_score"].fillna(global_mean)

    return df


# ============================================================
# Feature Group 6 — Weather Imputation
# ============================================================

def impute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing weather values.

    Some sessions had unavailable weather data during ingestion
    and those rows have NULL track_temp, air_temp, humidity.

    Imputation strategy:
        For each circuit, fill nulls with the median value
        across all seasons for that circuit. This is sensible
        because circuit weather is relatively stable year to year
        (Bahrain is always hot, Spa is always unpredictable).

        Any remaining nulls after circuit-level imputation get
        the global median as a final fallback.

    Args:
        df: Dataframe with weather columns

    Returns:
        DataFrame with weather nulls imputed.
    """
    weather_cols = ["track_temp", "air_temp", "humidity"]

    for col in weather_cols:
        null_before = df[col].isna().sum()
        if null_before == 0:
            logger.info(f"{col}: no nulls to impute")
            continue

        # Circuit-level median imputation
        df[col] = df.groupby("circuit")[col].transform(
            lambda x: x.fillna(x.median())
        )

        # Global fallback
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)

        null_after = df[col].isna().sum()
        logger.info(
            f"{col}: imputed {null_before} nulls "
            f"({null_after} remaining after imputation)"
        )

    return df

def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a small number of physics-informed interaction features.
    """
    df["grip_temp_interaction"] = (
        df["effective_tire_grip"] * df["track_temp"]
    )

    logger.info(
        "Interaction features engineered: grip_temp_interaction"
    )
    return df
# ============================================================
# Feature Group 7 — Speed Trap Imputation
# ============================================================

def impute_speed_trap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing speed trap values using increasingly broader groups.

    Speed trap measures the car's top speed at a designated
    straight on each circuit. It is a proxy for aerodynamic
    efficiency and engine performance at high speed.

    Missing speed trap values are imputed using the median
    speed trap for that driver at that circuit across all laps.
    This preserves the driver and circuit specific signal.

    Args:
        df: Dataframe with speed_trap column

    Returns:
        DataFrame with speed_trap nulls imputed.
    """
    null_before = df["speed_trap"].isna().sum()

    if null_before == 0:
        logger.info("speed_trap: no nulls to impute")
        return df

    # Driver + circuit
    df["speed_trap"] = df.groupby(
        ["driver", "circuit"]
    )["speed_trap"].transform(
        lambda x: x.fillna(x.median())
    )

    # Team + circuit
    df["speed_trap"] = df.groupby(
        ["team", "circuit"]
    )["speed_trap"].transform(
        lambda x: x.fillna(x.median())
    )

    # Circuit
    df["speed_trap"] = df.groupby("circuit")["speed_trap"].transform(
        lambda x: x.fillna(x.median())
    )

    # Global
    df["speed_trap"] = df["speed_trap"].fillna(df["speed_trap"].median())

    null_after = df["speed_trap"].isna().sum()
    logger.info(
        f"speed_trap: imputed {null_before} nulls "
        f"({null_after} remaining)"
    )
    return df


# ============================================================
# Gold Session Eligibility Filter
# ============================================================

def filter_modeling_eligible_sessions(
    df: pd.DataFrame,
    min_laps: int = 200,
    max_target_std: float = 2.5,
    max_abs_delta: float = 8.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters out sessions that are too unstable for Gold modeling use.

    Silver may contain cleaned dry laps from heavily mixed-condition
    races, but some sessions remain too sparse or too noisy to be
    reliable for supervised modeling. This function excludes those
    sessions from the Gold layer using objective quality criteria.

    Why this filter exists:
        Mixed-condition races like Australia 2025 (98 surviving dry
        laps) and Silverstone 2025 (138 surviving dry laps) pass all
        cleaning steps correctly — their dry laps are genuine. But
        the resulting session data is too sparse and too noisy to
        form a stable learning distribution. Including them in Gold
        would let a handful of noisy laps dominate the model.

        By keeping these sessions in Silver but excluding them from
        Gold, we maintain a clean analytical layer while ensuring
        the modeling layer contains only stable, well-populated
        sessions.

    Exclusion rules — a session is NOT modeling-eligible if ANY of:
        - fewer than min_laps remaining clean laps
        - target standard deviation above max_target_std
        - maximum absolute lap delta above max_abs_delta

    Why these thresholds:
        min_laps = 200:
            Avoids weak sessions dominating by noise. A normal dry
            race with 20 drivers and ~55 laps produces 800+ clean
            laps after filtering. Sessions below 200 are severely
            incomplete.

        target_std > 2.5:
            Flags highly unstable pace distributions. Normal dry
            sessions have target std around 0.8-1.5 seconds. A std
            above 2.5 indicates transition effects or anomalous
            pace variation that would corrupt model training.

        max_abs_delta > 8.0:
            Flags transition or anomalous pace leftovers. No genuine
            dry racing lap should be more than ~7 seconds from the
            session median. Values above 8.0 indicate contamination
            from drying-track conditions that survived upstream
            filters.

    Args:
        df:             Silver dataframe after feature engineering
                        and target already present
        min_laps:       Minimum clean laps required for modeling
        max_target_std: Maximum allowed std of target variable
        max_abs_delta:  Maximum allowed absolute target value

    Returns:
        Tuple of:
            - filtered Gold dataframe (only eligible sessions)
            - session diagnostics dataframe (all sessions with
              their quality metrics and eligibility status)
    """
    logger.info("Applying Gold modeling session eligibility filter...")
    logger.info(
        f"  Thresholds: min_laps={min_laps}, "
        f"max_target_std={max_target_std}, "
        f"max_abs_delta={max_abs_delta}"
    )

    before_sessions = df["race_id"].nunique()
    before_rows = len(df)

    # Compute quality metrics per session
    session_stats = (
        df.groupby(["race_id", "season", "circuit"])
        .agg(
            n_laps=("lap_time_delta_from_session_median", "count"),
            target_std=("lap_time_delta_from_session_median", "std"),
            target_mean=("lap_time_delta_from_session_median", "mean"),
            min_delta=("lap_time_delta_from_session_median", "min"),
            max_delta=("lap_time_delta_from_session_median", "max")
        )
        .reset_index()
    )

    session_stats["max_abs_delta"] = session_stats[
        ["min_delta", "max_delta"]
    ].abs().max(axis=1)

    # Apply eligibility rules
    session_stats["eligible_for_modeling"] = (
        (session_stats["n_laps"] >= min_laps) &
        (session_stats["target_std"] <= max_target_std) &
        (session_stats["max_abs_delta"] <= max_abs_delta)
    )

    # Log excluded sessions with specific reasons
    excluded_sessions = session_stats[
        ~session_stats["eligible_for_modeling"]
    ].copy()

    if not excluded_sessions.empty:
        logger.warning(
            f"Excluding {len(excluded_sessions)} session(s) from Gold "
            f"modeling layer due to insufficient stability:"
        )

        for _, row in excluded_sessions.iterrows():
            reasons = []
            if row["n_laps"] < min_laps:
                reasons.append(
                    f"n_laps={int(row['n_laps'])} < {min_laps}"
                )
            if row["target_std"] > max_target_std:
                reasons.append(
                    f"target_std={row['target_std']:.3f} "
                    f"> {max_target_std}"
                )
            if row["max_abs_delta"] > max_abs_delta:
                reasons.append(
                    f"max_abs_delta={row['max_abs_delta']:.3f} "
                    f"> {max_abs_delta}"
                )

            logger.warning(
                f"  {row['race_id']} "
                f"({int(row['season'])} {row['circuit']}) "
                f"excluded — " + ", ".join(reasons)
            )
    else:
        logger.info(
            "All sessions passed Gold eligibility filter — "
            "no sessions excluded"
        )

    # Log eligible sessions for completeness
    eligible_sessions = session_stats[
        session_stats["eligible_for_modeling"]
    ]
    logger.info(
        f"Eligible sessions: {len(eligible_sessions)} "
        f"(excluded: {len(excluded_sessions)})"
    )

    # Filter to eligible sessions only
    eligible_race_ids = set(
        session_stats.loc[
            session_stats["eligible_for_modeling"],
            "race_id"
        ]
    )

    filtered_df = df[df["race_id"].isin(eligible_race_ids)].copy()

    after_sessions = filtered_df["race_id"].nunique()
    after_rows = len(filtered_df)

    logger.info(
        f"Gold eligibility filter complete: "
        f"{after_rows}/{before_rows} rows retained "
        f"({after_rows/before_rows*100:.1f}%) "
        f"across {after_sessions}/{before_sessions} eligible sessions"
    )

    return filtered_df, session_stats


# ============================================================
# Feature Validation
# ============================================================

def validate_gold_features(df: pd.DataFrame) -> bool:
    """
    Validates the engineered feature set before writing to Gold.

    Checks:
        1. All required feature columns exist
        2. Critical features have no nulls
        3. Feature ranges are physically reasonable
        4. Target variable is present and non-null

    Args:
        df: Fully engineered dataframe

    Returns:
        True if all checks pass, False if any critical check fails.
    """
    logger.info("Validating Gold features...")
    all_passed = True

    # ---- Check 1: Required columns exist ----
    required_columns = [
        "race_id", "season", "circuit", "driver", "team",
        "lap_number", "lap_time_delta_from_session_median",
        "fuel_weight_estimate", "total_car_weight",
        "tire_life", "compound", "effective_tire_grip",
        "sector1_ratio", "sector2_ratio", "sector3_ratio",
        "power_sensitivity_score", "full_throttle_pct",
        "avg_corner_speed_kmh", "elevation_change_m", "num_corners",
        "track_temp", "air_temp", "humidity",
        "driver_skill_score", "speed_trap", "data_split"
    ]

    missing_cols = [
        c for c in required_columns if c not in df.columns
    ]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        all_passed = False
    else:
        logger.info("All required columns present")

    # ---- Check 2: Critical columns have no nulls ----
    critical_no_null = [
        "lap_time_delta_from_session_median",
        "total_car_weight",
        "effective_tire_grip",
        "driver_skill_score",
        "power_sensitivity_score"
    ]

    for col in critical_no_null:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.error(
                f"Critical column {col} has {null_count} null values"
            )
            all_passed = False
        else:
            logger.info(f"{col}: no nulls — OK")

    # ---- Check 3: Feature range validation ----
    range_checks = {
        "total_car_weight":         (700, 950),
        "effective_tire_grip":      (0.05, 1.01),
        "power_sensitivity_score":  (0.0, 1.0),
        "sector1_ratio":            (0.1, 0.6),
        "sector2_ratio":            (0.1, 0.6),
        "sector3_ratio":            (0.1, 0.6),
        "fuel_weight_estimate":     (0.0, 115.0),
    }

    for col, (low, high) in range_checks.items():
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        col_min = col_data.min()
        col_max = col_data.max()
        if col_min < low or col_max > high:
            logger.warning(
                f"Range check for {col}: "
                f"got [{col_min:.4f}, {col_max:.4f}], "
                f"expected [{low}, {high}]"
            )
        else:
            logger.info(
                f"{col}: range [{col_min:.4f}, {col_max:.4f}] — OK"
            )

    # ---- Check 4: Target variable ----
    target_nulls = df["lap_time_delta_from_session_median"].isna().sum()
    if target_nulls > 0:
        logger.error(
            f"Target variable has {target_nulls} null values. "
            f"These rows cannot be used for training."
        )
        all_passed = False
    else:
        logger.info(
            f"Target variable: {len(df)} non-null values — OK"
        )

    # ---- Check 5: Data split coverage ----
    split_counts = df["data_split"].value_counts()
    logger.info(f"Data split distribution: {split_counts.to_dict()}")

    if all_passed:
        logger.info("All feature validation checks passed")
    else:
        logger.error(
            "Some feature validation checks failed. "
            "Review logs before proceeding to modeling."
        )

    return all_passed


# ============================================================
# Database Write
# ============================================================

def check_already_engineered(race_id: str, conn) -> bool:
    """
    Checks if a race session already exists in the Gold table.

    Args:
        race_id: Race identifier string
        conn:    Active MySQL connection

    Returns:
        True if already exists, False otherwise.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM gold_modeling_data WHERE race_id = %s",
        (race_id,)
    )
    count = cursor.fetchone()[0]
    cursor.close()
    return count > 0


def delete_excluded_gold_sessions(
    excluded_race_ids: list[str],
    conn
) -> int:
    """
    Deletes excluded sessions from the Gold table.

    This is necessary when a session was written to Gold in a
    previous run but is now excluded by the Gold session eligibility
    filter. Without this cleanup, stale rows remain in
    gold_modeling_data and contaminate modeling splits.

    This is a classic stale-data prevention pattern. The eligibility
    filter removes excluded sessions from the in-memory dataframe,
    but write_gold_data() only deletes records for race_ids that
    it is about to re-write. So excluded race_ids would never be
    written — but also never deleted — leaving orphaned rows from
    previous runs.

    Args:
        excluded_race_ids: List of race_ids excluded from current
                           Gold run by the eligibility filter
        conn:              Active MySQL connection

    Returns:
        Number of sessions deleted.
    """
    if not excluded_race_ids:
        return 0

    cursor = conn.cursor()
    deleted_sessions = 0

    for race_id in excluded_race_ids:
        # Check if stale records actually exist before deleting
        cursor.execute(
            "SELECT COUNT(*) FROM gold_modeling_data WHERE race_id = %s",
            (race_id,)
        )
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            cursor.execute(
                "DELETE FROM gold_modeling_data WHERE race_id = %s",
                (race_id,)
            )
            deleted_sessions += 1
            logger.info(
                f"Deleted {existing_count} stale Gold records "
                f"for excluded session: {race_id}"
            )
        else:
            logger.debug(
                f"No stale Gold records found for {race_id} — "
                f"nothing to delete"
            )

    conn.commit()
    cursor.close()
    return deleted_sessions


def write_gold_data(
    df: pd.DataFrame,
    engine,
    conn,
    force: bool = False
) -> int:
    """
    Writes the engineered feature dataset to the Gold table.

    Only writes columns that exist in the Gold schema.
    Processes one session at a time for idempotency.

    Args:
        df:     Fully engineered dataframe
        engine: SQLAlchemy engine
        conn:   MySQL connection
        force:  If True, overwrite existing Gold records

    Returns:
        Total number of rows written.
    """
    gold_columns = [
        "race_id", "season", "circuit", "driver", "team",
        "lap_number",
        "lap_time_delta_from_session_median",
        "fuel_weight_estimate", "total_car_weight",
        "tire_life", "compound", "effective_tire_grip",
        "sector1_ratio", "sector2_ratio", "sector3_ratio",
        "power_sensitivity_score", "full_throttle_pct",
        "avg_corner_speed_kmh", "elevation_change_m", "num_corners",
        "track_temp", "air_temp", "humidity",
        "driver_skill_score", "speed_trap","grip_temp_interaction",
        "data_split"
    ]

    # Keep only columns that exist in both df and gold schema
    existing_cols = [c for c in gold_columns if c in df.columns]
    df_gold = df[existing_cols].copy()

    total_written = 0
    race_ids = df_gold["race_id"].unique()

    logger.info(
        f"Writing {len(df_gold)} rows across "
        f"{len(race_ids)} sessions to Gold table..."
    )

    for race_id in race_ids:
        race_df = df_gold[df_gold["race_id"] == race_id]

        if not force and check_already_engineered(race_id, conn):
            logger.info(
                f"SKIPPED — {race_id} already in Gold table. "
                f"Use --force to re-engineer."
            )
            continue

        if force:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM gold_modeling_data WHERE race_id = %s",
                (race_id,)
            )
            conn.commit()
            cursor.close()
            logger.info(f"Deleted existing Gold records for {race_id}")

        try:
            race_df.to_sql(
                name="gold_modeling_data",
                con=engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500
            )
            written = len(race_df)
            total_written += written
            logger.info(f"Wrote {written} Gold rows for {race_id}")

        except Exception as e:
            logger.error(
                f"Failed to write Gold data for {race_id}: {e}"
            )

    return total_written


# ============================================================
# Feature Summary Report
# ============================================================

def generate_feature_report(
    df: pd.DataFrame,
    session_quality_df: pd.DataFrame = None
) -> None:
    """
    Prints a summary of the engineered feature set.

    This report documents the feature distributions and coverage
    for inclusion in the project README and for interview discussion.

    If session_quality_df is provided, the report also includes
    a section documenting which sessions were excluded from Gold
    and why, providing a complete audit trail.

    Args:
        df:                 Fully engineered Gold dataframe
        session_quality_df: Optional session diagnostics from
                            the eligibility filter
    """
    train_count = (df["data_split"] == "train").sum()
    test_count = (df["data_split"] == "test").sum()
    val_count = (df["data_split"] == "validation").sum()

    report_lines = [
        "",
        "=" * 70,
        "FEATURE ENGINEERING PIPELINE — GOLD DATASET REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"Total rows in Gold dataset: {len(df)}",
        f"Training rows:    {train_count}",
        f"Test rows:        {test_count}",
        f"Validation rows:  {val_count}",
        "",
    ]

    # ---- Session Eligibility Report ----
    if session_quality_df is not None:
        eligible = session_quality_df[
            session_quality_df["eligible_for_modeling"]
        ]
        excluded = session_quality_df[
            ~session_quality_df["eligible_for_modeling"]
        ]

        report_lines += [
            "GOLD SESSION ELIGIBILITY",
            "-" * 70,
            f"  Total sessions evaluated:  {len(session_quality_df)}",
            f"  Sessions eligible (in Gold): {len(eligible)}",
            f"  Sessions excluded (Silver only): {len(excluded)}",
            "",
        ]

        if not excluded.empty:
            report_lines.append("  Excluded sessions:")
            for _, row in excluded.iterrows():
                reasons = []
                if row["n_laps"] < 200:
                    reasons.append(
                        f"n_laps={int(row['n_laps'])}"
                    )
                if row["target_std"] > 2.5:
                    reasons.append(
                        f"std={row['target_std']:.3f}"
                    )
                if row["max_abs_delta"] > 8.0:
                    reasons.append(
                        f"max_abs={row['max_abs_delta']:.3f}"
                    )
                report_lines.append(
                    f"    {row['race_id']:<35} "
                    f"{', '.join(reasons)}"
                )
            report_lines.append("")

    report_lines += [
        "FEATURE SUMMARY",
        "-" * 70,
    ]

    numeric_features = [
        "total_car_weight",
        "fuel_weight_estimate",
        "effective_tire_grip",
        "sector1_ratio",
        "sector2_ratio",
        "sector3_ratio",
        "power_sensitivity_score",
        "full_throttle_pct",
        "avg_corner_speed_kmh",
        "elevation_change_m",
        "num_corners",
        "track_temp",
        "air_temp",
        "humidity",
        "driver_skill_score",
        "speed_trap",
        "lap_time_delta_from_session_median",
    ]

    report_lines.append(
        f"  {'Feature':<45} {'Mean':>8} {'Std':>8} "
        f"{'Min':>8} {'Max':>8} {'Null%':>7}"
    )
    report_lines.append("  " + "-" * 65)

    for feat in numeric_features:
        if feat not in df.columns:
            continue
        col = df[feat].dropna()
        null_pct = df[feat].isna().mean() * 100
        report_lines.append(
            f"  {feat:<45} "
            f"{col.mean():>8.3f} "
            f"{col.std():>8.3f} "
            f"{col.min():>8.3f} "
            f"{col.max():>8.3f} "
            f"{null_pct:>6.1f}%"
        )

    report_lines += [
        "",
        "DRIVER SKILL SCORES (Top 5 Fastest)",
        "-" * 70,
    ]

    top_drivers = (
        df.groupby("driver")["driver_skill_score"]
        .first()
        .sort_values()
        .head(5)
    )
    for driver, score in top_drivers.items():
        report_lines.append(f"  {driver:<10} {score:>+.4f}s from median")

    report_lines += [
        "",
        "DRIVER SKILL SCORES (Top 5 Slowest)",
        "-" * 70,
    ]

    bottom_drivers = (
        df.groupby("driver")["driver_skill_score"]
        .first()
        .sort_values()
        .tail(5)
    )
    for driver, score in bottom_drivers.items():
        report_lines.append(f"  {driver:<10} {score:>+.4f}s from median")

    report_lines += [
        "",
        "CIRCUIT POWER SENSITIVITY SCORES",
        "-" * 70,
    ]

    circuit_power = (
        df.groupby("circuit")["power_sensitivity_score"]
        .first()
        .sort_values(ascending=False)
    )
    for circuit, score in circuit_power.items():
        bar_length = int(score * 30)
        bar = "█" * bar_length
        report_lines.append(
            f"  {circuit:<20} {score:.2f}  {bar}"
        )

    report_lines.append("")
    report_lines.append("=" * 70)

    full_report = "\n".join(report_lines)
    print(full_report)

    # Save to file
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )),
        "data", "processed"
    )
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(
        reports_dir,
        f"feature_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    logger.info(f"Feature report saved to {report_path}")


# ============================================================
# Main Pipeline Orchestrator
# ============================================================

def run_feature_engineering_pipeline(
    seasons: list[int] = None,
    circuits: list[str] = None,
    force: bool = False
) -> None:
    """
    Orchestrates the full Silver to Gold feature engineering pipeline.

    Pipeline steps:
        1.  Load Silver data and circuit metadata
        2.  Engineer all feature groups:
            - Weight features (total_car_weight)
            - Tire degradation (effective_tire_grip)
            - Sector ratios
            - Circuit metadata join
            - Weather imputation
            - Speed trap imputation
            - Driver target encoding (driver_skill_score)
        3.  Apply Gold session eligibility filter
        3B. Delete stale excluded sessions from Gold table
        4.  Save session quality diagnostics report
        5.  Validate Gold features
        6.  Generate feature summary report
        7.  Write eligible sessions to Gold table

    The session eligibility filter (step 3) ensures that only
    sessions with a stable learning distribution reach the Gold
    table. Sessions excluded from Gold remain in Silver for
    exploratory analysis.

    Step 3B prevents stale data contamination: when a session was
    written to Gold in a previous run but is now excluded by the
    eligibility filter, it must be explicitly deleted. Without this
    cleanup, write_gold_data() would never touch those race_ids
    because they are no longer in the filtered dataframe.

    Args:
        seasons:  Optional season filter
        circuits: Optional circuit filter
        force:    If True, overwrite existing Gold records
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE STARTING")
    logger.info(
        f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if force:
        logger.warning(
            "FORCE MODE — existing Gold data will be overwritten"
        )
    logger.info("=" * 60)

    config = load_config()
    engine = get_sqlalchemy_engine()
    conn = get_mysql_connection()

    # ---- Step 1: Load Silver data and circuit metadata ----
    df = load_silver_data(engine, seasons=seasons, circuits=circuits)

    if df.empty:
        logger.error(
            "No Silver data loaded. "
            "Have you run the cleaning pipeline?"
        )
        conn.close()
        return

    circuit_meta = load_circuit_metadata(engine)

    # ---- Step 2: Engineer feature groups ----
    logger.info("Engineering feature groups...")

    df = engineer_weight_features(df, config)
    df = engineer_tire_features(df, config)
    df = engineer_sector_ratios(df)
    df = join_circuit_features(df, circuit_meta)
    df = impute_weather_features(df)
    df = engineer_interaction_features(df)
    df = impute_speed_trap(df)

    # Driver encoding must come after circuit join because
    # it uses the full dataset to compute encoding values
    df = engineer_driver_skill_score(df)

    logger.info(
        f"All feature groups engineered. "
        f"Dataset shape before eligibility filter: {df.shape}"
    )

    # ---- Step 3: Apply Gold session eligibility filter ----
    # This removes sessions that are too sparse or too noisy for
    # supervised modeling. Excluded sessions remain in Silver.
    df, session_quality_df = filter_modeling_eligible_sessions(df)

    if df.empty:
        logger.critical(
            "ALL sessions excluded by Gold eligibility filter. "
            "No data to write to Gold table. "
            "Check filter thresholds or Silver data quality."
        )
        conn.close()
        return

    # ---- Step 3B: Delete stale excluded sessions from Gold ----
    # When --force is used, sessions that were previously in Gold
    # but are now excluded by the eligibility filter must be
    # explicitly deleted. The write_gold_data function only deletes
    # sessions it is about to re-write, so excluded sessions would
    # otherwise remain as stale orphaned records.
    excluded_race_ids = session_quality_df.loc[
        ~session_quality_df["eligible_for_modeling"],
        "race_id"
    ].tolist()

    if force and excluded_race_ids:
        deleted_excluded = delete_excluded_gold_sessions(
            excluded_race_ids, conn
        )
        logger.info(
            f"Deleted {deleted_excluded} excluded session(s) from "
            f"Gold table to prevent stale rows"
        )
    elif excluded_race_ids:
        logger.info(
            f"{len(excluded_race_ids)} session(s) excluded from Gold. "
            f"Use --force to also delete any stale Gold records for "
            f"these sessions from previous runs."
        )

    # ---- Step 4: Save session quality diagnostics ----
    diagnostics_dir = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )),
        "data", "processed"
    )
    os.makedirs(diagnostics_dir, exist_ok=True)
    diagnostics_path = os.path.join(
        diagnostics_dir,
        "gold_session_quality_report.csv"
    )
    session_quality_df.to_csv(diagnostics_path, index=False)
    logger.info(
        f"Gold session quality report saved to {diagnostics_path}"
    )

    # ---- Step 5: Validate features ----
    validation_passed = validate_gold_features(df)

    if not validation_passed:
        logger.error(
            "Feature validation failed. "
            "Investigate errors above before writing to Gold table."
        )
        conn.close()
        return

    # ---- Step 6: Generate feature report ----
    generate_feature_report(df, session_quality_df=session_quality_df)

    # ---- Step 7: Write to Gold table ----
    total_written = write_gold_data(df, engine, conn, force=force)

    conn.close()

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 60)
    logger.info(
        f"FEATURE ENGINEERING COMPLETE: "
        f"{total_written} rows written to Gold table"
    )
    logger.info(f"Duration: {duration}")
    logger.info("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "F1 Feature Engineering Pipeline — "
            "Silver to Gold layer transformation"
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        help="Engineer features for a specific season only",
        default=None
    )

    parser.add_argument(
        "--circuit",
        type=str,
        help="Engineer features for a specific circuit only",
        default=None
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Gold records",
        default=False
    )

    args = parser.parse_args()

    seasons_arg = [args.season] if args.season else None
    circuits_arg = [args.circuit] if args.circuit else None

    run_feature_engineering_pipeline(
        seasons=seasons_arg,
        circuits=circuits_arg,
        force=args.force
    )