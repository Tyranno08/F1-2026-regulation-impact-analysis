# src/pipelines/clean_data.py

"""
Data Cleaning Pipeline — Bronze to Silver Layer

Reads raw lap data from the MySQL Bronze table (raw_lap_data),
applies a series of documented cleaning steps, and writes the
validated output to the Silver table (clean_lap_data).

Cleaning steps applied in order:
    1.  Parse lap times from string timedelta format to seconds
    2.  Remove laps with unparseable or null lap times
    3.  Remove lap zero and formation laps
    4.  Remove pit stop laps (pit-in and pit-out laps)
    5.  Remove safety car and VSC laps
    6.  Remove unknown or missing driver identifiers
    7.  Remove physically impossible lap times
    8.  Drop rain-dominated sessions (>50% wet/intermediate laps)
    9.  Remove individual wet and intermediate laps
    10. Remove statistical outliers (>2.5 sigma from session mean)
    11. Remove drying-track transition laps
    12. Compute fuel weight estimate
    13. Compute lap_time_delta_from_session_median (dry-only reference)
    14. Assign data_split label (train vs validation)
    15. Write to Silver table

Design principle:
    Every removal decision is logged with a count so the cleaning
    report shows exactly how many laps were removed at each step
    and why. This is called an audit trail and it is expected in
    any professional data pipeline.

Safety principle:
    After every major filtering step the pipeline checks whether
    all rows have been removed. If so it logs a CRITICAL error and
    stops immediately instead of crashing downstream. The statistical
    outlier filter additionally protects individual sessions: if
    removing outliers would empty a session, that session is kept
    unfiltered and a warning is logged.

Usage:
    python src/pipelines/clean_data.py

    Optional — clean only a specific season:
    python src/pipelines/clean_data.py --season 2024

    Optional — clean only a specific circuit:
    python src/pipelines/clean_data.py --circuit Monza

    Optional — force re-clean even if already in Silver table:
    python src/pipelines/clean_data.py --force
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

logger = get_logger("cleaning")


# ============================================================
# Constants
# ============================================================

# Compounds treated as dry-weather tyres for median reference
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

# Compounds treated as wet-weather tyres — removed from training data
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

# Sessions where more than this fraction of laps are wet are dropped
# entirely rather than partially cleaned
RAIN_SESSION_THRESHOLD = 0.50

# Sigma threshold for statistical outlier removal.
# Set to 2.5 (tighter than the standard 3.0) because our median
# reference is now computed from dry laps only, so the distribution
# is narrower and genuine outliers will be closer to the centre.
OUTLIER_SIGMA_THRESHOLD = 2.5


# ============================================================
# Time Parsing Utilities
# ============================================================

def parse_timedelta_to_seconds(time_str: str) -> float | None:
    """
    Converts a FastF1 timedelta string to total seconds as a float.

    FastF1 stores lap times as strings like:
        "0:01:22.456000"   (1 minute 22.456 seconds = 82.456s)
        "0:00:28.123000"   (sector time = 28.123s)

    Some edge cases we handle:
        "NaT"              -> None  (pandas not-a-time)
        "None"             -> None
        "nan"              -> None
        ""                 -> None
        Negative values    -> None  (physically impossible)

    Args:
        time_str: A time string in timedelta format from FastF1

    Returns:
        Total seconds as float, or None if unparseable.
    """
    if time_str is None:
        return None

    time_str = str(time_str).strip()

    if time_str in ("NaT", "None", "nan", "", "NaN"):
        return None

    try:
        parts = time_str.split(":")

        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total = hours * 3600 + minutes * 60 + seconds

        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            total = minutes * 60 + seconds

        else:
            total = float(time_str)

        if total <= 0:
            return None

        return round(total, 6)

    except (ValueError, TypeError):
        return None


def parse_all_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies parse_timedelta_to_seconds to all time columns in the
    dataframe and adds the converted _seconds columns.

    Args:
        df: Raw lap dataframe from Bronze table

    Returns:
        DataFrame with added numeric time columns.
    """
    logger.info("Parsing time columns to seconds...")

    df["lap_time_seconds"] = df["lap_time"].apply(
        parse_timedelta_to_seconds
    )
    df["sector1_seconds"] = df["sector1_time"].apply(
        parse_timedelta_to_seconds
    )
    df["sector2_seconds"] = df["sector2_time"].apply(
        parse_timedelta_to_seconds
    )
    df["sector3_seconds"] = df["sector3_time"].apply(
        parse_timedelta_to_seconds
    )

    parsed = df["lap_time_seconds"].notna().sum()
    failed = df["lap_time_seconds"].isna().sum()
    logger.info(
        f"Lap time parsing: {parsed} successful, {failed} failed"
    )

    return df


# ============================================================
# Individual Cleaning Steps
# ============================================================

def remove_null_lap_times(
    df: pd.DataFrame,
    audit: dict
) -> pd.DataFrame:
    """
    Removes laps where lap_time_seconds could not be parsed.

    These are laps where FastF1 recorded no valid time. This
    commonly happens on the final lap of a stint when a driver
    retires, during red flag periods, or when timing data
    was simply not captured.
    """
    before = len(df)
    df = df[df["lap_time_seconds"].notna()].copy()
    removed = before - len(df)
    audit["null_lap_times"] = removed
    logger.info(f"Removed {removed} laps with null lap times")
    return df


def remove_formation_and_lap_zero(
    df: pd.DataFrame,
    audit: dict
) -> pd.DataFrame:
    """
    Removes lap number 0 and lap number 1 in specific conditions.

    Lap 0 represents the formation lap driven before the race start.
    We keep lap 1 because it is the first racing lap, but it will
    often be slower due to standing start acceleration and cold tires.
    The statistical outlier filter downstream will handle truly
    anomalous lap 1 times.
    """
    before = len(df)
    df = df[df["lap_number"] > 0].copy()
    removed = before - len(df)
    audit["formation_laps"] = removed
    logger.info(f"Removed {removed} formation and lap-zero laps")
    return df


def remove_pit_laps(
    df: pd.DataFrame,
    audit: dict
) -> pd.DataFrame:
    """
    Removes laps flagged as pit laps.

    We remove both pit-in laps (the slow lap into the pits) and
    pit-out laps (the slow lap exiting the pits) because neither
    represents true race pace.
    """
    before = len(df)
    df = df[df["is_pit_lap"] == False].copy()
    removed = before - len(df)
    audit["pit_laps"] = removed
    logger.info(f"Removed {removed} pit stop laps")
    return df


def remove_safety_car_laps(
    df: pd.DataFrame,
    audit: dict
) -> pd.DataFrame:
    """
    Removes laps completed under safety car or VSC conditions.

    During safety car periods all cars follow the safety car at
    a pace that has nothing to do with car performance. Including
    these laps would corrupt the model's understanding of how car
    characteristics affect lap time.
    """
    before = len(df)
    df = df[df["is_safety_car"] == False].copy()
    removed = before - len(df)
    audit["safety_car_laps"] = removed
    logger.info(f"Removed {removed} safety car and VSC laps")
    return df


def remove_unknown_drivers(
    df: pd.DataFrame,
    audit: dict
) -> pd.DataFrame:
    """
    Removes laps where the driver could not be identified.

    Laps with driver "UNK" represent cases where neither FastF1
    nor our reference table could identify the driver. These laps
    cannot be used for driver-level analysis or target encoding.
    """
    before = len(df)

    d_number_count = df[df["driver"].str.startswith("D")].shape[0]
    if d_number_count > 0:
        logger.warning(
            f"{d_number_count} laps have D-number driver codes "
            f"(number-only fallback). These are kept but flagged."
        )

    df = df[df["driver"] != "UNK"].copy()
    removed = before - len(df)
    audit["unknown_drivers"] = removed
    logger.info(f"Removed {removed} laps with unknown drivers")
    return df


def remove_physically_impossible_times(
    df: pd.DataFrame,
    audit: dict
) -> pd.DataFrame:
    """
    Removes laps with physically impossible lap times.

    We define physically impossible as:
        - Lap time under 40 seconds (no F1 circuit has a lap this short)
        - Lap time over 300 seconds (5 minutes)
    """
    before = len(df)
    df = df[
        (df["lap_time_seconds"] >= 40) &
        (df["lap_time_seconds"] <= 300)
    ].copy()
    removed = before - len(df)
    audit["physically_impossible"] = removed
    logger.info(
        f"Removed {removed} laps with physically impossible times"
    )
    return df


def remove_wet_weather_laps(
    df: pd.DataFrame,
    audit: dict
) -> pd.DataFrame:
    """
    Removes wet and intermediate laps, but does NOT drop entire
    sessions anymore.

    Rationale:
        Earlier versions of the pipeline dropped entire sessions if
        more than 50% of laps were wet/intermediate. While this was
        conservative for dry-race modeling, it removed too much useful
        dry-compound data from sessions like Australia 2025 and
        Silverstone 2025.

        The revised approach keeps all dry-compound laps and removes
        only the wet/intermediate laps. This preserves circuit coverage
        for modeling while still excluding wet-weather physics from
        the training target.

    Args:
        df:    Input dataframe after safety car laps have been removed
        audit: Audit trail dictionary

    Returns:
        Filtered dataframe with wet/intermediate laps removed.
    """
    before = len(df)

    # Diagnostic only: log wet share by session
    session_wet_pct = (
        df.groupby(["season", "circuit"])
        .apply(lambda g: g["compound"].isin(WET_COMPOUNDS).mean())
        .reset_index()
        .rename(columns={0: "wet_pct"})
    )

    high_wet_sessions = session_wet_pct[
        session_wet_pct["wet_pct"] > RAIN_SESSION_THRESHOLD
    ]

    if len(high_wet_sessions) > 0:
        logger.warning(
            "Rain-dominated sessions detected, but entire-session "
            "dropping is disabled. Keeping dry laps and removing only "
            "wet/intermediate laps."
        )
        for _, row in high_wet_sessions.iterrows():
            logger.warning(
                f"  {int(row['season'])} {row['circuit']} — "
                f"{row['wet_pct']*100:.0f}% wet/intermediate laps"
            )

    # Remove individual wet/intermediate laps only
    wet_lap_mask = df["compound"].isin(WET_COMPOUNDS)
    individual_wet_laps = wet_lap_mask.sum()

    if individual_wet_laps > 0:
        wet_by_session = (
            df[wet_lap_mask]
            .groupby(["season", "circuit", "compound"])
            .size()
            .reset_index(name="count")
        )
        for _, row in wet_by_session.iterrows():
            logger.info(
                f"  Removing {int(row['count'])} {row['compound']} laps "
                f"from {int(row['season'])} {row['circuit']}"
            )

        df = df[~wet_lap_mask].copy()
        logger.info(
            f"Removed {individual_wet_laps} individual "
            f"wet/intermediate laps from mixed-condition sessions"
        )

    total_wet_removed = before - len(df)
    audit["rain_dominated_sessions"] = 0
    audit["wet_laps_removed"] = individual_wet_laps
    audit["total_wet_removed"] = total_wet_removed

    logger.info(
        f"Wet weather removal complete. "
        f"Total removed: {total_wet_removed} laps "
        f"({total_wet_removed / before * 100:.1f}% of pre-wet input)"
    )

    return df


def remove_statistical_outliers(
    df: pd.DataFrame,
    audit: dict,
    sigma_threshold: float = OUTLIER_SIGMA_THRESHOLD
) -> pd.DataFrame:
    """
    Removes laps that are statistical outliers within their session.

    We compute the mean and standard deviation of lap times for each
    race session independently. Laps beyond sigma_threshold standard
    deviations from the session mean are removed.

    Per-session safeguard:
        If removing outliers would empty an entire session, we keep
        all laps for that session and log a warning. This prevents
        silent data loss for sessions with unusual distributions
        (e.g., very few laps remaining after wet removal).

    Critically:
    - This runs AFTER wet lap removal, so the distribution being
      analysed is dry-compound laps only. This is why we can use
      a tighter threshold of 2.5 sigma instead of the standard 3.0.
    - This runs PER SESSION because Monza laps (~80s) and Monaco
      laps (~74s) have completely different distributions and a
      global threshold would be meaningless.

    Args:
        df:              Input dataframe
        audit:           Audit trail dictionary
        sigma_threshold: Standard deviation cutoff. Default 2.5.

    Returns:
        Filtered dataframe.
    """
    before = len(df)
    kept_sessions = []
    total_removed = 0
    fallback_sessions = 0

    for race_id, session_df in df.groupby("race_id"):
        session_df = session_df.copy()
        session_mean = session_df["lap_time_seconds"].mean()
        session_std = session_df["lap_time_seconds"].std()

        # If std is NaN or zero (single-lap session), skip filtering
        if pd.isna(session_std) or session_std == 0:
            logger.warning(
                f"  {race_id}: std is zero or NaN — "
                f"skipping outlier filter for this session "
                f"({len(session_df)} laps kept as-is)"
            )
            kept_sessions.append(session_df)
            continue

        inlier_mask = (
            np.abs(session_df["lap_time_seconds"] - session_mean)
            <= (sigma_threshold * session_std)
        )

        filtered = session_df[inlier_mask]

        if filtered.empty:
            # SAFEGUARD: keep entire session if filter would empty it
            logger.warning(
                f"  {race_id}: outlier filter would remove ALL "
                f"{len(session_df)} laps — keeping entire session. "
                f"Check cleaning thresholds for this race."
            )
            kept_sessions.append(session_df)
            fallback_sessions += 1
        else:
            removed_count = len(session_df) - len(filtered)
            if removed_count > 0:
                logger.debug(
                    f"  {race_id}: removed {removed_count} outlier laps "
                    f"(kept {len(filtered)}/{len(session_df)})"
                )
            total_removed += removed_count
            kept_sessions.append(filtered)

    df = pd.concat(kept_sessions, ignore_index=True)

    removed = before - len(df)
    audit["statistical_outliers"] = removed

    logger.info(
        f"Removed {removed} laps as statistical outliers "
        f"(>{sigma_threshold} sigma from session mean)"
    )
    if fallback_sessions > 0:
        logger.warning(
            f"  {fallback_sessions} session(s) triggered the "
            f"empty-session safeguard and were kept unfiltered"
        )

    return df


def remove_transition_laps(
    df: pd.DataFrame,
    audit: dict,
    max_delta_seconds: float = 8.0
) -> pd.DataFrame:
    """
    Removes drying-track transition laps that slipped through all
    previous filters.

    These are laps where a driver is on dry-compound tyres but the
    track surface is still damp from earlier rain. They appear as
    clean laps in all earlier filters because:
        - Compound is SOFT/MEDIUM/HARD  (not flagged as wet)
        - Safety car has already come in (not flagged as SC lap)
        - They are statistically rare enough to survive sigma filtering
          in some sessions

    Real-world example caught by this filter:
        2023 Interlagos laps 6-9 — SOFT tyres on a drying track,
        running 87-92 seconds against a dry median of ~76 seconds.
        Delta of +16 seconds is physically impossible for a dry lap.

    We use a hard cap of max_delta_seconds above the per-session
    dry median. Any dry-compound lap slower than this cap is removed.

    The cap is set at 8.0 seconds, which is:
        - Above the legitimate maximum seen at any circuit (~7.7s
          at Silverstone after fixing wet lap contamination)
        - Well below the transition lap deltas being caught (+10 to
          +16 seconds)

    Args:
        df:                Input dataframe — must contain only dry
                           compound laps at this point in the pipeline.
        audit:             Audit trail dictionary.
        max_delta_seconds: Hard cap on how slow a lap can be relative
                           to the session dry median. Default 8.0s.

    Returns:
        Filtered dataframe with transition laps removed.
    """
    before = len(df)

    # Compute a temporary per-session dry median for the cap check
    # This is the same logic as compute_lap_delta but used only for
    # filtering — the proper delta column is computed separately later
    session_dry_medians = (
        df[df["compound"].isin(DRY_COMPOUNDS)]
        .groupby(["season", "circuit"])["lap_time_seconds"]
        .median()
        .reset_index()
        .rename(columns={"lap_time_seconds": "dry_median"})
    )

    df = df.merge(session_dry_medians, on=["season", "circuit"], how="left")

    # Flag laps that are more than max_delta_seconds above the dry median
    transition_mask = (
        df["compound"].isin(DRY_COMPOUNDS) &
        ((df["lap_time_seconds"] - df["dry_median"]) > max_delta_seconds)
    )

    transition_count = transition_mask.sum()

    if transition_count > 0:
        # Log exactly which sessions and laps are being removed
        flagged = df[transition_mask][
            ["season", "circuit", "driver", "lap_number",
             "lap_time_seconds", "compound", "tire_life"]
        ].copy()
        flagged["delta"] = flagged["lap_time_seconds"] - df.loc[
            transition_mask, "dry_median"
        ].values

        by_session = flagged.groupby(["season", "circuit"]).size()
        for (season, circuit), count in by_session.items():
            logger.warning(
                f"  Removing {count} transition laps from "
                f"{int(season)} {circuit} — dry-compound laps "
                f">{max_delta_seconds}s above session median "
                f"(drying track detected)"
            )

    df = df[~transition_mask].copy()
    df = df.drop(columns=["dry_median"])

    removed = before - len(df)
    audit["transition_laps"] = removed
    logger.info(
        f"Removed {removed} drying-track transition laps "
        f"(dry compound laps >{max_delta_seconds}s above session median)"
    )

    return df


# ============================================================
# Derived Field Computation
# ============================================================

def compute_fuel_weight_estimate(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Estimates the car's fuel weight at each lap using a linear model.

    F1 cars start a race with approximately 110kg of fuel.
    The average burn rate is approximately 1.8kg per lap.
    Therefore at lap N the estimated remaining fuel is:

        fuel_weight = starting_fuel - (lap_number * burn_rate)

    We floor the result at 0 because fuel cannot be negative.
    """
    fuel_cfg = config.get("fuel_model", {})
    starting_fuel = fuel_cfg.get("starting_fuel_kg", 110)
    burn_rate = fuel_cfg.get("burn_rate_kg_per_lap", 1.8)

    df["fuel_weight_estimate"] = (
        starting_fuel - (df["lap_number"] * burn_rate)
    ).clip(lower=0)

    logger.info(
        f"Fuel weight computed. Range: "
        f"{df['fuel_weight_estimate'].min():.1f}kg to "
        f"{df['fuel_weight_estimate'].max():.1f}kg"
    )
    return df


def compute_lap_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes each lap's time delta from the DRY-COMPOUND session median.

    This is our ML target variable.

    Why dry-only median:
        In mixed-condition races (Monaco 2023, Silverstone 2024/2025,
        Interlagos 2023) the race switches between dry and wet
        conditions. If we compute the median across all laps, wet
        laps on intermediate tyres (~100-110s at Monaco) inflate the
        median and make dry laps appear as extreme negatives. This
        was observed producing deltas of -13.6s and +31.6s which are
        physically impossible for dry racing laps.

        By the time this function runs, all wet/intermediate laps
        have already been removed. The dry-only median is therefore
        computed from the actual data remaining in the dataframe,
        which is already exclusively dry laps. The comment below
        is kept for documentation clarity — the DRY_COMPOUNDS filter
        is an extra safety net in case any edge case slips through.

    Why use delta instead of raw lap time:
        Raw lap times vary enormously between circuits.
        Monza: ~80 seconds. Monaco: ~74 seconds.
        By using delta from session median we normalise all circuits
        to a common scale. +0.5 means 0.5s slower than median pace.
        -1.0 means 1.0s faster. These are comparable across circuits
        and carry genuine performance signal.

    We use MEDIAN not mean because the median is robust to any
    remaining slow laps that survived earlier filters.

    Args:
        df: Input dataframe — must contain only dry-compound laps
            by this point in the pipeline.

    Returns:
        DataFrame with lap_time_delta_from_session_median column added.
    """
    logger.info(
        "Computing lap time delta from dry-compound session median..."
    )

    results = []

    for (season, circuit), session_df in df.groupby(
        ["season", "circuit"]
    ):
        session_df = session_df.copy()

        # Safety net: use only dry compound laps to set the reference
        # By this stage all wet laps should already be removed, but
        # this guard prevents any edge case from corrupting the median.
        dry_laps = session_df[
            session_df["compound"].isin(DRY_COMPOUNDS)
        ]

        if len(dry_laps) < 10:
            logger.warning(
                f"  {season} {circuit}: only {len(dry_laps)} dry laps "
                f"available for median reference — setting delta to NaN. "
                f"This session will be excluded from the Silver table."
            )
            session_df["lap_time_delta_from_session_median"] = float("nan")
        else:
            dry_median = dry_laps["lap_time_seconds"].median()
            session_df["lap_time_delta_from_session_median"] = (
                session_df["lap_time_seconds"] - dry_median
            )
            logger.debug(
                f"  {season} {circuit}: dry median = {dry_median:.3f}s "
                f"from {len(dry_laps)} laps"
            )

        results.append(session_df)

    df = pd.concat(results, ignore_index=True)

    # Drop any laps where delta could not be computed
    nan_delta_count = df["lap_time_delta_from_session_median"].isna().sum()
    if nan_delta_count > 0:
        logger.warning(
            f"Dropping {nan_delta_count} laps with NaN delta "
            f"(sessions with <10 dry laps after wet removal)"
        )
        df = df[
            df["lap_time_delta_from_session_median"].notna()
        ].copy()

    logger.info(
        f"Lap delta computed. Range: "
        f"{df['lap_time_delta_from_session_median'].min():.3f}s to "
        f"{df['lap_time_delta_from_session_median'].max():.3f}s"
    )
    return df


def assign_data_split(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Assigns temporal data splits for modeling.

    Split design:
        Train:      2023, 2024
        Test:       2025
        Validation: 2026

    This prevents leakage and preserves a true temporal holdout.

    Args:
        df:     Clean dataframe
        config: Project config dict

    Returns:
        DataFrame with data_split column assigned.
    """
    training_seasons = config.get("training_seasons", [2023, 2024])
    test_seasons = config.get("test_seasons", [2025])
    validation_seasons = config.get("validation_seasons", [2026])

    df["data_split"] = "train"
    df.loc[df["season"].isin(test_seasons), "data_split"] = "test"
    df.loc[df["season"].isin(validation_seasons), "data_split"] = "validation"

    split_counts = df["data_split"].value_counts().to_dict()
    logger.info(f"Data split assigned: {split_counts}")

    return df


# ============================================================
# Data Quality Report
# ============================================================

def generate_data_quality_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    audit: dict
) -> None:
    """
    Prints a comprehensive data quality report to the terminal
    and saves it to data/processed/.

    Documents every cleaning decision and shows the final dataset
    composition including the new wet weather removal steps.
    """
    total_raw = len(df_raw)
    total_clean = len(df_clean)
    total_removed = total_raw - total_clean
    retention_rate = (total_clean / total_raw * 100) if total_raw > 0 else 0

    report_lines = [
        "",
        "=" * 70,
        "DATA CLEANING PIPELINE — QUALITY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "REMOVAL AUDIT TRAIL",
        "-" * 70,
    ]

    removal_steps = [
        ("null_lap_times",           "Null or unparseable lap times"),
        ("formation_laps",           "Formation laps and lap zero"),
        ("pit_laps",                 "Pit stop laps (in-lap + out-lap)"),
        ("safety_car_laps",          "Safety car and VSC laps"),
        ("unknown_drivers",          "Unknown driver identifier"),
        ("physically_impossible",    "Physically impossible times"),
        ("rain_dominated_sessions",  "Rain-dominated sessions (>50% wet) — entire session dropped"),
        ("wet_laps_removed",         "Individual wet/intermediate laps from partial-rain sessions"),
        ("statistical_outliers",     f"Statistical outliers (>{OUTLIER_SIGMA_THRESHOLD} sigma)"),
        ("transition_laps",          "Drying-track transition laps (dry compound, >8s above median)"),
    ]

    for key, description in removal_steps:
        count = audit.get(key, 0)
        pct = (count / total_raw * 100) if total_raw > 0 else 0
        report_lines.append(
            f"  {description:<58} {count:>5} laps  ({pct:.1f}%)"
        )

    report_lines += [
        "",
        "-" * 70,
        f"  {'Total raw laps':<58} {total_raw:>5}",
        f"  {'Total removed':<58} {total_removed:>5} "
        f"({100 - retention_rate:.1f}%)",
        f"  {'Total clean laps retained':<58} {total_clean:>5} "
        f"({retention_rate:.1f}%)",
        "",
        "CLEAN DATASET COMPOSITION",
        "-" * 70,
    ]

    # Season breakdown
    report_lines.append("  By Season:")
    for season in sorted(df_clean["season"].unique()):
        season_count = (df_clean["season"] == season).sum()
        split = df_clean[
            df_clean["season"] == season
        ]["data_split"].iloc[0]
        report_lines.append(
            f"    {season}  {season_count:>6} laps  [{split}]"
        )

    report_lines.append("")
    report_lines.append("  By Circuit:")
    circuit_counts = (
        df_clean.groupby("circuit")
        .size()
        .sort_values(ascending=False)
    )
    for circuit, count in circuit_counts.items():
        report_lines.append(f"    {circuit:<25} {count:>6} laps")

    report_lines += [
        "",
        "  By Compound:",
    ]
    compound_counts = (
        df_clean.groupby("compound")
        .size()
        .sort_values(ascending=False)
    )
    for compound, count in compound_counts.items():
        pct = count / total_clean * 100
        report_lines.append(
            f"    {compound:<15} {count:>6} laps  ({pct:.1f}%)"
        )

    report_lines += [
        "",
        "  Lap Time Delta (target variable):",
        f"    Mean:   {df_clean['lap_time_delta_from_session_median'].mean():.4f}s",
        f"    Median: {df_clean['lap_time_delta_from_session_median'].median():.4f}s",
        f"    Std:    {df_clean['lap_time_delta_from_session_median'].std():.4f}s",
        f"    Min:    {df_clean['lap_time_delta_from_session_median'].min():.4f}s",
        f"    Max:    {df_clean['lap_time_delta_from_session_median'].max():.4f}s",
        "",
        "  Weather Coverage:",
        f"    Track temp available: "
        f"{df_clean['track_temp'].notna().sum():>6} laps "
        f"({df_clean['track_temp'].notna().mean()*100:.1f}%)",
        f"    Air temp available:   "
        f"{df_clean['air_temp'].notna().sum():>6} laps "
        f"({df_clean['air_temp'].notna().mean()*100:.1f}%)",
        "",
        "=" * 70,
    ]

    full_report = "\n".join(report_lines)
    print(full_report)
    logger.info("Data quality report generated")

    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )),
        "data", "processed"
    )
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(
        reports_dir,
        f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_path, "w") as f:
        f.write(full_report)
    logger.info(f"Report saved to {report_path}")


# ============================================================
# Database Read and Write
# ============================================================

def load_raw_data(
    engine,
    seasons: list[int] = None,
    circuits: list[str] = None
) -> pd.DataFrame:
    """
    Reads raw lap data from the Bronze table into a dataframe.

    Args:
        engine:   SQLAlchemy engine
        seasons:  Optional list of seasons to filter by
        circuits: Optional list of circuits to filter by

    Returns:
        Raw lap dataframe from Bronze table.
    """
    logger.info("Loading raw data from Bronze table...")

    base_query = "SELECT * FROM raw_lap_data WHERE 1=1"
    params = {}

    if seasons:
        placeholders = ", ".join(
            [f":season_{i}" for i in range(len(seasons))]
        )
        base_query += f" AND season IN ({placeholders})"
        for i, s in enumerate(seasons):
            params[f"season_{i}"] = s

    if circuits:
        placeholders = ", ".join(
            [f":circuit_{i}" for i in range(len(circuits))]
        )
        base_query += f" AND circuit IN ({placeholders})"
        for i, c in enumerate(circuits):
            params[f"circuit_{i}"] = c

    from sqlalchemy import text
    df = pd.read_sql(text(base_query), engine, params=params)

    logger.info(
        f"Loaded {len(df)} raw rows from Bronze table "
        f"across {df['season'].nunique()} season(s) "
        f"and {df['circuit'].nunique()} circuit(s)"
    )
    return df


def check_already_cleaned(
    race_id: str,
    conn
) -> bool:
    """
    Checks if a race session has already been written to the
    Silver clean_lap_data table.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM clean_lap_data WHERE race_id = %s",
        (race_id,)
    )
    count = cursor.fetchone()[0]
    cursor.close()
    return count > 0


def write_clean_data_to_silver(
    df: pd.DataFrame,
    engine,
    conn,
    force: bool = False
) -> int:
    """
    Writes the cleaned dataframe to the Silver clean_lap_data table.

    Processes one race session at a time to enable the idempotency
    check. If a session is already in the Silver table and force
    is False, it is skipped.

    Args:
        df:     Cleaned dataframe with all required columns
        engine: SQLAlchemy engine for pandas to_sql
        conn:   MySQL connection for the idempotency check
        force:  If True, delete existing records and re-insert

    Returns:
        Total number of rows written.
    """
    silver_columns = [
        "race_id", "season", "circuit", "driver", "team",
        "lap_number", "lap_time_seconds", "sector1_seconds",
        "sector2_seconds", "sector3_seconds", "compound",
        "tire_life", "track_temp", "air_temp", "humidity",
        "speed_trap", "is_valid_lap", "data_split",
        "fuel_weight_estimate", "lap_time_delta_from_session_median"
    ]

    df["is_valid_lap"] = True

    existing_cols = [c for c in silver_columns if c in df.columns]
    df_silver = df[existing_cols].copy()

    total_written = 0
    race_ids = df_silver["race_id"].unique()

    logger.info(
        f"Writing {len(df_silver)} rows across "
        f"{len(race_ids)} sessions to Silver table..."
    )

    for race_id in race_ids:
        race_df = df_silver[df_silver["race_id"] == race_id]

        if not force and check_already_cleaned(race_id, conn):
            logger.info(
                f"SKIPPED — {race_id} already in Silver table. "
                f"Use --force to re-clean."
            )
            continue

        if force:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM clean_lap_data WHERE race_id = %s",
                (race_id,)
            )
            conn.commit()
            cursor.close()
            logger.info(
                f"Deleted existing Silver records for {race_id}"
            )

        try:
            race_df.to_sql(
                name="clean_lap_data",
                con=engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500
            )
            written = len(race_df)
            total_written += written
            logger.info(
                f"Wrote {written} clean rows for {race_id}"
            )

        except Exception as e:
            logger.error(
                f"Failed to write Silver data for {race_id}: {e}"
            )

    return total_written


# ============================================================
# Pipeline Safety Guard
# ============================================================

def _check_empty(df: pd.DataFrame, step_name: str) -> bool:
    """
    Returns True (and logs a critical error) if the dataframe is
    empty after a cleaning step.  The caller should return early
    to avoid downstream crashes on operations like .median() or
    .groupby() on an empty frame.

    Args:
        df:        The dataframe to check.
        step_name: Human-readable name of the cleaning step that
                   just ran, used in the log message.

    Returns:
        True if the dataframe is empty, False otherwise.
    """
    if df.empty:
        logger.critical(
            f"ALL rows removed after '{step_name}'. "
            f"Pipeline cannot continue. "
            f"Check cleaning thresholds or input data."
        )
        return True
    return False


# ============================================================
# Main Cleaning Pipeline Orchestrator
# ============================================================

def run_cleaning_pipeline(
    seasons: list[int] = None,
    circuits: list[str] = None,
    force: bool = False
) -> None:
    """
    Orchestrates the full Bronze to Silver cleaning pipeline.

    Cleaning order:
        1.  Load raw Bronze data
        2.  Parse time strings to float seconds
        3.  Remove null lap times
        4.  Remove formation / lap-zero laps
        5.  Remove pit stop laps
        6.  Remove safety car and VSC laps
        7.  Remove unknown drivers
        8.  Remove physically impossible times
        9.  Remove wet/intermediate laps                     ← REVISED
        10. Remove statistical outliers (2.5 sigma, per-session safe)
        10B. Remove drying-track transition laps
        11. Compute fuel weight estimate
        12. Compute lap delta from dry-only session median
        13. Assign train/validation split
        14. Generate quality report
        15. Write to Silver table

    After every filtering step, the pipeline checks whether any
    rows remain. If all rows have been removed, it logs a CRITICAL
    error, closes the database connection, and exits cleanly.

    Args:
        seasons:  Optional list of seasons to clean.
        circuits: Optional list of circuits to clean.
        force:    If True, re-cleans data already in Silver table.
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE STARTING")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if force:
        logger.warning(
            "FORCE MODE ENABLED — existing Silver data will be overwritten"
        )
    logger.info("=" * 60)

    config = load_config()
    engine = get_sqlalchemy_engine()
    conn = get_mysql_connection()

    # ---- Step 1: Load raw data from Bronze ----
    df = load_raw_data(engine, seasons=seasons, circuits=circuits)

    if df.empty:
        logger.error(
            "No data loaded from Bronze table. "
            "Have you run the ingestion pipeline?"
        )
        conn.close()
        return

    df_raw = df.copy()
    audit = {}

    logger.info(f"Starting cleaning with {len(df)} raw laps")

    # ---- Step 2: Parse time strings to seconds ----
    df = parse_all_time_columns(df)

    # ---- Step 3: Remove null lap times ----
    df = remove_null_lap_times(df, audit)
    if _check_empty(df, "null lap time removal"):
        conn.close()
        return

    # ---- Step 4: Remove formation / lap-zero laps ----
    df = remove_formation_and_lap_zero(df, audit)
    if _check_empty(df, "formation lap removal"):
        conn.close()
        return

    # ---- Step 5: Remove pit stop laps ----
    df = remove_pit_laps(df, audit)
    if _check_empty(df, "pit lap removal"):
        conn.close()
        return

    # ---- Step 6: Remove safety car and VSC laps ----
    df = remove_safety_car_laps(df, audit)
    if _check_empty(df, "safety car removal"):
        conn.close()
        return

    # ---- Step 7: Remove unknown drivers ----
    df = remove_unknown_drivers(df, audit)
    if _check_empty(df, "unknown driver removal"):
        conn.close()
        return

    # ---- Step 8: Remove physically impossible times ----
    df = remove_physically_impossible_times(df, audit)
    if _check_empty(df, "physically impossible time removal"):
        conn.close()
        return

    # ---- Step 9: Remove wet weather laps ----
    # This must run BEFORE the outlier filter so that wet laps
    # do not distort the per-session mean and std used for outlier
    # detection. It must also run BEFORE compute_lap_delta so that
    # the session median reference is computed from dry laps only.
    df = remove_wet_weather_laps(df, audit)
    if _check_empty(df, "wet weather removal"):
        conn.close()
        return

    # ---- Step 10: Statistical outlier removal ----
    # Runs on dry-only data with tighter 2.5 sigma threshold.
    # Per-session safeguard: if removing outliers would empty a
    # session, that session is kept unfiltered.
    df = remove_statistical_outliers(
        df, audit, sigma_threshold=OUTLIER_SIGMA_THRESHOLD
    )
    if _check_empty(df, "statistical outlier removal"):
        conn.close()
        return

    # ---- Step 10B: Remove drying-track transition laps ----
    # Catches dry-compound laps on a damp track that survived all
    # previous filters. Must run BEFORE compute_lap_delta.
    df = remove_transition_laps(df, audit, max_delta_seconds=8.0)
    if _check_empty(df, "transition lap removal"):
        conn.close()
        return

    # ---- Step 11–13: Compute derived fields ----
    df = compute_fuel_weight_estimate(df, config)
    df = compute_lap_delta(df)
    df = assign_data_split(df, config)

    logger.info(
        f"Cleaning complete. {len(df)} laps remain "
        f"from {len(df_raw)} raw laps "
        f"({len(df)/len(df_raw)*100:.1f}% retained)"
    )

    # ---- Step 14: Generate quality report ----
    generate_data_quality_report(df_raw, df, audit)

    # ---- Step 15: Write to Silver table ----
    total_written = write_clean_data_to_silver(
        df, engine, conn, force=force
    )

    conn.close()

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 60)
    logger.info(
        f"CLEANING PIPELINE COMPLETE: "
        f"{total_written} rows written to Silver table"
    )
    logger.info(f"Duration: {duration}")
    logger.info("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "F1 Data Cleaning Pipeline — "
            "Bronze to Silver layer transformation"
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        help="Clean only a specific season (e.g., --season 2024)",
        default=None
    )

    parser.add_argument(
        "--circuit",
        type=str,
        help="Clean only a specific circuit (e.g., --circuit Monza)",
        default=None
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-clean sessions already present in Silver table",
        default=False
    )

    args = parser.parse_args()

    seasons_arg = [args.season] if args.season else None
    circuits_arg = [args.circuit] if args.circuit else None

    run_cleaning_pipeline(
        seasons=seasons_arg,
        circuits=circuits_arg,
        force=args.force
    )