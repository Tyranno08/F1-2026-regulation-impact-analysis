# src/ingestion/fetch_fastf1_data.py

"""
F1 Data Ingestion Pipeline — Bronze Layer

This script fetches lap-level data for the 2022, 2023, 2024 and 2025 F1 seasons
using the FastF1 library and stores the raw data into the MySQL Bronze table
(raw_lap_data) without any cleaning or transformation.

The philosophy here is that the Bronze layer stores data EXACTLY as received.
Cleaning is a separate concern handled in Phase 3.

Usage:
    python src/ingestion/fetch_fastf1_data.py

    Optional — fetch only a specific season:
    python src/ingestion/fetch_fastf1_data.py --season 2024

    Optional — fetch only a specific circuit:
    python src/ingestion/fetch_fastf1_data.py --season 2024 --circuit Monza
"""

import sys
import os
import argparse
import time
from datetime import datetime

import fastf1
import pandas as pd
import numpy as np
import mysql.connector

# Add project root to path so we can import from src/
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from db_connection import get_mysql_connection
from config_loader import load_config
from logger import get_logger

# Initialize logger for this module
logger = get_logger("ingestion")


# ============================================================
# FastF1 Setup
# ============================================================

def setup_fastf1_cache(cache_path: str) -> None:
    """
    Enables FastF1 local caching.
    This is critical — without it we re-download data on every run.
    """
    os.makedirs(cache_path, exist_ok=True)
    fastf1.Cache.enable_cache(cache_path)
    logger.info(f"FastF1 cache enabled at: {cache_path}")


# ============================================================
# Helper — Timedelta to Seconds
# ============================================================

def td_to_seconds(val) -> float | None:
    """
    Converts a pandas Timedelta (or NaT) to a plain float in seconds.

    FastF1 returns lap times and sector times as pandas Timedelta objects
    e.g. Timedelta('0 days 00:01:22.456000000').
    MySQL FLOAT columns cannot store these — we convert to seconds (82.456).

    Args:
        val: A pandas Timedelta, NaT, or None.

    Returns:
        Float seconds, or None if the value is missing/invalid.
    """
    if val is None:
        return None
    try:
        if pd.isnull(val):
            return None
    except Exception:
        pass
    try:
        return val.total_seconds()
    except AttributeError:
        return None


# ============================================================
# Session Loading
# ============================================================

def load_race_session(season: int, circuit: str) -> fastf1.core.Session | None:
    """
    Loads a single race session from FastF1.

    Args:
        season:  The year (e.g., 2023)
        circuit: The circuit name as FastF1 knows it (e.g., 'Monza')

    Returns:
        A loaded FastF1 Session object, or None if loading fails.
    """
    try:
        logger.info(f"Loading session: {season} {circuit} Race")
        session = fastf1.get_session(season, circuit, "R")
        session.load(
            laps=True,
            telemetry=True,
            weather=True,
            messages=False
        )
        logger.info(
            f"Session loaded successfully: {season} {circuit} "
            f"({len(session.laps)} laps)"
        )
        return session

    except Exception as e:
        logger.error(
            f"Failed to load session {season} {circuit}: {type(e).__name__}: {e}"
        )
        return None


# ============================================================
# Data Extraction
# ============================================================

def extract_laps_from_session(
    session: fastf1.core.Session,
    season: int,
    circuit: str
) -> list[dict]:
    """
    Extracts lap-level data from a loaded FastF1 session.

    We extract only the fields our Bronze schema expects.
    We do NOT clean, filter, or transform here — that is Phase 3's job.
    We store everything including pit laps, safety car laps, and outlaps
    so the Silver cleaning pipeline has complete information to work with.

    Args:
        session: A loaded FastF1 Session object
        season:  The race season year
        circuit: The circuit name

    Returns:
        A list of dictionaries, each representing one lap row.
    """
    rows = []
    laps = session.laps

    if laps is None or len(laps) == 0:
        logger.warning(f"No lap data found for {season} {circuit}")
        return rows

    # Build a race_id like "2023_Monza" for easy filtering later
    race_id = f"{season}_{circuit.replace(' ', '_')}"

    # Try to get weather data — not always available
    try:
        weather_data = session.weather_data
        has_weather = weather_data is not None and len(weather_data) > 0
    except Exception:
        has_weather = False
        logger.warning(f"Weather data unavailable for {season} {circuit}")

    for _, lap in laps.iterrows():

        # ---- Lap Time & Sector Times ----
        # FastF1 returns these as pandas Timedelta objects.
        # We convert to plain float seconds (e.g. 82.456) so they
        # fit into MySQL FLOAT columns cleanly.
        lap_time   = td_to_seconds(lap.get("LapTime", None))
        sector1    = td_to_seconds(lap.get("Sector1Time", None))
        sector2    = td_to_seconds(lap.get("Sector2Time", None))
        sector3    = td_to_seconds(lap.get("Sector3Time", None))

        # ---- Safety Car Detection ----
        # FastF1 tracks track status during laps.
        # Status codes: 1=Clear, 2=Yellow, 4=SC, 6=VSC, 7=Red
        track_status = str(lap.get("TrackStatus", "1"))
        is_safety_car = any(code in track_status for code in ["4", "6", "7"])

        # ---- Pit Lap Detection ----
        pit_out_time = lap.get("PitOutTime", None)
        pit_in_time = lap.get("PitInTime", None)
        is_pit_lap = bool(
            (pit_out_time is not None and not pd.isna(pit_out_time)) or
            (pit_in_time is not None and not pd.isna(pit_in_time))
        )

        # ---- Speed Trap ----
        speed_trap = lap.get("SpeedST", None)
        if speed_trap is not None and not pd.isna(speed_trap):
            speed_trap = float(speed_trap)
        else:
            speed_trap = None

        # ---- Weather ----
        # We match weather to lap by finding the closest weather
        # timestamp to the lap's session time
        track_temp = None
        air_temp = None
        humidity = None

        if has_weather:
            try:
                lap_time_val = lap.get("Time", None)
                if lap_time_val is not None and not pd.isna(lap_time_val):
                    # Find the weather row closest in time to this lap
                    time_diff = (
                        weather_data["Time"] - lap_time_val
                    ).abs()
                    closest_idx = time_diff.idxmin()
                    weather_row = weather_data.loc[closest_idx]

                    track_temp = float(weather_row.get("TrackTemp", 0))
                    air_temp = float(weather_row.get("AirTemp", 0))
                    humidity = float(weather_row.get("Humidity", 0))
            except Exception:
                pass

        # ---- Driver and Team ----
        driver = str(lap.get("Driver", "UNK"))
        team = str(lap.get("Team", "Unknown"))

        # ---- Compound and Tire Life ----
        compound = str(lap.get("Compound", "UNKNOWN"))
        tire_life = lap.get("TyreLife", None)
        if tire_life is not None and not pd.isna(tire_life):
            tire_life = int(tire_life)
        else:
            tire_life = None

        # ---- Lap Number ----
        lap_number = lap.get("LapNumber", None)
        if lap_number is not None and not pd.isna(lap_number):
            lap_number = int(lap_number)
        else:
            lap_number = None

        # Build the row dictionary matching Bronze schema exactly
        row = {
            "race_id":      race_id,
            "season":       season,
            "circuit":      circuit,
            "driver":       driver,
            "team":         team,
            "lap_number":   lap_number,
            "lap_time":     lap_time,       # float seconds e.g. 82.456
            "sector1_time": sector1,        # float seconds e.g. 27.123
            "sector2_time": sector2,        # float seconds e.g. 31.456
            "sector3_time": sector3,        # float seconds e.g. 23.877
            "compound":     compound,
            "tire_life":    tire_life,
            "track_temp":   track_temp,
            "air_temp":     air_temp,
            "humidity":     humidity,
            "speed_trap":   speed_trap,
            "is_pit_lap":   is_pit_lap,
            "is_safety_car": is_safety_car
        }

        rows.append(row)

    logger.info(
        f"Extracted {len(rows)} lap rows from {season} {circuit}"
    )
    return rows


# ============================================================
# Database Insertion
# ============================================================

def check_session_already_ingested(
    cursor: mysql.connector.cursor.MySQLCursor,
    race_id: str
) -> bool:
    """
    Checks if a race session has already been loaded into the database.
    This prevents duplicate rows if the script is run multiple times.

    Args:
        cursor:  An active MySQL cursor
        race_id: The race identifier string (e.g., "2023_Monza")

    Returns:
        True if the race already exists in the table, False otherwise.
    """
    cursor.execute(
        "SELECT COUNT(*) FROM raw_lap_data WHERE race_id = %s",
        (race_id,)
    )
    count = cursor.fetchone()[0]
    return count > 0


def insert_laps_to_bronze(
    rows: list[dict],
    conn: mysql.connector.connection.MySQLConnection
) -> int:
    """
    Inserts a list of lap dictionaries into the raw_lap_data Bronze table.

    Uses batch insertion (executemany) for efficiency.
    Commits the entire session as one transaction so either all laps
    from a session are inserted or none are — no partial sessions.

    Args:
        rows: List of lap dictionaries from extract_laps_from_session()
        conn: An active MySQL connection

    Returns:
        The number of rows successfully inserted.
    """
    if not rows:
        logger.warning("No rows to insert.")
        return 0

    insert_query = """
        INSERT INTO raw_lap_data (
            race_id,
            season,
            circuit,
            driver,
            team,
            lap_number,
            lap_time,
            sector1_time,
            sector2_time,
            sector3_time,
            compound,
            tire_life,
            track_temp,
            air_temp,
            humidity,
            speed_trap,
            is_pit_lap,
            is_safety_car
        )
        VALUES (
            %(race_id)s,
            %(season)s,
            %(circuit)s,
            %(driver)s,
            %(team)s,
            %(lap_number)s,
            %(lap_time)s,
            %(sector1_time)s,
            %(sector2_time)s,
            %(sector3_time)s,
            %(compound)s,
            %(tire_life)s,
            %(track_temp)s,
            %(air_temp)s,
            %(humidity)s,
            %(speed_trap)s,
            %(is_pit_lap)s,
            %(is_safety_car)s
        )
    """

    cursor = conn.cursor()

    try:
        cursor.executemany(insert_query, rows)
        conn.commit()
        inserted = cursor.rowcount
        logger.info(f"Inserted {inserted} rows into raw_lap_data")
        return inserted

    except Exception as e:
        conn.rollback()
        logger.error(f"Insert failed, transaction rolled back: {e}")
        return 0

    finally:
        cursor.close()


# ============================================================
# Ingestion Progress Tracker
# ============================================================

def log_ingestion_summary(summary: list[dict]) -> None:
    """
    Prints a clean summary table at the end of the ingestion run
    showing what succeeded, what failed, and how many rows were inserted.

    Args:
        summary: List of result dictionaries built during the pipeline run.
    """
    print("\n")
    print("=" * 70)
    print("INGESTION PIPELINE SUMMARY")
    print("=" * 70)
    print(f"{'Season':<8} {'Circuit':<25} {'Status':<12} {'Rows':>8}")
    print("-" * 70)

    total_rows = 0
    total_success = 0
    total_failed = 0

    for result in summary:
        status_display = "SUCCESS" if result["success"] else "FAILED"
        print(
            f"{result['season']:<8} "
            f"{result['circuit']:<25} "
            f"{status_display:<12} "
            f"{result['rows_inserted']:>8}"
        )
        total_rows += result["rows_inserted"]
        if result["success"]:
            total_success += 1
        else:
            total_failed += 1

    print("-" * 70)
    print(
        f"{'TOTAL':<8} {'':<25} "
        f"S:{total_success} F:{total_failed}  "
        f"{total_rows:>8} rows"
    )
    print("=" * 70)
    print()


# ============================================================
# Main Pipeline Orchestrator
# ============================================================

def run_ingestion_pipeline(
    seasons: list[int] = None,
    circuits: list[str] = None
) -> None:
    """
    Orchestrates the full ingestion pipeline.

    Loops through every season and circuit in configuration,
    loads each race session from FastF1, extracts lap data,
    and inserts into the MySQL Bronze table.

    Args:
        seasons:  Optional override list of seasons to process.
                  If None, reads from config.yaml.
        circuits: Optional override list of circuits to process.
                  If None, reads from config.yaml.
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("F1 DATA INGESTION PIPELINE STARTING")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()

    # Set up FastF1 cache
    cache_path = os.getenv(
        "FASTF1_CACHE_PATH",
        config.get("fastf1_cache_path", "./fastf1_cache")
    )
    setup_fastf1_cache(cache_path)

    # Resolve seasons and circuits to process
    if seasons is None:
        seasons = config["seasons"]

    if circuits is None:
        circuits = [c["name"] for c in config["circuits"]]

    logger.info(f"Seasons to process: {seasons}")
    logger.info(f"Circuits to process: {circuits}")
    logger.info(
        f"Total sessions to attempt: {len(seasons) * len(circuits)}"
    )

    # Open database connection (reuse across all sessions)
    conn = get_mysql_connection()
    cursor = conn.cursor()

    summary = []

    # ---- Main Loop ----
    for season in seasons:
        for circuit in circuits:

            race_id = f"{season}_{circuit.replace(' ', '_')}"
            result = {
                "season": season,
                "circuit": circuit,
                "success": False,
                "rows_inserted": 0
            }

            logger.info(f"--- Processing: {season} {circuit} ---")

            # Skip if already in database
            if check_session_already_ingested(cursor, race_id):
                logger.info(
                    f"SKIPPED — {race_id} already exists in database"
                )
                result["success"] = True
                result["rows_inserted"] = 0
                summary.append(result)
                continue

            # Load session from FastF1
            session = load_race_session(season, circuit)

            if session is None:
                logger.warning(
                    f"SKIPPED — Could not load session for {race_id}"
                )
                summary.append(result)
                # Pause briefly before trying the next session
                time.sleep(2)
                continue

            # Extract lap data from session
            rows = extract_laps_from_session(session, season, circuit)

            if not rows:
                logger.warning(f"SKIPPED — No rows extracted for {race_id}")
                summary.append(result)
                continue

            # Insert into Bronze table
            inserted = insert_laps_to_bronze(rows, conn)

            result["success"] = inserted > 0
            result["rows_inserted"] = inserted
            summary.append(result)

            # Be respectful to the API — pause between sessions
            logger.info(f"Pausing 3 seconds before next session...")
            time.sleep(3)

    # Clean up
    cursor.close()
    conn.close()

    # Print summary
    log_ingestion_summary(summary)

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pipeline completed in {duration}")


# ============================================================
# Entry Point with Argument Parsing
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="F1 Data Ingestion Pipeline — Loads race data into MySQL Bronze layer"
    )

    parser.add_argument(
        "--season",
        type=int,
        help="Process only a specific season (e.g., --season 2024)",
        default=None
    )

    parser.add_argument(
        "--circuit",
        type=str,
        help="Process only a specific circuit (e.g., --circuit Monza)",
        default=None
    )

    args = parser.parse_args()

    # Build override lists from arguments if provided
    seasons_override = [args.season] if args.season else None
    circuits_override = [args.circuit] if args.circuit else None

    run_ingestion_pipeline(
        seasons=seasons_override,
        circuits=circuits_override
    )