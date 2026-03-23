# src/ingestion/fetch_2026_validation_data.py

"""
2026 Validation Data Ingestion

This script fetches the real 2026 race data for the circuits where
races have already occurred. This data is NOT used for training.
It is stored separately and used exclusively to validate the accuracy
of our simulation predictions in Phase 8B.

Currently available 2026 races:
    - Race 1: Australian Grand Prix — Melbourne
    - Race 2: Chinese Grand Prix — Shanghai

As more 2026 races occur during the season, add them to
validation_circuits_2026 in config/config.yaml and re-run
this script. The pipeline is idempotent and will skip
already-loaded sessions automatically.

Usage:
    python src/ingestion/fetch_2026_validation_data.py
"""

import sys
import os
import time

import fastf1
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from db_connection import get_mysql_connection
from config_loader import load_config
from logger import get_logger

logger = get_logger("ingestion_2026")


def fetch_and_store_2026_race(
    circuit: str,
    conn,
    driver_lookup: dict = None
) -> dict:
    """
    Fetches a single 2026 race session and stores it in raw_lap_data.
    Uses driver_lookup fallback to handle 2026 lineup changes.
    """
    result = {
        "circuit": circuit,
        "season": 2026,
        "success": False,
        "rows_inserted": 0
    }

    race_id = f"2026_{circuit.replace(' ', '_')}"

    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM raw_lap_data WHERE race_id = %s",
        (race_id,)
    )
    if cursor.fetchone()[0] > 0:
        logger.info(f"SKIPPED — {race_id} already in database")
        result["success"] = True
        cursor.close()
        return result

    try:
        logger.info(f"Loading 2026 {circuit} Race from FastF1")
        session = fastf1.get_session(2026, circuit, "R")
        session.load(
            laps=True,
            telemetry=True,
            weather=True,
            messages=False
        )
        logger.info(
            f"2026 {circuit} loaded: {len(session.laps)} laps"
        )
    except KeyError as e:
        logger.warning(
            f"Driver number KeyError loading 2026 {circuit}: {e}. "
            f"Trying laps-only load."
        )
        try:
            session = fastf1.get_session(2026, circuit, "R")
            session.load(
                laps=True,
                telemetry=False,
                weather=True,
                messages=False
            )
            logger.info(
                f"2026 {circuit} laps-only load successful"
            )
        except Exception as e2:
            logger.error(
                f"All load attempts failed for 2026 {circuit}: {e2}"
            )
            cursor.close()
            return result

    except Exception as e:
        logger.error(f"Failed to load 2026 {circuit}: {e}")
        cursor.close()
        return result

    from fetch_fastf1_data import extract_laps_from_session
    rows = extract_laps_from_session(
        session,
        2026,
        circuit,
        driver_lookup=driver_lookup
    )

    if not rows:
        logger.warning(f"No rows extracted for 2026 {circuit}")
        cursor.close()
        return result

    insert_query = """
        INSERT INTO raw_lap_data (
            race_id, season, circuit, driver, team,
            lap_number, lap_time, sector1_time, sector2_time,
            sector3_time, compound, tire_life, track_temp,
            air_temp, humidity, speed_trap,
            is_pit_lap, is_safety_car
        )
        VALUES (
            %(race_id)s, %(season)s, %(circuit)s, %(driver)s,
            %(team)s, %(lap_number)s, %(lap_time)s,
            %(sector1_time)s, %(sector2_time)s, %(sector3_time)s,
            %(compound)s, %(tire_life)s, %(track_temp)s,
            %(air_temp)s, %(humidity)s, %(speed_trap)s,
            %(is_pit_lap)s, %(is_safety_car)s
        )
    """

    try:
        cursor.executemany(insert_query, rows)
        conn.commit()
        result["rows_inserted"] = cursor.rowcount
        result["success"] = True
        logger.info(
            f"Inserted {result['rows_inserted']} rows for 2026 {circuit}"
        )
    except Exception as e:
        conn.rollback()
        logger.error(f"Insert failed for 2026 {circuit}: {e}")

    cursor.close()
    return result


def run_2026_validation_ingestion():
    """
    Main entry point for fetching all available 2026 race data.
    """
    config = load_config()

    cache_path = os.getenv("FASTF1_CACHE_PATH", "./fastf1_cache")
    os.makedirs(cache_path, exist_ok=True)
    fastf1.Cache.enable_cache(cache_path)

    validation_circuits = [
        c["name"] for c in config.get("validation_circuits_2026", [])
    ]

    if not validation_circuits:
        logger.warning(
            "No 2026 validation circuits found in config.yaml"
        )
        return

    logger.info(f"2026 validation circuits to fetch: {validation_circuits}")

    conn = get_mysql_connection()

    # Build 2026 driver lookup before fetching sessions
    from fetch_fastf1_data import build_driver_lookup
    driver_lookup_2026 = build_driver_lookup(2026, conn)
    logger.info(
        f"2026 driver lookup ready: {len(driver_lookup_2026)} drivers"
    )

    results = []

    for circuit in validation_circuits:
        result = fetch_and_store_2026_race(
            circuit,
            conn,
            driver_lookup=driver_lookup_2026
        )
        results.append(result)
        time.sleep(3)

    conn.close()

    print("\n" + "=" * 60)
    print("2026 VALIDATION DATA INGESTION SUMMARY")
    print("=" * 60)
    for r in results:
        status = "SUCCESS" if r["success"] else "FAILED"
        print(
            f"  2026 {r['circuit']:<20} {status:<10} "
            f"{r['rows_inserted']:>6} rows"
        )
    print("=" * 60)


if __name__ == "__main__":
    run_2026_validation_ingestion()