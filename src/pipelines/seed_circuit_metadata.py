# src/pipelines/seed_circuit_metadata.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_connection import get_mysql_connection
from config_loader import load_config


def seed_circuit_metadata():
    """
    Reads circuit configuration from config.yaml and
    inserts records into the circuit_metadata table.
    Safe to run multiple times due to ON DUPLICATE KEY UPDATE.
    """
    config = load_config()
    circuits = config["circuits"]

    conn = get_mysql_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO circuit_metadata (
            circuit,
            country,
            track_length_km,
            num_corners,
            drs_zones,
            elevation_change_m,
            power_sensitivity_score,
            avg_corner_speed_kmh,
            full_throttle_pct
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            power_sensitivity_score = VALUES(power_sensitivity_score),
            full_throttle_pct       = VALUES(full_throttle_pct),
            avg_corner_speed_kmh    = VALUES(avg_corner_speed_kmh),
            updated_at              = CURRENT_TIMESTAMP
    """

    inserted = 0
    for circuit in circuits:
        cursor.execute(insert_query, (
            circuit["name"],
            circuit["country"],
            circuit["track_length_km"],
            circuit["num_corners"],
            circuit["drs_zones"],
            circuit["elevation_change_m"],
            circuit["power_sensitivity_score"],
            circuit["avg_corner_speed_kmh"],
            circuit["full_throttle_pct"]
        ))
        inserted += 1

    conn.commit()
    print(f"Successfully seeded {inserted} circuit records.")

    # Verify
    cursor.execute("SELECT circuit, power_sensitivity_score FROM circuit_metadata")
    rows = cursor.fetchall()
    print("\nCircuit metadata in database:")
    for row in rows:
        print(f"  {row[0]:<20} Power sensitivity: {row[1]}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    seed_circuit_metadata()