# src/pipelines/seed_driver_reference.py

"""
Driver Reference Seeder

Populates the driver_reference table with all drivers across
seasons 2023, 2024, 2025, and 2026.

This table serves two purposes:
1. Provides fallback driver information when FastF1 mapping fails
2. Gives the ML pipeline clean, consistent driver identifiers

Usage:
    python src/pipelines/seed_driver_reference.py
"""

import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from db_connection import get_mysql_connection
from logger import get_logger

logger = get_logger("seed_drivers")


# ============================================================
# Complete Driver Registry — 2023 to 2026
# ============================================================
# Format: (abbreviation, number, full_name, nationality,
#          season, team, is_full_season, notes)
#
# is_full_season = False for replacement or part-season drivers
# ============================================================

DRIVER_REGISTRY = [

    # --------------------------------------------------------
    # 2023 SEASON — 10 Teams, 20 drivers
    # Constructor names as they were in 2023
    # --------------------------------------------------------
    ("VER",  1,  "Max Verstappen",        "Dutch",         2023, "Red Bull Racing",   True,  None),
    ("PER",  11, "Sergio Perez",          "Mexican",       2023, "Red Bull Racing",   True,  None),
    ("HAM",  44, "Lewis Hamilton",        "British",       2023, "Mercedes",          True,  None),
    ("RUS",  63, "George Russell",        "British",       2023, "Mercedes",          True,  None),
    ("LEC",  16, "Charles Leclerc",       "Monegasque",    2023, "Ferrari",           True,  None),
    ("SAI",  55, "Carlos Sainz",          "Spanish",       2023, "Ferrari",           True,  None),
    ("NOR",   4, "Lando Norris",          "British",       2023, "McLaren",           True,  None),
    ("PIA",  81, "Oscar Piastri",         "Australian",    2023, "McLaren",           True,  None),
    ("ALO",  14, "Fernando Alonso",       "Spanish",       2023, "Aston Martin",      True,  None),
    ("STR",  18, "Lance Stroll",          "Canadian",      2023, "Aston Martin",      True,  None),
    ("OCO",  31, "Esteban Ocon",          "French",        2023, "Alpine",            True,  None),
    ("GAS",  10, "Pierre Gasly",          "French",        2023, "Alpine",            True,  None),
    ("ALB",  23, "Alexander Albon",       "Thai",          2023, "Williams",          True,  None),
    ("SAR",   2, "Logan Sargeant",        "American",      2023, "Williams",          True,  None),
    ("TSU",  22, "Yuki Tsunoda",          "Japanese",      2023, "AlphaTauri",        True,  None),
    ("DEV",  21, "Nyck de Vries",         "Dutch",         2023, "AlphaTauri",        False, "Replaced by Ricciardo from Hungarian GP"),
    ("RIC",   3, "Daniel Ricciardo",      "Australian",    2023, "AlphaTauri",        False, "Replaced de Vries from Hungarian GP; injured at Dutch GP"),
    ("LAW",  40, "Liam Lawson",           "New Zealander", 2023, "AlphaTauri",        False, "Replaced injured Ricciardo from Dutch GP to Singapore GP"),
    ("BOT",  77, "Valtteri Bottas",       "Finnish",       2023, "Alfa Romeo",        True,  None),
    ("ZHO",  24, "Zhou Guanyu",           "Chinese",       2023, "Alfa Romeo",        True,  None),
    ("HUL",  27, "Nico Hulkenberg",       "German",        2023, "Haas",              True,  None),
    ("MAG",  20, "Kevin Magnussen",       "Danish",        2023, "Haas",              True,  None),

    # --------------------------------------------------------
    # 2024 SEASON — 10 Teams, 20 drivers
    # AlphaTauri renamed to RB / Visa Cash App RB
    # Alfa Romeo renamed to Sauber (Kick Sauber)
    # --------------------------------------------------------
    ("VER",   1, "Max Verstappen",        "Dutch",         2024, "Red Bull Racing",   True,  None),
    ("PER",  11, "Sergio Perez",          "Mexican",       2024, "Red Bull Racing",   True,  None),
    ("HAM",  44, "Lewis Hamilton",        "British",       2024, "Mercedes",          True,  None),
    ("RUS",  63, "George Russell",        "British",       2024, "Mercedes",          True,  None),
    ("LEC",  16, "Charles Leclerc",       "Monegasque",    2024, "Ferrari",           True,  None),
    ("SAI",  55, "Carlos Sainz",          "Spanish",       2024, "Ferrari",           True,  None),
    ("NOR",   4, "Lando Norris",          "British",       2024, "McLaren",           True,  None),
    ("PIA",  81, "Oscar Piastri",         "Australian",    2024, "McLaren",           True,  None),
    ("ALO",  14, "Fernando Alonso",       "Spanish",       2024, "Aston Martin",      True,  None),
    ("STR",  18, "Lance Stroll",          "Canadian",      2024, "Aston Martin",      True,  None),
    ("OCO",  31, "Esteban Ocon",          "French",        2024, "Alpine",            True,  None),
    ("GAS",  10, "Pierre Gasly",          "French",        2024, "Alpine",            True,  None),
    ("ALB",  23, "Alexander Albon",       "Thai",          2024, "Williams",          True,  None),
    ("SAR",   2, "Logan Sargeant",        "American",      2024, "Williams",          False, "Replaced by Colapinto from Italian GP"),
    ("COL",  43, "Franco Colapinto",      "Argentine",     2024, "Williams",          False, "Replaced Sargeant from Italian GP onwards"),
    ("TSU",  22, "Yuki Tsunoda",          "Japanese",      2024, "RB",                True,  None),
    ("RIC",   3, "Daniel Ricciardo",      "Australian",    2024, "RB",                False, "Replaced by Lawson from United States GP"),
    ("LAW",  40, "Liam Lawson",           "New Zealander", 2024, "RB",                False, "Replaced Ricciardo from United States GP"),
    ("BOT",  77, "Valtteri Bottas",       "Finnish",       2024, "Kick Sauber",       True,  None),
    ("ZHO",  24, "Zhou Guanyu",           "Chinese",       2024, "Kick Sauber",       True,  None),
    ("HUL",  27, "Nico Hulkenberg",       "German",        2024, "Haas",              True,  None),
    ("MAG",  20, "Kevin Magnussen",       "Danish",        2024, "Haas",              True,  None),

    # --------------------------------------------------------
    # 2025 SEASON — 10 Teams, 20 drivers
    # Major changes:
    #   - Hamilton to Ferrari, replaced by Antonelli at Mercedes
    #   - Sainz to Williams (replaced Sargeant/Colapinto slot)
    #   - Lawson promoted to Red Bull (replaced Perez)
    #   - Tsunoda initially at Racing Bulls; swapped to Red Bull
    #     after Chinese GP replacing Lawson; Lawson back to Racing Bulls
    #   - Hadjar debut at Racing Bulls (replaced Lawson initially)
    #   - Hulkenberg to Sauber (now transitioning to Audi)
    #   - Bortoleto debut at Sauber alongside Hulkenberg
    #   - Haas all new: Bearman + Ocon
    #   - Colapinto replaced Doohan at Alpine from Miami GP
    # --------------------------------------------------------
    ("VER",   1, "Max Verstappen",        "Dutch",         2025, "Red Bull Racing",   True,  None),
    ("LAW",  30, "Liam Lawson",           "New Zealander", 2025, "Red Bull Racing",   False, "Started season; replaced by Tsunoda after 2 races"),
    ("TSU",  22, "Yuki Tsunoda",          "Japanese",      2025, "Red Bull Racing",   False, "Promoted from Racing Bulls after Chinese GP"),
    ("RUS",  63, "George Russell",        "British",       2025, "Mercedes",          True,  None),
    ("ANT",  12, "Kimi Antonelli",        "Italian",       2025, "Mercedes",          True,  None),
    ("LEC",  16, "Charles Leclerc",       "Monegasque",    2025, "Ferrari",           True,  None),
    ("HAM",  44, "Lewis Hamilton",        "British",       2025, "Ferrari",           True,  None),
    ("NOR",   4, "Lando Norris",          "British",       2025, "McLaren",           True,  None),
    ("PIA",  81, "Oscar Piastri",         "Australian",    2025, "McLaren",           True,  None),
    ("ALO",  14, "Fernando Alonso",       "Spanish",       2025, "Aston Martin",      True,  None),
    ("STR",  18, "Lance Stroll",          "Canadian",      2025, "Aston Martin",      True,  None),
    ("GAS",  10, "Pierre Gasly",          "French",        2025, "Alpine",            True,  None),
    ("DOO",   7, "Jack Doohan",           "Australian",    2025, "Alpine",            False, "Replaced by Colapinto after Miami GP"),
    ("COL",  43, "Franco Colapinto",      "Argentine",     2025, "Alpine",            False, "Replaced Doohan from Miami GP onwards"),
    ("ALB",  23, "Alexander Albon",       "Thai",          2025, "Williams",          True,  None),
    ("SAI",  55, "Carlos Sainz",          "Spanish",       2025, "Williams",          True,  None),
    ("TSU",  22, "Yuki Tsunoda",          "Japanese",      2025, "Racing Bulls",      False, "Started season; promoted to Red Bull after Chinese GP"),
    ("HAD",  6,  "Isack Hadjar",          "French",        2025, "Racing Bulls",      True,  None),
    ("LAW",  30, "Liam Lawson",           "New Zealander", 2025, "Racing Bulls",      False, "Returned to Racing Bulls after being replaced at Red Bull"),
    ("HUL",  27, "Nico Hulkenberg",       "German",        2025, "Sauber",            True,  None),
    ("BOR",   5, "Gabriel Bortoleto",     "Brazilian",     2025, "Sauber",            True,  None),
    ("OCO",  31, "Esteban Ocon",          "French",        2025, "Haas",              True,  None),
    ("BEA",  87, "Oliver Bearman",        "British",       2025, "Haas",              True,  None),

    # --------------------------------------------------------
    # 2026 SEASON — 11 Teams, 22 drivers
    # NEW in 2026:
    #   - Cadillac debut (11th team): Perez + Bottas
    #   - Sauber rebranded as Audi works team
    #   - Alpine now uses Mercedes customer engines
    #   - Colapinto replaces Doohan permanently at Alpine
    #   - Hadjar promoted to Red Bull; replaced by Lindblad at Racing Bulls
    #   - Tsunoda dropped to reserve; Hadjar takes Red Bull seat
    #   - Bahrain and Saudi Arabian GPs cancelled for 2026
    # --------------------------------------------------------
    ("VER",   1, "Max Verstappen",        "Dutch",         2026, "Red Bull Racing",   True,  None),
    ("HAD",   6, "Isack Hadjar",          "French",        2026, "Red Bull Racing",   True,  None),
    ("RUS",  63, "George Russell",        "British",       2026, "Mercedes",          True,  None),
    ("ANT",  12, "Kimi Antonelli",        "Italian",       2026, "Mercedes",          True,  None),
    ("LEC",  16, "Charles Leclerc",       "Monegasque",    2026, "Ferrari",           True,  None),
    ("HAM",  44, "Lewis Hamilton",        "British",       2026, "Ferrari",           True,  None),
    ("NOR",   4, "Lando Norris",          "British",       2026, "McLaren",           True,  None),
    ("PIA",  81, "Oscar Piastri",         "Australian",    2026, "McLaren",           True,  None),
    ("ALO",  14, "Fernando Alonso",       "Spanish",       2026, "Aston Martin",      True,  None),
    ("STR",  18, "Lance Stroll",          "Canadian",      2026, "Aston Martin",      True,  None),
    ("GAS",  10, "Pierre Gasly",          "French",        2026, "Alpine",            True,  None),
    ("COL",  43, "Franco Colapinto",      "Argentine",     2026, "Alpine",            True,  None),
    ("ALB",  23, "Alexander Albon",       "Thai",          2026, "Williams",          True,  None),
    ("SAI",  55, "Carlos Sainz",          "Spanish",       2026, "Williams",          True,  None),
    ("LAW",  30, "Liam Lawson",           "New Zealander", 2026, "Racing Bulls",      True,  None),
    ("LIN",  45, "Arvid Lindblad",        "Swedish",       2026, "Racing Bulls",      True,  None),
    ("HUL",  27, "Nico Hulkenberg",       "German",        2026, "Audi",              True,  None),
    ("BOR",   5, "Gabriel Bortoleto",     "Brazilian",     2026, "Audi",              True,  None),
    ("OCO",  31, "Esteban Ocon",          "French",        2026, "Haas",              True,  None),
    ("BEA",  87, "Oliver Bearman",        "British",       2026, "Haas",              True,  None),
    ("PER",  11, "Sergio Perez",          "Mexican",       2026, "Cadillac",          True,  None),
    ("BOT",  77, "Valtteri Bottas",       "Finnish",       2026, "Cadillac",          True,  None),
]


def seed_driver_reference():
    """
    Inserts all driver records into the driver_reference table.
    Uses INSERT IGNORE so it is safe to re-run without creating
    duplicates. Existing records are preserved.
    """
    conn = get_mysql_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT IGNORE INTO driver_reference (
            abbreviation,
            driver_number,
            full_name,
            nationality,
            season,
            team,
            is_full_season,
            notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    inserted = 0
    skipped = 0

    for record in DRIVER_REGISTRY:
        try:
            cursor.execute(insert_query, record)
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            logger.warning(
                f"Could not insert {record[0]} {record[4]}: {e}"
            )

    conn.commit()

    logger.info(
        f"Driver reference seeded: {inserted} inserted, {skipped} skipped"
    )

    # Print summary by season
    print("\n" + "=" * 60)
    print("DRIVER REFERENCE — SEASON SUMMARY")
    print("=" * 60)

    for season in [2023, 2024, 2025, 2026]:
        cursor.execute(
            "SELECT COUNT(*) FROM driver_reference WHERE season = %s",
            (season,)
        )
        count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*) FROM driver_reference
            WHERE season = %s AND is_full_season = FALSE
            """,
            (season,)
        )
        part_season = cursor.fetchone()[0]

        print(
            f"  {season}: {count} drivers total  "
            f"({part_season} part-season or replacement)"
        )

    print("=" * 60)

    # Show 2026 grid explicitly — 11 teams, 22 drivers
    print("\n2026 Full Driver Lineup (Validation Season):")
    print(f"  {'#':<5} {'ABB':<6} {'Driver':<25} {'Team'}")
    print("  " + "-" * 55)
    cursor.execute(
        """
        SELECT abbreviation, driver_number, full_name, team
        FROM driver_reference
        WHERE season = 2026 AND is_full_season = TRUE
        ORDER BY team, driver_number
        """
    )
    for row in cursor.fetchall():
        print(f"  #{row[1]:<4} {row[0]:<6} {row[2]:<25} {row[3]}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    seed_driver_reference()