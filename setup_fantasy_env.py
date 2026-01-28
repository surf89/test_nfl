"""
setup_fantasy_env.py

Run this once to:
- Show required packages
- Create fantasy_config.json with your ESPN + season settings
"""

import json
from getpass import getpass

REQUIREMENTS = [
    "streamlit",
    "nfl_data_py",
    "espn-api",
    "pandas",
    "numpy",
    "scikit-learn",
    "altair",
]

def print_install_instructions():
    print("=== Install these packages (once) ===")
    print("pip install " + " ".join(REQUIREMENTS))
    print("=====================================\n")


def main():
    print_install_instructions()

    print("Let's create your fantasy_config.json\n")

    league_id = input("ESPN League ID: ").strip()
    year = input("ESPN League Year (e.g., 2025): ").strip()

    target_season = input("Target NFL season for projections (e.g., 2025): ").strip()
    target_week = input("Target NFL week to project (e.g., 1): ").strip()

    print("\nNow we need your ESPN cookies (kept local only).")
    espn_s2 = getpass("espn_s2 cookie: ").strip()
    swid = getpass("swid (including braces {}): ").strip()

    config = {
        "ESPN_LEAGUE_ID": int(league_id),
        "ESPN_YEAR": int(year),
        "ESPN_S2": espn_s2,
        "ESPN_SWID": swid,
        "SEASONS": [2022, 2023, 2024],  # default; engine will auto-roll
        "TARGET_SEASON": int(target_season),
        "TARGET_WEEK": int(target_week),
        "RANDOM_STATE": 42,
        "EMAIL_ENABLED": False,
        "EMAIL_FROM": "geosurf89@gmail.com.com",
        "EMAIL_TO": "geosurf89@gmail.com",
        "SMTP_SERVER": "smtp.gmail.com",
        "SMTP_PORT": 587,
        "SMTP_USER": "you@example.com",
        "SMTP_PASS": "your_app_password"
    }

    with open("fantasy_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nfantasy_config.json created.")
    print("Next step:")
    print("  pip install -r requirements.txt")
    print("  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
