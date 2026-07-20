"""
Stage 3 (revised): Extract 2026 FIFA World Cup matches from the same
historical dataset used in Stage 2, since it is kept updated live.
No API key required.
"""

import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def fetch_latest_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_URL)
    df["date"] = pd.to_datetime(df["date"])
    return df




def get_2026_world_cup_matches(df: pd.DataFrame, cutoff_date: str = None) -> pd.DataFrame:
    """
    cutoff_date: if provided, excludes any match on or after this date.
    Use this to reconstruct the pre-final data state even if the source
    dataset has since been updated with later results.
    """
    wc2026 = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"] >= "2026-06-11") &
        (df["date"] <= "2026-07-19")
    ].copy()

    if cutoff_date:
        wc2026 = wc2026[wc2026["date"] < cutoff_date]

    wc2026 = wc2026.sort_values("date").reset_index(drop=True)
    return wc2026

def build_team_tournament_summary(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team-level stats across all 2026 matches played so far (excludes the final by date if not yet played)."""
    teams = pd.unique(matches_df[["home_team", "away_team"]].values.ravel())

    summary = []
    for team in teams:
        home = matches_df[matches_df["home_team"] == team]
        away = matches_df[matches_df["away_team"] == team]

        goals_for = home["home_score"].sum() + away["away_score"].sum()
        goals_against = home["away_score"].sum() + away["home_score"].sum()
        matches_played = len(home) + len(away)

        wins = (home["home_score"] > home["away_score"]).sum() + \
               (away["away_score"] > away["home_score"]).sum()
        draws = (home["home_score"] == home["away_score"]).sum() + \
                (away["away_score"] == away["home_score"]).sum()
        losses = matches_played - wins - draws

        summary.append({
            "team": team,
            "matches_played": matches_played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_diff": goals_for - goals_against,
            "points": wins * 3 + draws,
        })

    return pd.DataFrame(summary).sort_values("points", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Fetching latest match data...")
    all_matches = fetch_latest_results()

    CUTOFF = "2026-07-19"
    wc2026 = get_2026_world_cup_matches(all_matches, cutoff_date=CUTOFF)
    wc2026.to_csv(f"{RAW_DIR}/wc2026_matches_raw.csv", index=False)
    print(f"Found {len(wc2026)} World Cup 2026 rows (played + scheduled).")

    # Only matches that have actually been played (both scores present)
    played = wc2026.dropna(subset=["home_score", "away_score"]).copy()
    played.to_csv(f"{PROCESSED_DIR}/wc2026_matches_played.csv", index=False)
    print(f"{len(played)} matches actually played so far.")
    print(played.tail(5)[["date", "home_team", "away_team", "home_score", "away_score"]])

    # Keep the unplayed/upcoming rows separately — useful later to auto-detect the two finalists
    upcoming = wc2026[wc2026["home_score"].isna()].copy()
    upcoming.to_csv(f"{PROCESSED_DIR}/wc2026_matches_upcoming.csv", index=False)
    print(f"\n{len(upcoming)} match(es) not yet played:")
    print(upcoming[["date", "home_team", "away_team"]])

    team_summary = build_team_tournament_summary(played)
    team_summary.to_csv(f"{PROCESSED_DIR}/wc2026_team_summary.csv", index=False)
    print(f"\nTeam summary saved -> {PROCESSED_DIR}/wc2026_team_summary.csv")
    print(team_summary.head(10))