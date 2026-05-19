from pathlib import Path
import argparse
from typing import Optional

import pandas as pd


DATA_PATH = Path("ekstraklasa_players_prepped.csv")
MODEL_PATH = Path("models.joblib")
FIXTURES_PATH = Path("ekstraklasa_fixtures.csv")
MIN_MINUTES_ROLL5 = 30


OPPONENT_FEATURES = [
    "opponent_goals_for_roll5",
    "opponent_goals_against_roll5",
    "opponent_clean_sheet_roll5",
]

FIXTURE_COLUMNS = ["team", "fixture_date", "next_opponent", "next_venue", "fixture_is_home"]


def empty_fixtures() -> pd.DataFrame:
    return pd.DataFrame(columns=FIXTURE_COLUMNS)


def load_next_fixtures(next_gw_date: Optional[str] = None) -> pd.DataFrame:
    if not FIXTURES_PATH.exists():
        return empty_fixtures()

    fixtures = pd.read_csv(FIXTURES_PATH)
    fixtures = fixtures.dropna(subset=["team", "opponent"])
    if fixtures.empty:
        return empty_fixtures()

    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = fixtures.dropna(subset=["date"])
    if fixtures.empty:
        return empty_fixtures()

    target_date = pd.to_datetime(next_gw_date) if next_gw_date else pd.Timestamp.today().normalize()

    next_fixtures = fixtures.loc[fixtures["date"] >= target_date].copy()
    if next_fixtures.empty:
        return empty_fixtures()

    next_fixtures = next_fixtures.sort_values(["team", "date"]).groupby("team").first().reset_index()
    next_fixtures["fixture_is_home"] = next_fixtures["venue"].map({"Home": 1, "Away": 0}).fillna(0.5)
    next_fixtures = next_fixtures.rename(
        columns={
            "date": "fixture_date",
            "opponent": "next_opponent",
            "venue": "next_venue",
        }
    )
    return next_fixtures[FIXTURE_COLUMNS]


def latest_team_strength(df: pd.DataFrame) -> pd.DataFrame:
    strength_cols = [
        "team_goals_for_roll5",
        "team_goals_against_roll5",
        "team_clean_sheet_roll5",
    ]
    if not set(strength_cols).issubset(df.columns):
        return pd.DataFrame(columns=["next_opponent", *OPPONENT_FEATURES])

    league = df.loc[df["competition"].eq("Ekstraklasa")].copy()
    league["date"] = pd.to_datetime(league["date"], errors="coerce")
    latest_strength = (
        league.dropna(subset=["team"])
        .sort_values("date")
        .groupby("team")
        .last()
        .reset_index()[["team", *strength_cols]]
    )
    latest_strength = latest_strength.rename(
        columns={
            "team": "next_opponent",
            "team_goals_for_roll5": "opponent_goals_for_roll5",
            "team_goals_against_roll5": "opponent_goals_against_roll5",
            "team_clean_sheet_roll5": "opponent_clean_sheet_roll5",
        }
    )
    return latest_strength


def predict_top5_per_position(
    df: pd.DataFrame,
    models: dict[str, dict[str, object]],
    next_gw_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    For each player, take their most recent row and predict expected points
    for the next available league fixture.
    """
    results: dict[str, pd.DataFrame] = {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    league_df = df.loc[df["competition"].eq("Ekstraklasa")].copy()
    fixtures = load_next_fixtures(next_gw_date)
    if fixtures.empty:
        print("\nNo upcoming fixtures found.")
        return results

    active_teams = set(fixtures["team"])
    latest = (
        df.dropna(subset=["date", "team"])
        .sort_values("date")
        .groupby("player_id")
        .last()
        .reset_index()
    )
    latest = latest.loc[latest["team"].isin(active_teams)].copy()
    latest = latest.merge(fixtures, on="team", how="left")
    fixture_mask = latest["fixture_is_home"].notna()
    latest.loc[fixture_mask, "is_home"] = latest.loc[fixture_mask, "fixture_is_home"].astype(float)
    latest = latest.drop(columns=OPPONENT_FEATURES, errors="ignore")
    latest = latest.merge(latest_team_strength(league_df), on="next_opponent", how="left")

    eligible = latest.loc[latest["minutes_roll5"] >= MIN_MINUTES_ROLL5].copy()
    fixture_dates = fixtures["fixture_date"].dt.strftime("%Y-%m-%d").dropna().unique()
    fixture_label = ", ".join(sorted(fixture_dates))

    for pos, payload in models.items():
        model = payload["model"]
        features = payload["features"]

        pos_players = eligible.loc[eligible["position_fantasy"] == pos].copy()
        pos_players = pos_players.dropna(subset=features)

        if pos_players.empty:
            print(f"\nNo eligible players available for {pos} on fixtures from {fixture_label}")
            continue

        pos_players["predicted_pts"] = model.predict(pos_players[features])

        top5 = (
            pos_players.sort_values("predicted_pts", ascending=False)
            .head(5)[
                [
                    "player",
                    "team",
                    "fixture_date",
                    "next_opponent",
                    "next_venue",
                    "minutes_roll5",
                    "predicted_pts",
                ]
            ]
        )
        top5["fixture_date"] = top5["fixture_date"].dt.strftime("%Y-%m-%d")

        results[pos] = top5
        print(f"\nTop 5 {pos} for fixtures from {fixture_label}:")
        print(top5.to_string(index=False))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top Ekstraklasa fantasy players.")
    parser.add_argument(
        "--date",
        dest="next_gw_date",
        default=None,
        help="Only consider fixtures on or after this date (YYYY-MM-DD). Defaults to today.",
    )
    return parser.parse_args()


def main() -> None:
    import joblib

    args = parse_args()
    df = pd.read_csv(DATA_PATH)
    models = joblib.load(MODEL_PATH)
    predict_top5_per_position(df, models, next_gw_date=args.next_gw_date)


if __name__ == "__main__":
    main()
