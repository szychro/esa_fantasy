from pathlib import Path

import joblib
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


def load_next_fixtures(next_gw_date: str) -> pd.DataFrame:
    if not FIXTURES_PATH.exists():
        return pd.DataFrame(columns=["team", "fixture_date", "next_opponent", "next_venue", "fixture_is_home"])

    fixtures = pd.read_csv(FIXTURES_PATH)
    fixtures = fixtures.dropna(subset=["team", "opponent"])
    if fixtures.empty:
        return pd.DataFrame(columns=["team", "fixture_date", "next_opponent", "next_venue", "fixture_is_home"])

    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    target_date = pd.to_datetime(next_gw_date)

    next_fixtures = fixtures.loc[fixtures["date"] >= target_date].copy()
    next_fixtures = next_fixtures.sort_values(["team", "date"]).groupby("team").first().reset_index()
    next_fixtures["fixture_is_home"] = next_fixtures["venue"].map({"Home": 1, "Away": 0}).fillna(0.5)
    next_fixtures = next_fixtures.rename(
        columns={
            "date": "fixture_date",
            "opponent": "next_opponent",
            "venue": "next_venue",
        }
    )
    return next_fixtures[["team", "fixture_date", "next_opponent", "next_venue", "fixture_is_home"]]


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
    next_gw_date: str,
) -> dict[str, pd.DataFrame]:
    """
    For each player, take their most recent row and predict expected points
    for the next available league fixture.
    """
    results: dict[str, pd.DataFrame] = {}
    league_df = df.loc[df["competition"].eq("Ekstraklasa")].copy()
    latest = league_df.sort_values("date").groupby("player_id").last().reset_index()
    fixtures = load_next_fixtures(next_gw_date)
    latest = latest.merge(fixtures, on="team", how="left")
    if not fixtures.empty:
        fixture_mask = latest["fixture_is_home"].notna()
        latest.loc[fixture_mask, "is_home"] = latest.loc[fixture_mask, "fixture_is_home"].astype(float)
        latest = latest.drop(columns=OPPONENT_FEATURES, errors="ignore")
        latest = latest.merge(latest_team_strength(league_df), on="next_opponent", how="left")

    eligible = latest.loc[latest["minutes_roll5"] >= MIN_MINUTES_ROLL5].copy()

    for pos, payload in models.items():
        model = payload["model"]
        features = payload["features"]

        pos_players = eligible.loc[eligible["position_fantasy"] == pos].copy()
        pos_players = pos_players.dropna(subset=features)

        if pos_players.empty:
            print(f"\nNo eligible players available for {pos} on {next_gw_date}")
            continue

        pos_players["predicted_pts"] = model.predict(pos_players[features])

        top5 = (
            pos_players.sort_values("predicted_pts", ascending=False)
            .head(5)[
                [
                    "player",
                    "team",
                    "next_opponent",
                    "next_venue",
                    "minutes_roll5",
                    "predicted_pts",
                ]
            ]
        )

        results[pos] = top5
        print(f"\nTop 5 {pos} for {next_gw_date}:")
        print(top5.to_string(index=False))

    return results


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    models = joblib.load(MODEL_PATH)
    predict_top5_per_position(df, models, next_gw_date="2026-04-23")


if __name__ == "__main__":
    main()
