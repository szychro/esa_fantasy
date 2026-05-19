import pandas as pd

POSITION_MAP = {
    "GK": "GK",
    "CB": "DF",
    "LB": "DF",
    "RB": "DF",
    "DM": "MF",
    "CM": "MF",
    "AM": "MF",
    "RM": "MF",
    "LM": "MF",
    "LW": "FW",
    "RW": "FW",
    "FW": "FW",
}


def fantasy_points_calc(df: pd.DataFrame) -> pd.Series:
    league_mask = df["competition"].eq("Ekstraklasa")
    league_df = df.loc[league_mask].copy()

    # Fantasy scoring is defined only for Ekstraklasa matches.
    league_df["position_fantasy"] = league_df["position"].map(POSITION_MAP)

    fpts = pd.Series(0.0, index=league_df.index)

    goal_points = {'GK': 8, 'DF': 6, 'MF': 5, 'FW': 4}
    assist_points = {'GK': 6, 'DF': 4, 'MF': 3, 'FW': 3}
    clean_sheet_points = {'GK': 3, 'DF': 3, 'MF': 1, 'FW': 0}
    win_points = {'GK': 1, 'DF': 1, 'MF': 1, 'FW': 1}
    lost_goals_points = {'GK': -1, 'DF': -1, 'MF': 0, 'FW': 0}
    og_goals_points = {'GK': -3, 'DF': -3, 'MF': -3, 'FW': -3}
    red_card_points = {'GK': -3, 'DF': -3, 'MF': -3, 'FW': -3}
    yellow_card_points = {'GK': -1, 'DF': -1, 'MF': -1, 'FW': -1}
    first_squad_points = {'GK': 1, 'DF': 1, 'MF': 1, 'FW': 1}

    score_parts = league_df["result"].fillna("").str.extract(r"(\d+)[–-](\d+)")
    goals_against = pd.to_numeric(score_parts[1], errors="coerce")
    result_code = league_df["result"].fillna("").str[0]

    fpts += league_df["goals"].fillna(0) * league_df["position_fantasy"].map(goal_points).fillna(0)
    fpts += league_df["assists"].fillna(0) * league_df["position_fantasy"].map(assist_points).fillna(0)
    fpts += league_df["yellow_cards"].fillna(0) * league_df["position_fantasy"].map(yellow_card_points).fillna(0)
    fpts += league_df["red_cards"].fillna(0) * league_df["position_fantasy"].map(red_card_points).fillna(0)
    started_mask = league_df["started"].isin(["Y", "Y*"])
    fpts += started_mask.astype(int) * league_df["position_fantasy"].map(first_squad_points).fillna(0)


    clean_sheet_mask = goals_against.eq(0)
    fpts += clean_sheet_mask.astype(int) * league_df["position_fantasy"].map(clean_sheet_points).fillna(0)

    win_mask = result_code.eq("W")
    fpts += win_mask.astype(int) * league_df["position_fantasy"].map(win_points).fillna(0)

    gk_mask = league_df["position_fantasy"].eq("GK")
    fpts += (league_df["gk_saves"].fillna(0) // 3).where(gk_mask, 0)

    conceded_penalty_units = goals_against.sub(1).clip(lower=0).fillna(0)
    fpts += league_df["position_fantasy"].map(lost_goals_points).fillna(0) * conceded_penalty_units

    fantasy_points = pd.Series(0.0, index=df.index)
    fantasy_points.loc[league_df.index] = fpts
    return fantasy_points


def add_fixture_context_features(df: pd.DataFrame) -> pd.DataFrame:
    league = df.loc[df["competition"].eq("Ekstraklasa")].copy()
    league["date"] = pd.to_datetime(league["date"], errors="coerce")

    team_matches = (
        league[["date", "team", "opponent", "venue", "result"]]
        .dropna(subset=["date", "team", "opponent", "result"])
        .drop_duplicates()
        .sort_values(["team", "date"])
    )

    score_parts = team_matches["result"].str.extract(r"(\d+)[–-](\d+)")
    team_matches["team_goals_for"] = pd.to_numeric(score_parts[0], errors="coerce")
    team_matches["team_goals_against"] = pd.to_numeric(score_parts[1], errors="coerce")
    team_matches["team_clean_sheet"] = team_matches["team_goals_against"].eq(0).astype(int)

    rolling_cols = ["team_goals_for", "team_goals_against", "team_clean_sheet"]
    for col in rolling_cols:
        team_matches[f"{col}_roll5"] = (
            team_matches.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

    team_features = team_matches[
        [
            "date",
            "team",
            "team_goals_for_roll5",
            "team_goals_against_roll5",
            "team_clean_sheet_roll5",
        ]
    ]

    opponent_features = team_features.rename(
        columns={
            "team": "opponent",
            "team_goals_for_roll5": "opponent_goals_for_roll5",
            "team_goals_against_roll5": "opponent_goals_against_roll5",
            "team_clean_sheet_roll5": "opponent_clean_sheet_roll5",
        }
    )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.merge(team_features, on=["date", "team"], how="left")
    df = df.merge(opponent_features, on=["date", "opponent"], how="left")
    return df

def rolling_features(df, windows=[3, 5, 10]):
    stat_cols = ["fantasy_points", "goals", "assists", "minutes",
                 "shots", "xg", "gk_saves", "gk_save_pct", "started_num"]

    df["position_fantasy"] = df["position"].map(POSITION_MAP)
    df["started_num"] = df["started"].isin(["Y", "Y*"]).astype(int)
    
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    for w in windows:
        for col in stat_cols:
            df[f"{col}_roll{w}"] = (
                df.groupby("player_id")[col]
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    df["is_home"] = df["venue"].map({"Home": 1, "Away": 0}).fillna(0.5)
    df["position_enc"] = df["position_fantasy"].map({"GK": 0, "DF": 1, "MF": 2, "FW": 3})

    return df


def main() -> None:
    df_original = pd.read_csv("ekstraklasa_players.csv")
    df = df_original.copy()
    df["fantasy_points"] = fantasy_points_calc(df)
    df = df.sort_values(["player_id", "date"])
    df = add_fixture_context_features(df)
    df = rolling_features(df)
    df.to_csv("ekstraklasa_players_prepped.csv", index=False)


if __name__ == "__main__":
    main()
