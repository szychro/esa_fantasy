from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("ekstraklasa_players_prepped.csv")
MODEL_PATH = Path("models.joblib")

BASE_FEATURES = [
    "is_home",
    "started_num_roll5",
    "minutes_roll3",
    "minutes_roll5",
    "fantasy_points_roll3",
    "fantasy_points_roll5",
    "fantasy_points_roll10",
    "opponent_goals_for_roll5",
    "opponent_goals_against_roll5",
    "opponent_clean_sheet_roll5",
]

POSITION_FEATURES = {
    "GK": BASE_FEATURES + ["gk_saves_roll5", "gk_save_pct_roll5"],
    "DF": BASE_FEATURES + ["goals_roll5", "assists_roll5", "xg_roll5"],
    "MF": BASE_FEATURES + ["goals_roll5", "assists_roll5", "shots_roll5", "xg_roll5"],
    "FW": BASE_FEATURES + ["goals_roll5", "shots_roll5", "xg_roll5", "assists_roll5"],
}


def train_models(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    models: dict[str, dict[str, object]] = {}

    for pos, features in POSITION_FEATURES.items():
        pos_df = df.loc[df["position_fantasy"] == pos].copy()
        pos_df = pos_df.dropna(subset=features + ["fantasy_points", "player_id"])

        if pos_df.empty:
            print(f"{pos} -> skipped (no rows after filtering)")
            continue

        if pos_df["player_id"].nunique() < 2:
            print(f"{pos} -> skipped (not enough unique players for GroupShuffleSplit)")
            continue

        X = pos_df[features]
        y = pos_df["fantasy_points"]
        groups = pos_df["player_id"]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = GradientBoostingRegressor(
            n_estimators=600,
            learning_rate=0.1,
            max_depth=2,
            min_samples_leaf=25,
            subsample=0.9,
            loss="squared_error",
            random_state=42)
        model.fit(X_train, y_train)

        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"{pos} -> MAE: {mae:.2f} pts ({len(pos_df)} rows, {pos_df['player_id'].nunique()} players)")

        models[pos] = {
            "model": model,
            "features": features,
            "mae": mae,
        }

    return models


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    models = train_models(df)
    joblib.dump(models, MODEL_PATH)
    print(f"Saved {len(models)} model(s) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
