from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

POSITIONS = {"GK": [...], "DF": [...], "MF": [...], "FW": [...]}

# Define features per position
base_features = ["is_home", "minutes_roll5", "fantasy_points_roll3",
                 "fantasy_points_roll5", "fantasy_points_roll10"]

pos_features = {
    "GK":  base_features + ["gk_saves_roll5", "gk_save_pct_roll5", "gk_goals_against_roll5"],
    "DF":  base_features + ["goals_roll5", "assists_roll5", "xg_roll5"],
    "MF":  base_features + ["goals_roll5", "assists_roll5", "shots_roll5", "xg_roll5"],
    "FW":  base_features + ["goals_roll5", "shots_roll5", "xg_roll5", "assists_roll5"],
}

models = {}

for pos, features in pos_features.items():
    pos_df = df[df["position"] == pos].dropna(subset=features + ["fantasy_points"])
    
    X = pos_df[features]
    y = pos_df["fantasy_points"]
    groups = pos_df["player_id"]  # split by player, not random row

    # GroupShuffleSplit ensures a player is fully in train OR test
    # This prevents data leakage from rolling features
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"{pos} → MAE: {mae:.2f} pts")
    models[pos] = (model, features)