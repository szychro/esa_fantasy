import pandas as pd

df = pd.read_csv("ekstraklasa_players.csv")

def fantasy_points_calc(df: pd.DataFrame) -> pd.Series:
    df = df[df.competition == 'Ekstraklasa']

    #fantasy points
    print(df['position'].unique())
    df['position_fantasy'] = df['position'].map({
        'GK': 'GK',
        'CB': 'DF',
        'LB': 'DF',
        'RB': 'DF',
        'DM': 'MF',
        'CM': 'MF',
        'AM': 'MF',
        'RM': 'MF',
        'LM': 'MF',
        'LW': 'FW',
        'RW': 'FW',
        'FW': 'FW'})

    fpts = 0

    goal_points = {'GK': 8, 'DF': 6, 'MF': 5, 'FW': 4}
    assist_points = {'GK': 6, 'DF': 4, 'MF': 3, 'FW': 3}
    clean_sheet_points = {'GK': 3, 'DF': 3, 'MF': 1, 'FW': 0}
    win_points = {'GK': 1, 'DF': 1, 'MF': 1, 'FW': 1}
    lost_goals_points = {'GK': -1, 'DF': -1, 'MF': 0, 'FW': 0}
    og_goals_points = {'GK': -3, 'DF': -3, 'MF': -3, 'FW': -3}
    red_card_points = {'GK': -3, 'DF': -3, 'MF': -3, 'FW': -3}
    yellow_card_points = {'GK': -1, 'DF': -1, 'MF': -1, 'FW': -1}


    fpts += df['goals'] * df['position_fantasy'].map(goal_points).fillna(0)
    fpts += df['assists'] * df['position_fantasy'].map(assist_points).fillna(0)
    fpts += df['yellow_cards'] * df['position_fantasy'].map(yellow_card_points).fillna(0)
    fpts += df['red_cards'] * df['position_fantasy'].map(red_card_points).fillna(0)

    if df['result'].str.split('–').str[1].astype(int)== 0:
        fpts += df['position_fantasy'].map(clean_sheet_points).fillna(0)

    if df['result'][0] == 'W':
        fpts += df['position_fantasy'].map(clean_sheet_points).fillna(0)

    if df['position_fantasy'] == 'GK':
        fpts += df['gk_saves'] // 3

    if df['result'].str.split('–').str[1].astype(int) >= 2:
        x = df['result'].str.split('–').str[1].astype(int)
        fpts += df['position_fantasy'].map(lost_goals_points).fillna(0) * (x-1)

    return fpts

df["fantasy_points"] = fantasy_points_calc(df)

df = df.sort_values(["player_id", "date"])

def rolling_features(df, windows=[3, 5, 10]):
    stat_cols = ["fantasy_points", "goals", "assists", "minutes",
                 "shots", "xg", "gk_saves", "gk_save_pct"]
    
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    for w in windows:
        for col in stat_cols:
            df[f"{col}_roll{w}"] = (
                df.groupby("player_id")[col]
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
    return df

df = rolling_features(df)

# Encode categoricals
df["is_home"] = df["venue"].map({"Home": 1, "Away": 0}).fillna(0.5)
df["position_enc"] = df["position"].map({"GK": 0, "DF": 1, "MF": 2, "FW": 3})