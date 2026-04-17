def predict_top5_per_position(df, models, next_gw_date):
    """
    For each player, take their most recent row (latest rolling stats)
    and predict their expected points for the upcoming gameweek.
    """
    results = {}
    
    # Get latest feature snapshot per player
    latest = df.sort_values("date").groupby("player_id").last().reset_index()
    
    for pos, (model, features) in models.items():
        pos_players = latest[latest["position"] == pos].copy()
        pos_players = pos_players.dropna(subset=features)
        
        pos_players["predicted_pts"] = model.predict(pos_players[features])
        
        top5 = (pos_players
                .sort_values("predicted_pts", ascending=False)
                .head(5)[["player", "team", "predicted_pts"] + features[:3]])
        
        results[pos] = top5
        print(f"\n🏆 Top 5 {pos}:")
        print(top5.to_string(index=False))
    
    return results

top5 = predict_top5_per_position(df, models, next_gw_date="2024-04-15")