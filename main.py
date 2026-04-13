from scrap import close_driver, get_players, get_match_logs, get_teams
import pandas as pd

def main():
    all_data = []

    print("Fetching teams...")
    teams = get_teams("https://fbref.com/en/comps/36/Ekstraklasa-Stats")
    print(f"Found {len(teams)} teams\n")

    try:
        for team_name, team_url in teams.items():
            print(f"\n{'─'*50}")
            print(f"Team: {team_name}")
            players = get_players(team_url)
            print(f"  {len(players)} players found")

            for player_name, player in players.items():
                print(f"\n  Player: {player_name}")
                rows = get_match_logs(player)
                all_data.extend(rows)
    finally:
        close_driver()

    df = pd.DataFrame(all_data)
    df.to_csv("ekstraklasa_players.csv", index=False)
    print(f"\n✅ Done. {len(df)} rows saved to ekstraklasa_players.csv")

if __name__ == "__main__":
    main()
