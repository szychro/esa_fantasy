from pathlib import Path
import sys

import pandas as pd
from scrap import SEASONS, close_driver, get_match_logs, get_players, get_teams, get_upcoming_fixtures


OUTPUT_CSV = Path("ekstraklasa_players.csv")
FIXTURES_CSV = Path("ekstraklasa_fixtures.csv")
KEY_COLUMNS = ["player_id", "match_id"]


def load_existing_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for column in KEY_COLUMNS + ["season"]:
        if column not in df.columns:
            df[column] = None
    return df


def merge_player_rows(existing_df: pd.DataFrame, new_rows: list[dict], refresh_seasons: list[str]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        if existing_df.empty:
            return new_df
        return existing_df.copy()

    if existing_df.empty:
        combined = new_df
    else:
        preserved_df = existing_df[~existing_df["season"].isin(refresh_seasons)].copy()
        existing_refresh_df = existing_df[existing_df["season"].isin(refresh_seasons)].copy()
        combined = pd.concat([preserved_df, existing_refresh_df, new_df], ignore_index=True, sort=False)

    combined = combined.sort_values(by=["season", "date", "player", "match_id"], na_position="last")
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    combined = combined.reset_index(drop=True)
    return combined


def parse_args() -> tuple[bool, list[str]]:
    full_refresh = "--full-refresh" in sys.argv
    if full_refresh or "--all-seasons" in sys.argv:
        seasons = SEASONS
    else:
        seasons = [SEASONS[0]]
    return full_refresh, seasons


def main():
    all_data = []
    upcoming_fixtures = []
    full_refresh, seasons_to_scrape = parse_args()
    refresh_scope = "full rebuild" if full_refresh else f"incremental refresh for {', '.join(seasons_to_scrape)}"

    print(f"Mode: {refresh_scope}")

    print("Fetching teams...")
    teams = get_teams("https://fbref.com/en/comps/36/Ekstraklasa-Stats")
    print(f"Found {len(teams)} teams\n")

    try:
        for team_name, team_url in teams.items():
            print(f"\n{'─'*50}")
            print(f"Team: {team_name}")
            fixtures = get_upcoming_fixtures(team_url, team_name)
            if fixtures:
                print(f"  {len(fixtures)} upcoming league fixtures found")
                upcoming_fixtures.extend(fixtures)
            players = get_players(team_url)
            print(f"  {len(players)} players found")

            for player_name, player in players.items():
                print(f"\n  Player: {player_name}")
                rows = get_match_logs(player, seasons=seasons_to_scrape)
                all_data.extend(rows)
    finally:
        close_driver()

    if full_refresh:
        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset=KEY_COLUMNS, keep="last").reset_index(drop=True)
    else:
        existing_df = load_existing_data(OUTPUT_CSV)
        df = merge_player_rows(existing_df, all_data, seasons_to_scrape)

    df.to_csv(OUTPUT_CSV, index=False)
    if upcoming_fixtures:
        fixtures_df = (
            pd.DataFrame(upcoming_fixtures)
            .drop_duplicates(subset=["team", "date", "opponent", "venue"], keep="last")
            .sort_values(by=["date", "team"])
            .reset_index(drop=True)
        )
        fixtures_df.to_csv(FIXTURES_CSV, index=False)
        print(f"Saved {len(fixtures_df)} upcoming fixtures to {FIXTURES_CSV}")
    print(f"\n✅ Done. {len(df)} rows saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
