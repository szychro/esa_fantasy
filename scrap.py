import re
import time
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Comment
import undetected_chromedriver as uc


BASE = "https://fbref.com"
SEASONS = ["2025-2026", "2024-2025", "2023-2024"]
LEAGUE_URL = f"{BASE}/en/comps/36/Ekstraklasa-Stats"

_driver = None


def create_driver(headless: bool = False):
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    return uc.Chrome(options=options, headless=headless)


def get_driver():
    global _driver
    if _driver is None:
        _driver = create_driver(headless=False)
    return _driver


def fetch(url: str, delay: int = 4) -> BeautifulSoup:
    driver = get_driver()
    driver.get(url)
    time.sleep(delay)
    return BeautifulSoup(driver.page_source, "html.parser")


def close_driver():
    global _driver
    if _driver is not None:
        _driver.quit()
        _driver = None


def expanded_soup(soup: BeautifulSoup) -> BeautifulSoup:
    # FBref frequently wraps full tables in HTML comments.
    parts = [str(soup)]
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        text = comment.strip()
        if "<table" in text or "<div" in text:
            parts.append(text)
    return BeautifulSoup("\n".join(parts), "html.parser")


def absolute_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return BASE + href


def parse_player_href(href: str) -> Optional[Dict[str, str]]:
    match = re.search(r"/players/([a-z0-9]+)/([^/?#]+)", href)
    if not match:
        return None
    return {
        "player_id": match.group(1),
        "player_slug": match.group(2),
        "player_url": absolute_url(href),
    }


def extract_match_id(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    match = re.search(r"/matches/([a-z0-9]+)/", url)
    return match.group(1) if match else None


def get_teams(league_url: str = LEAGUE_URL) -> Dict[str, str]:
    soup = expanded_soup(fetch(league_url))
    teams: Dict[str, str] = {}

    for row in soup.select("tr"):
        team_cell = row.find("td", attrs={"data-stat": "team"})
        if not team_cell:
            continue

        link = team_cell.find("a", href=re.compile(r"/squads/"))
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "").strip()
        if name and href:
            teams[name] = absolute_url(href)

    return teams


def get_players(team_url: str) -> Dict[str, Dict[str, str]]:
    soup = expanded_soup(fetch(team_url))
    players: Dict[str, Dict[str, str]] = {}

    for row in soup.select("tr"):
        player_cell = row.find(["th", "td"], attrs={"data-stat": "player"})
        if not player_cell:
            continue

        link = player_cell.find("a", href=re.compile(r"/players/"))
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "").strip()
        parsed = parse_player_href(href)
        if not name or not parsed:
            continue

        players[name] = {
            "name": name,
            **parsed,
        }

    return players


def get_upcoming_fixtures(team_url: str, team_name: str) -> List[Dict[str, str]]:
    soup = expanded_soup(fetch(team_url))
    fixtures: List[Dict[str, str]] = []

    for table in soup.find_all("table"):
        table_id = table.get("id", "")
        if "matchlogs" not in table_id and "schedule" not in table_id:
            continue

        for tr in table.select("tbody tr[data-row]"):
            classes = tr.get("class", [])
            if "thead" in classes or "partial_table" in classes:
                continue

            def text_cell(stat: str) -> Optional[str]:
                cell = tr.find(attrs={"data-stat": stat})
                if cell is None:
                    return None
                link = cell.find("a")
                text = link.get_text(strip=True) if link else cell.get_text(strip=True)
                return text or None

            date = text_cell("date")
            competition = text_cell("comp")
            opponent = text_cell("opponent")
            venue = text_cell("venue")
            result = text_cell("result")

            if not date or competition != "Ekstraklasa" or not opponent or result:
                continue

            fixtures.append(
                {
                    "date": date,
                    "competition": competition,
                    "round": text_cell("round"),
                    "day": text_cell("dayofweek"),
                    "venue": venue,
                    "team": team_name,
                    "opponent": opponent,
                }
            )

        if fixtures:
            break

    return fixtures


def build_matchlog_url(player_id: str, player_slug: str, season: str) -> str:
    return f"{BASE}/en/players/{player_id}/matchlogs/{season}/{player_slug}-Match-Logs"


def get_match_logs(player: Dict[str, str], seasons: Optional[List[str]] = None) -> List[Dict[str, str]]:
    seasons = seasons or SEASONS
    all_rows: List[Dict[str, str]] = []

    for season in seasons:
        url = build_matchlog_url(player["player_id"], player["player_slug"], season)
        soup = expanded_soup(fetch(url))

        table = soup.find("table", id=re.compile(r"matchlogs"))
        if not table:
            print(f"  x No match log table for {player['name']} in {season}")
            continue

        rows = parse_match_table(table, player, season)
        print(f"  OK {player['name']} {season}: {len(rows)} matches")
        all_rows.extend(rows)

    return all_rows


def parse_match_table(table, player: Dict[str, str], season: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for tr in table.select("tbody tr[data-row]"):
        classes = tr.get("class", [])
        if "thead" in classes or "partial_table" in classes:
            continue

        def text_cell(stat: str) -> Optional[str]:
            cell = tr.find(attrs={"data-stat": stat})
            if cell is None:
                return None
            link = cell.find("a")
            text = link.get_text(strip=True) if link else cell.get_text(strip=True)
            return text or None

        def href_cell(stat: str) -> Optional[str]:
            cell = tr.find(attrs={"data-stat": stat})
            if cell is None:
                return None
            link = cell.find("a")
            if not link or not link.get("href"):
                return None
            return absolute_url(link["href"])

        match_url = href_cell("match_report") or href_cell("date")

        row = {
            "player": player["name"],
            "player_id": player["player_id"],
            "player_slug": player["player_slug"],
            "player_url": player["player_url"],
            "season": season,
            "date": text_cell("date"),
            "day": text_cell("dayofweek"),
            "competition": text_cell("comp"),
            "round": text_cell("round"),
            "venue": text_cell("venue"),
            "result": text_cell("result"),
            "team": text_cell("team"),
            "opponent": text_cell("opponent"),
            "started": text_cell("game_started"),
            "position": text_cell("position"),
            "minutes": text_cell("minutes"),
            "goals": text_cell("goals"),
            "assists": text_cell("assists"),
            "shots": text_cell("shots"),
            "xg": text_cell("xg"),
            'yellow_cards': text_cell("cards_yellow"),
            'red_cards': text_cell("cards_red"),
            "gk_shots_against": text_cell("gk_shots_on_target_against"),
            "gk_goals_against": text_cell("gk_goals_against"),
            "gk_saves": text_cell("gk_saves"),
            "gk_save_pct": text_cell("gk_save_pct"),
            "gk_clean_sheet": text_cell("gk_clean_sheets"),
            "match_url": match_url,
            "match_id": extract_match_id(match_url),
        }

        if row["date"]:
            rows.append(row)

    return rows
