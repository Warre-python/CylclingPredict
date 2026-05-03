"""
cycling_scraper.py — ProCyclingStats scraper for ML datasets
============================================================
Run --debug-stage FIRST to inspect the real HTML of a page and
confirm selectors before doing a full scrape.

Usage:
  python cycling_scraper.py --debug-stage tour-de-france 2020 1
  python cycling_scraper.py --debug-rider rider/tadej-pogacar
  python cycling_scraper.py --mode stages --races grand_tours
  python cycling_scraper.py --mode riders --races grand_tours
  python cycling_scraper.py --mode all   --races both
"""

import argparse
import json
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ─── Race catalogues ──────────────────────────────────────────────────────────
base_url = "https://procyclingstats.com"

GRAND_TOURS = {
    "tour-de-france": range(2019, 2027),
    "giro-d-italia":  range(2019, 2027),
    "vuelta-a-espana": range(2019, 2027),
}

CLASSICS = {
    "tour-de-flandres":     range(2019, 2027),
    "paris-roubaix":        range(2019, 2027),
    "liege-bastogne-liege": range(2019, 2027),
    "il-lombardia":         range(2019, 2027),
    "milano-sanremo":       range(2019, 2027),
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    name = " ".join(name.strip().split())
    if name == name.upper():
        name = name.title()
    return " ".join(p.capitalize() for p in name.split())


def safe_int(text) -> int | None:
    if not text:
        return None
    m = re.search(r"\d+", re.sub(r"[,.](?=\d{3})", "", str(text)))
    return int(m.group()) if m else None


def safe_float(text) -> float | None:
    if not text:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(text))
    return float(m.group().replace(",", ".")) if m else None


def parse_time_gap(raw: str | None) -> int | None:
    if not raw:
        return None
    raw = raw.strip().lstrip("+")
    if raw in ("", "-", "s.t.", ",,", "0:00"):
        return 0
    m = re.match(r"(?:(\d+):)?(\d+):(\d+)", raw)
    if m:
        h, mn, s = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
        return h * 3600 + mn * 60 + s
    try:
        return int(raw)
    except ValueError:
        return None


def classify_terrain(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score < 15:  return "flat"
    if score < 35:  return "semi_hilly"
    if score < 55:  return "hilly"
    if score < 75:  return "mountain"
    return "high_mountain"


def find_number_with_unit(text: str, unit: str) -> float | None:
    m = re.search(rf"([\d.,]+)\s*{re.escape(unit)}", text, re.IGNORECASE)
    return float(m.group(1).replace(",", ".")) if m else None


# ─── HTTP session ─────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://procyclingstats.com/",
})


def fetch(url: str, delay: float = 1.2) -> BeautifulSoup | None:
    time.sleep(delay)
    try:
        r = SESSION.get(url, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [ERROR] {url}: {e}")
        return None


# ─── Debug helper ─────────────────────────────────────────────────────────────
def debug_page(soup: BeautifulSoup, url: str):
    """
    Print everything about a page: CSS classes, <ul> structures,
    and visible text. Run this once per page type to identify selectors.
    """
    print(f"\n{'='*60}\nDEBUG: {url}\n{'='*60}")

    classes = sorted({c for tag in soup.find_all(True) for c in tag.get("class", [])})
    print(f"\n[CSS classes] {len(classes)} total:")
    print("  " + ", ".join(classes))

    print("\n[<ul> tags with classes]")
    for ul in soup.find_all("ul"):
        cls = ul.get("class", [])
        if cls:
            preview = [li.get_text(" ", strip=True)[:70] for li in ul.select("li")][:5]
            print(f"  ul.{'.'.join(cls)}")
            for p in preview:
                print(f"    · {p}")

    print("\n[Tables]")
    for tbl in soup.select("table"):
        cls = tbl.get("class", [])
        rows = tbl.select("tr")
        print(f"  table.{'.'.join(cls) if cls else '(no class)'} — {len(rows)} rows")
        for row in rows[:3]:
            print(f"    {row.get_text(' | ', strip=True)[:100]}")

    print("\n[Page text — first 4000 chars]")
    print(soup.get_text(" ", strip=True)[:4000])


# ─── Infolist parser (multi-strategy) ────────────────────────────────────────
def parse_infolist(soup: BeautifulSoup) -> dict:
    """
    Extract key→value pairs from the info block using several strategies
    so the scraper survives PCS HTML changes.
    """
    data = {}

    # Strategy A: <ul class="infolist|list"> with title/value child divs
    for ul in soup.select("ul.infolist, ul.list"):
        for li in ul.select("li"):
            # Clone the li to safely extract the label
            li_copy = BeautifulSoup(str(li), "html.parser").select_one("li")
            label_tag = li_copy.select_one(".title, .bold, b, strong, dt")
            if not label_tag:
                continue
            key = label_tag.get_text(strip=True).rstrip(":").lower().replace(" ", "_")
            label_tag.decompose()
            val = li_copy.get_text(" ", strip=True).strip(": ")
            if key and val:
                data[key] = val

    # Strategy B: <dl> definition lists
    for dl in soup.select("dl"):
        for dt, dd in zip(dl.select("dt"), dl.select("dd")):
            key = dt.get_text(strip=True).rstrip(":").lower().replace(" ", "_")
            data[key] = dd.get_text(strip=True)

    # Strategy C: table rows where first cell is a label
    for tbl in soup.select("table.basic, table.info, .raceinfo table"):
        for row in tbl.select("tr"):
            cells = row.select("td, th")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True).rstrip(":").lower().replace(" ", "_")
                val = cells[1].get_text(strip=True)
                if key and val:
                    data.setdefault(key, val)

    # Strategy D: divs/spans in .right / .info containers
    for container in soup.select(".right, .infos, .info-container, .race-info, .w50"):
        for li in container.select("li"):
            spans = li.select("span, div, b, strong")
            if len(spans) >= 2:
                key = spans[0].get_text(strip=True).rstrip(":").lower().replace(" ", "_")
                val = spans[-1].get_text(strip=True)
                if key and val:
                    data.setdefault(key, val)

    return data


# ─── Stage ────────────────────────────────────────────────────────────────────
class Stage:
    def __init__(self, race_slug: str, year: int, stage_num):
        self.race_slug = race_slug
        self.year = year
        self.stage_num = stage_num
        self.url = f"{base_url}/race/{race_slug}/{year}/stage-{stage_num}"
        self.soup = fetch(self.url)

    def get_profile(self) -> dict:
        if not self.soup:
            return {}

        info = parse_infolist(self.soup)
        page_text = self.soup.get_text(" ")

        # ── Distance ──────────────────────────────────────────────────────────
        distance_km = None
        for k in ("distance", "length", "afstand", "km"):
            if k in info:
                distance_km = safe_float(info[k])
                break
        if not distance_km:
            distance_km = find_number_with_unit(page_text, "km")

        # ── Vertical meters ───────────────────────────────────────────────────
        vertical_meters = None
        for k in ("vertical_meters", "vert._meters", "vert.meters",
                   "climbing", "elevation", "hoogtemeters", "hm"):
            if k in info:
                vertical_meters = safe_int(info[k])
                break
        if not vertical_meters:
            m = re.search(r"([\d,]+)\s*m\s*(climbing|elevation|vert|hm\b)", page_text, re.I)
            if m:
                vertical_meters = safe_int(m.group(1))

        # ── Profile score ─────────────────────────────────────────────────────
        profile_score = None
        for k in ("profilescore", "profile_score", "parcours_score", "score"):
            if k in info:
                profile_score = safe_int(info[k])
                break

        # ── Departure / arrival ───────────────────────────────────────────────
        departure = info.get("departure", info.get("start", info.get("from", "")))
        arrival   = info.get("arrival",   info.get("finish", info.get("to", "")))

        # ── Stage type ────────────────────────────────────────────────────────
        parcours_type = info.get(
            "parcours_type", info.get("type", info.get("stage_type", ""))
        )
        pt = parcours_type.lower()
        if "team time" in pt or "ttt" in pt:
            stage_type = "ttt"
        elif "time trial" in pt or " itt" in pt or pt.startswith("tt"):
            stage_type = "itt"
        else:
            stage_type = "road"

        start_time = info.get("starttime", info.get("start_time", info.get("depart", "")))

        # ── Weather ───────────────────────────────────────────────────────────
        weather = {}
        for k in ("temperature", "weather", "weer", "meteo"):
            if k in info:
                raw = info[k]
                m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?[Cc]", raw)
                if m:
                    weather["temperature_c"] = float(m.group(1))
                weather["description"] = raw
                break

        return {
            "distance_km": distance_km,
            "vertical_meters": vertical_meters,
            "profile_score": profile_score,
            "terrain": classify_terrain(profile_score),
            "departure": departure,
            "arrival": arrival,
            "parcours_type": parcours_type,
            "stage_type": stage_type,
            "start_time": start_time,
            "weather": weather,
        }

    def get_results(self, top_n: int = 20) -> list:
        if not self.soup:
            return []

        table = (
            self.soup.select_one("table.results") or
            self.soup.select_one("table.basic")    or
            self.soup.select_one("table")
        )
        if not table:
            return []

        results = []
        for row in table.select("tbody tr")[:top_n]:
            cols = row.select("td")
            if len(cols) < 3:
                continue

            pos_text = cols[0].get_text(strip=True)
            try:
                position = int(pos_text)
            except ValueError:
                position = pos_text   # DNF / DNS / OTL

            rider_tag = row.select_one("a[href*='rider']")
            rider_name = normalize_name(rider_tag.get_text(strip=True)) if rider_tag else None
            rider_url  = rider_tag["href"] if rider_tag else None

            team_tag = row.select_one("a[href*='team']")
            team = team_tag.get_text(strip=True) if team_tag else None

            # Time gap: scan from right, skip already-captured fields
            time_gap_raw = None
            skip = {str(position), rider_name or "", team or ""}
            for col in reversed(cols):
                txt = col.get_text(strip=True)
                if txt and txt not in skip:
                    time_gap_raw = txt
                    break

            results.append({
                "position": position,
                "rider_name": rider_name,
                "rider_url": rider_url,
                "team": team,
                "time_gap_raw": time_gap_raw,
                "time_gap_seconds": parse_time_gap(time_gap_raw),
            })

        return results

    def to_dict(self) -> dict:
        return {
            "race": self.race_slug,
            "year": self.year,
            "stage": self.stage_num,
            **self.get_profile(),
            "results": self.get_results(),
        }

    def debug(self):
        if self.soup:
            debug_page(self.soup, self.url)


# ─── Race ─────────────────────────────────────────────────────────────────────
class Race:
    def __init__(self, race_slug: str, year: int):
        self.race_slug = race_slug
        self.year = year
        self.soup = fetch(f"{base_url}/race/{race_slug}/{year}/stages")

    def get_stage_ids(self) -> list:
        if not self.soup:
            return []
        ids = set()
        for a in self.soup.select("a[href*='stage-']"):
            part = a["href"].split("stage-")[-1].split("/")[0]
            if part.isdigit():
                ids.add(int(part))
            elif re.match(r"^[a-z]+$", part):
                ids.add(part)
        return sorted(ids, key=lambda x: (isinstance(x, str), x))

    def scrape_all_stages(self) -> list:
        ids = self.get_stage_ids()
        print(f"  {self.race_slug} {self.year}: {len(ids)} stages")
        stages = []
        for sid in ids:
            print(f"    Stage {sid} ... ", end="", flush=True)
            d = Stage(self.race_slug, self.year, sid).to_dict()
            km = d.get("distance_km", "?")
            terrain = d.get("terrain", "?")
            vert = d.get("vertical_meters")
            print(f"{km} km | {terrain}" + (f" | {vert}m↑" if vert else ""))
            stages.append(d)
        return stages


# ─── Startlist ────────────────────────────────────────────────────────────────
class Startlist:
    def __init__(self, race_slug: str, year: int):
        self.soup = fetch(f"{base_url}/race/{race_slug}/{year}/startlist")

    def get_rider_urls(self) -> list[dict]:
        if not self.soup:
            return []
        seen, riders = set(), []
        for a in self.soup.select("a[href*='rider/']"):
            href = a["href"]
            name = normalize_name(a.get_text(strip=True))
            if name and href not in seen:
                seen.add(href)
                riders.append({"name": name, "url": href})
        return riders


# ─── Rider ────────────────────────────────────────────────────────────────────
class Rider:
    def __init__(self, url: str):
        self.url = url if url.startswith("http") else f"{base_url}/{url}"
        self.soup = fetch(self.url)

    def get_name(self) -> str | None:
        if not self.soup:
            return None
        h1 = self.soup.select_one("h1")
        return normalize_name(h1.get_text(strip=True)) if h1 else None

    def get_dob(self) -> str | None:
        if not self.soup:
            return None
        text = self.soup.get_text(" ")
        m = re.search(r"\b(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})\b", text)
        if m:
            return m.group(1)
        m2 = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        return m2.group(1) if m2 else None

    def get_age(self) -> int | None:
        dob_str = self.get_dob()
        if not dob_str:
            return None
        for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
            try:
                dob = datetime.strptime(dob_str, fmt)
                today = datetime.today()
                return today.year - dob.year - (
                    (today.month, today.day) < (dob.month, dob.day)
                )
            except ValueError:
                pass
        return None

    def get_nationality(self) -> str | None:
        if not self.soup:
            return None
        info = parse_infolist(self.soup)
        for k in ("nationality", "land", "country"):
            if k in info:
                return info[k]
        flag = self.soup.select_one("[class*='flag']")
        return (flag.get("title") or flag.get_text(strip=True)) if flag else None

    def get_weight_height(self) -> tuple[float | None, float | None]:
        if not self.soup:
            return None, None
        text = self.soup.get_text(" ")
        weight = find_number_with_unit(text, "kg")
        height = find_number_with_unit(text, "m")
        if height and height > 10:   # listed in cm
            height = round(height / 100, 2)
        return weight, height

    def get_current_team(self) -> str | None:
        if not self.soup:
            return None
        for a in self.soup.select("a[href*='team/']"):
            name = a.get_text(strip=True)
            if name:
                return name
        return None

    def get_specialties(self) -> dict:
        if not self.soup:
            return {}
        specialties = {}
        # PCS uses ul.pps or similar
        for item in self.soup.select("ul.pps li, .pps li"):
            value = item.select_one(".xvalue, .value, b, strong")
            title = item.select_one(".xtitle, .title, span")
            if value and title and value != title:
                key = (title.get_text(strip=True)
                       .lower().replace(" ", "_").replace("-", "_"))
                specialties[key] = safe_int(value.get_text(strip=True))
        return specialties

    def get_rankings(self) -> dict:
        if not self.soup:
            return {}
        rankings = {}
        text = self.soup.get_text(" ")
        uci = re.search(r"UCI\s+(?:World\s+)?[Rr]anking[:\s]+(\d+)", text)
        pcs = re.search(r"PCS\s+[Rr]anking[:\s]+(\d+)", text)
        if uci:
            rankings["uci_ranking"] = int(uci.group(1))
        if pcs:
            rankings["pcs_ranking"] = int(pcs.group(1))
        return rankings

    def to_dict(self) -> dict:
        weight, height = self.get_weight_height()
        return {
            "name": self.get_name(),
            "dob": self.get_dob(),
            "age": self.get_age(),
            "nationality": self.get_nationality(),
            "weight_kg": weight,
            "height_m": height,
            "current_team": self.get_current_team(),
            **self.get_rankings(),
            "specialties": self.get_specialties(),
            "profile_url": self.url,
        }

    def debug(self):
        if self.soup:
            debug_page(self.soup, self.url)


# ─── Orchestrators ────────────────────────────────────────────────────────────
def scrape_races(races: dict, output_file: str = "stages.json") -> list:
    all_data = []
    for race_slug, years in races.items():
        for year in years:
            print(f"\n{'─'*55}\n{race_slug} {year}")
            race = Race(race_slug, year)
            stages = race.scrape_all_stages()
            all_data.append({"race": race_slug, "year": year, "stages": stages})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Stage data → {output_file}")
    return all_data


def scrape_riders(races: dict, output_file: str = "riders.json") -> list:
    rider_index: dict[str, str] = {}
    for race_slug, years in races.items():
        for year in years:
            print(f"  Startlist {race_slug} {year} ...", end=" ", flush=True)
            sl = Startlist(race_slug, year)
            entries = sl.get_rider_urls()
            for e in entries:
                rider_index.setdefault(e["url"], e["name"])
            print(f"{len(entries)} riders")

    print(f"\nUnique riders: {len(rider_index)}")
    all_riders = []
    for i, (url, _) in enumerate(rider_index.items(), 1):
        print(f"  [{i}/{len(rider_index)}] {url} ... ", end="", flush=True)
        d = Rider(url).to_dict()
        print(d.get("name", "?"))
        all_riders.append(d)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_riders, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Rider data → {output_file}")
    return all_riders


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCS scraper — run --debug-stage first to verify HTML selectors"
    )
    parser.add_argument("--mode", choices=["stages", "riders", "all"], default="all")
    parser.add_argument("--races", choices=["grand_tours", "classics", "both"],
                        default="grand_tours")
    parser.add_argument("--stages-out", default="stages.json")
    parser.add_argument("--riders-out", default="riders.json")
    parser.add_argument(
        "--debug-stage", nargs=3, metavar=("RACE", "YEAR", "STAGE"),
        help="Dump HTML analysis + parsed data for one stage"
    )
    parser.add_argument(
        "--debug-rider", metavar="URL",
        help="Dump HTML analysis + parsed data for one rider"
    )
    args = parser.parse_args()

    if args.debug_stage:
        race, year, stage = args.debug_stage
        s = Stage(race, int(year), stage)
        s.debug()
        print("\n=== parse_infolist() ===")
        if s.soup:
            print(json.dumps(parse_infolist(s.soup), indent=2, ensure_ascii=False))
        print("\n=== get_profile() ===")
        print(json.dumps(s.get_profile(), indent=2, ensure_ascii=False))
        print("\n=== get_results()[:3] ===")
        print(json.dumps(s.get_results(3), indent=2, ensure_ascii=False))

    elif args.debug_rider:
        r = Rider(args.debug_rider)
        r.debug()
        print("\n=== to_dict() ===")
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))

    else:
        race_set = {
            "grand_tours": GRAND_TOURS,
            "classics":    CLASSICS,
            "both":        {**GRAND_TOURS, **CLASSICS},
        }[args.races]

        if args.mode in ("stages", "all"):
            scrape_races(race_set, args.stages_out)
        if args.mode in ("riders", "all"):
            scrape_riders(race_set, args.riders_out)