import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime

base_url = "https://procyclingstats.com"

GRAND_TOURS = {
    "tour-de-france": range(2019, 2027),
    "giro-d-italia": range(2019, 2027),
    "vuelta-a-espana": range(2019, 2027),
}

CLASSICS = {
    "tour-de-flandres": range(2019, 2027),
    "paris-roubaix": range(2019, 2027),
    "liege-bastogne-liege": range(2019, 2027),
    "il-lombardia": range(2019, 2027),
    "milano-sanremo": range(2019, 2027),
}

import re
from difflib import get_close_matches

def normalize_name(name: str) -> str:
    """Convert any name format to 'Lastname Firstname' titlecase."""
    # Remove extra whitespace
    name = " ".join(name.strip().split())
    # If all uppercase e.g. 'THIJSSEN Gerben' or 'THIJSSEN GERBEN'
    if name == name.upper():
        name = name.title()
    # If 'LASTNAME Firstname' mixed case — uppercase part is lastname
    parts = name.split()
    normalized = []
    for p in parts:
        normalized.append(p.capitalize())
    return " ".join(normalized)


class Scraper:
    def __init__(self, url):
        self.url = url
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }
        self.soup = self._scrape()

    def _scrape(self):
        try:
            time.sleep(1)  # Be polite, avoid rate limiting
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {self.url}: {e}")
            return None

    def _get_info_field(self, label_text):
        if not self.soup:
            return None
        for li in self.soup.select("ul.list li"):
            label = li.select_one(".bold")
            if label and label_text in label.text:
                return li
        return None


class Rider(Scraper):
    def __init__(self, url):
        super().__init__(f"{base_url}/{url}")

    def _get_info_field(self, label_text):
        for li in self.soup.select("ul.list li"):
            label = li.select_one(".bold")
            if label and label_text in label.text:
                return li
        return None

    def get_name(self):
        li = self._get_info_field("Name")
        if li:
            tag = li.select_one(".mr10")
            return tag.text.strip() if tag else None
        return None

    def get_date_of_birth(self):
        li = self._get_info_field("Date of birth")
        if li:
            parts = li.select(".mr3")
            if len(parts) >= 3:
                return f"{parts[0].text.strip()} {parts[1].text.strip()} {parts[2].text.strip()}"
        return None

    def get_nationality(self):
        li = self._get_info_field("Nationality")
        if li:
            tag = li.select_one("a")
            return tag.text.strip() if tag else None
        return None

    def get_weight(self):
        li = self._get_info_field("Weight")
        if li:
            parts = li.select(".mr3")
            return parts[0].text.strip() if parts else None
        return None

    def get_height(self):
        li = self._get_info_field("Weight")  # same li as weight
        if li:
            parts = li.select(".mr3")
            return parts[1].text.strip() if len(parts) >= 2 else None
        return None

    def get_specialties(self):
        if not self.soup:
            return {}
        specialties = {}
        for item in self.soup.select("ul.pps li"):
            value = item.select_one(".xvalue")
            title = item.select_one(".xtitle")
            if value and title:
                specialties[title.text.strip()] = int(value.text.strip())
        return specialties

    def to_dict(self):
        return {
            "name": self.get_name(),
            "dob": self.get_date_of_birth(),
            "nationality": self.get_nationality(),
            "weight_kg": self.get_weight(),
            "height_m": self.get_height(),
            "specialties": self.get_specialties(),
        }


class Stage(Scraper):
    def __init__(self, race_slug, year, stage_num):
        self.race_slug = race_slug
        self.year = year
        self.stage_num = stage_num
        url = f"{base_url}/race/{race_slug}/{year}/stage-{stage_num}"
        super().__init__(url)

    def get_profile(self):
        if not self.soup:
            return {}

        profile = {}

        # Distance
        for li in self.soup.select("ul.infolist li"):
            label = li.select_one(".title")
            value = li.select_one(".value")
            if label and value:
                key = label.text.strip().lower().replace(" ", "_")
                profile[key] = value.text.strip()

        return profile

    def get_results(self, top_n=20):
        if not self.soup:
            return []

        results = []
        rows = self.soup.select("table.results tbody tr")

        for row in rows[:top_n]:
            cols = row.select("td")
            if len(cols) < 4:
                continue

            # Position
            pos_text = cols[0].text.strip()
            try:
                position = int(pos_text)
            except ValueError:
                position = None  # DNF, DNS, OTL etc

            # Rider name + url
            rider_tag = row.select_one("a[href*='rider']")
            rider_name = rider_tag.text.strip() if rider_tag else None
            rider_url = rider_tag["href"] if rider_tag else None

            # Team
            team_tag = row.select_one("a[href*='team']")
            team = team_tag.text.strip() if team_tag else None

            # Time / gap
            time_tag = cols[-1] if cols else None
            time_gap = time_tag.text.strip() if time_tag else None

            results.append({
                "position": position,
                "rider_name": rider_name,
                "rider_url": rider_url,
                "team": team,
                "time_gap": time_gap,
            })

        return results

    def to_dict(self):
        return {
            "race": self.race_slug,
            "year": self.year,
            "stage": self.stage_num,
            "profile": self.get_profile(),
            "results": self.get_results(),
        }


class Race(Scraper):
    def __init__(self, race_slug, year):
        self.race_slug = race_slug
        self.year = year
        url = f"{base_url}/race/{race_slug}/{year}/stages"
        super().__init__(url)

    def get_stage_count(self):
        if not self.soup:
            return 0
        # Count stage links
        stage_links = self.soup.select("a[href*='stage-']")
        nums = set()
        for a in stage_links:
            href = a["href"]
            part = href.split("stage-")[-1].split("/")[0]
            if part.isdigit():
                nums.add(int(part))
        return max(nums) if nums else 0

    def scrape_all_stages(self):
        stage_count = self.get_stage_count()
        print(f"  Found {stage_count} stages for {self.race_slug} {self.year}")
        stages = []
        for i in range(1, stage_count + 1):
            print(f"    Scraping stage {i}...")
            stage = Stage(self.race_slug, self.year, i)
            stages.append(stage.to_dict())
        return stages


def scrape_all(races: dict, output_file="data.json"):
    import json

    all_data = []

    for race_slug, years in races.items():
        for year in years:
            print(f"Scraping {race_slug} {year}...")
            race = Race(race_slug, year)
            stages = race.scrape_all_stages()
            all_data.append({
                "race": race_slug,
                "year": year,
                "stages": stages,
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved to {output_file}")
    return all_data

class Startlist(Scraper):
    def __init__(self, race_slug, year):
        self.race_slug = race_slug
        self.year = year
        url = f"{base_url}/race/{race_slug}/{year}/startlist"
        super().__init__(url)

    def get_riders(self):
        if not self.soup:
            return []

        riders = []
        for a in self.soup.select("a[href*='rider/']"):
            name = a.text.strip()
            if name:
                riders.append(normalize_name(name))

        return list(set(riders))
    
# In scraper.py, temporarily add to Stage.get_profile():
def get_profile(self):
    if not self.soup:
        return {}
    
    # Debug: print all infolist items
    for li in self.soup.select("ul.infolist li"):
        print(repr(li.text.strip()))
    
    # Also try alternative selectors
    print("--- trying .w30 ---")
    for div in self.soup.select(".w30"):
        print(repr(div.text.strip()[:100]))


if __name__ == "__main__":
    # Start with grand tours only, add classics later
    scrape_all(GRAND_TOURS, output_file="grand_tours.json")