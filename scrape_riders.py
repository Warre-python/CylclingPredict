import json
import os
from scraper import Rider

def scrape_riders_from_json(json_path="grand_tours.json", output_path="riders.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all unique rider URLs
    rider_urls = set()
    for race in data:
        for stage in race["stages"]:
            for result in stage.get("results", []):
                url = result.get("rider_url")
                if url:
                    rider_urls.add(url)

    print(f"Found {len(rider_urls)} unique riders.")

    # Load existing riders to avoid re-scraping
    riders = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                old_list = json.load(f)
                riders = {r["profile_url"].split("com/")[-1]: r for r in old_list}
            except:
                pass

    new_count = 0
    # Scrape only a subset if too many, or all if feasible
    # For now, let's target riders from 2024-2025 primarily
    target_urls = set()
    for race in data:
        if int(race["year"]) >= 2024:
            for stage in race["stages"]:
                for result in stage.get("results", []):
                    url = result.get("rider_url")
                    if url: target_urls.add(url)

    print(f"Targeting {len(target_urls)} riders from 2024+.")

    results = list(riders.values())
    for i, url in enumerate(target_urls):
        if url in riders:
            continue
        
        print(f"[{i+1}/{len(target_urls)}] Scraping {url}...")
        try:
            r = Rider(url)
            info = r.to_dict()
            if info["name"]:
                results.append(info)
                riders[url] = info
                new_count += 1
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        
        # Save every 50 riders
        if new_count % 50 == 0 and new_count > 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Scraped {new_count} new riders. Total: {len(results)}")

if __name__ == "__main__":
    scrape_riders_from_json()
