import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scraper import Startlist, Stage, Race

# ── Wielermanager scoring ──────────────────────────────────────────────────────
POINTS_MAP = {
    1: 60, 2: 45, 3: 35, 4: 28, 5: 22,
    6: 18, 7: 15, 8: 12, 9: 10, 10: 8,
    11: 6, 12: 5, 13: 4, 14: 3, 15: 2,
}

# ── Feature Engineering ──────────────────────────────────────────────────────
FEATURE_COLS = [
    "stage_type", "distance_scaled", "vertical_meters_scaled", "temperature_scaled", "stage_scaled",
    "rider_avg_points_scaled", "rider_type_avg_scaled", "rider_recent_form_scaled",
    "spec_oneday_scaled", "spec_gc_scaled", "spec_tt_scaled", "spec_sprint_scaled", "spec_climber_scaled",
    "is_terrain_flat", "is_terrain_semi_hilly", "is_terrain_hilly", "is_terrain_mountain", "is_terrain_high_mountain",
    "match_climber", "match_sprint", "match_tt"
]

NUM_COLS = [
    "distance", "vertical_meters", "temperature", "stage",
    "rider_avg_points", "rider_type_avg", "rider_recent_form",
    "spec_oneday", "spec_gc", "spec_tt", "spec_sprint", "spec_climber",
    "match_climber", "match_sprint", "match_tt"
]

def normalize_name_for_lookup(name: str) -> str:
    """
    Ensure 'Philipsen Jasper' and 'Jasper Philipsen' match.
    Convert to sorted tuple of lowercased parts.
    """
    if not name: return ""
    return " ".join(sorted(name.lower().replace("-", " ").split()))

def position_to_points(position):
    if position is None: return 0.0
    try: return float(POINTS_MAP.get(int(position), 0))
    except: return 0.0

def infer_stage_type(stage_data: dict) -> int:
    st = str(stage_data.get("stage_type", "road")).lower()
    if st == "itt": return 3
    if st == "ttt": return 4
    raw = str(stage_data.get("parcours_type", "")).lower()
    if "mountain" in raw: return 2
    if "hilly" in raw or "hill" in raw: return 1
    return 0

def load_rider_specs(path="riders.json"):
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    specs = {}
    for r in data:
        # Key by normalized name
        key = normalize_name_for_lookup(r["name"])
        s = r.get("specialties", {})
        specs[key] = {
            "oneday": s.get("onedayraces", 0), "gc": s.get("gc", 0), "tt": s.get("tt", 0), "sprint": s.get("sprint", 0), "climber": s.get("climber", 0)
        }
    return specs

def load_data(path="grand_tours.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for race_entry in raw:
        race, year = race_entry["race"], int(race_entry["year"])
        race_riders = {}
        for stage in race_entry["stages"]:
            for res in stage.get("results", []):
                if res.get("rider_name"): race_riders[res["rider_name"]] = res.get("team", "unknown")
        for stage_data in race_entry["stages"]:
            stage_num = stage_data["stage"]
            dist = stage_data.get("distance_km") or float(str(stage_data.get("profile", {}).get("distance", "150")).split()[0])
            vert = stage_data.get("vertical_meters") or stage_data.get("profile", {}).get("vertical_meters", 0)
            terrain = stage_data.get("terrain", "unknown")
            temp = stage_data.get("weather", {}).get("temperature_c", 20.0) if isinstance(stage_data.get("weather"), dict) else 20.0
            st_type = infer_stage_type(stage_data)
            stage_results = {r["rider_name"]: r for r in stage_data.get("results", [])}
            for r_name, r_team in race_riders.items():
                res = stage_results.get(r_name, {})
                rows.append({
                    "race": race, "year": year, "stage": stage_num, "stage_type": st_type,
                    "distance": float(dist), "vertical_meters": float(vert or 0), "terrain": terrain, "temperature": float(temp),
                    "rider_name": r_name, "team": r_team, "points": position_to_points(res.get("position")),
                })
    return pd.DataFrame(rows)

def build_features(df: pd.DataFrame):
    df = df.sort_values(["year", "race", "stage"]).reset_index(drop=True)
    df["rider_avg_points"] = df.groupby("rider_name")["points"].transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    df["rider_type_avg"] = df.groupby(["rider_name", "stage_type"])["points"].transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    df["rider_recent_form"] = df.groupby("rider_name")["points"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(0))
    
    specs = load_rider_specs()
    for col in ["oneday", "gc", "tt", "sprint", "climber"]:
        df[f"spec_{col}"] = df["rider_name"].map(lambda x: specs.get(normalize_name_for_lookup(x), {}).get(col, 0)).fillna(0)
    
    df["match_climber"] = df["spec_climber"] * (df["terrain"].isin(["mountain", "high_mountain"])).astype(float)
    df["match_sprint"] = df["spec_sprint"] * (df["terrain"] == "flat").astype(float)
    df["match_tt"] = df["spec_tt"] * (df["stage_type"] == 3).astype(float)
    
    for t in ["flat", "semi_hilly", "hilly", "mountain", "high_mountain"]:
        df[f"is_terrain_{t}"] = (df["terrain"] == t).astype(float)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[NUM_COLS])
    for i, col in enumerate(NUM_COLS): df[f"{col}_scaled"] = scaled_data[:, i]
    
    le_team = LabelEncoder()
    df["team_id"] = le_team.fit_transform(df["team"])
    return df, le_team, scaler

class StageDataset(Dataset):
    def __init__(self, df):
        self.t, self.f, self.l = torch.tensor(df["team_id"].values, dtype=torch.long), torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32), torch.tensor(df["points"].values, dtype=torch.float32)
    def __len__(self): return len(self.l)
    def __getitem__(self, idx): return self.t[idx], self.f[idx], self.l[idx]

class WielermanagerModel(nn.Module):
    def __init__(self, n_teams, n_features):
        super().__init__()
        self.t_emb = nn.Embedding(n_teams + 1, 8)
        self.net = nn.Sequential(
            nn.Linear(8 + n_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, t, f):
        return torch.sigmoid(self.net(torch.cat([self.t_emb(t), f], dim=1)).squeeze(1)) * 60.0

def weighted_mse(preds, labels):
    weights = torch.where(labels > 0, torch.tensor(15.0), torch.tensor(1.0)).to(preds.device)
    return (weights * (preds - labels) ** 2).mean()

from difflib import get_close_matches
def find_rider_name(name: str, known_names: list) -> str | None:
    if name in known_names: return name
    parts = name.strip().split()
    if len(parts) >= 2:
        rev = f"{parts[-1]} {' '.join(parts[:-1])}"
        if rev in known_names: return rev
    m = get_close_matches(name, known_names, n=1, cutoff=0.7)
    return m[0] if m else None

def predict_stage(model, df, le_team, scaler, race_name, stage_num, info, riders, device):
    model.eval()
    specs = load_rider_specs()
    st_type = infer_stage_type(info)
    dist, vert, temp, terrain = float(info.get("distance_km", 150)), float(info.get("vertical_meters", 1000)), float(info.get("weather", {}).get("temperature_c", 20) if isinstance(info.get("weather"), dict) else 20), info.get("terrain", "flat")
    overall_map = df.groupby("rider_name")["rider_avg_points"].last().to_dict()
    type_map = df.groupby(["rider_name", "stage_type"])["rider_type_avg"].last().to_dict()
    form_map = df.groupby("rider_name")["rider_recent_form"].last().to_dict()
    team_map = df.groupby("rider_name")["team_id"].last().to_dict()
    known_riders = list(overall_map.keys())
    results = []
    for name in riders:
        matched = find_rider_name(name, known_riders)
        if not matched:
            r_avg, t_avg, r_rec, team_id = df["rider_avg_points"].median(), 0.0, 0.0, 0
            r_spec = {"oneday":0, "gc":0, "tt":0, "sprint":0, "climber":0}
        else:
            r_avg, t_avg, r_rec, team_id = overall_map.get(matched, 0.0), type_map.get((matched, st_type), 0.0), form_map.get(matched, 0.0), team_map.get(matched, 0)
            r_spec = specs.get(normalize_name_for_lookup(matched), {"oneday":0, "gc":0, "tt":0, "sprint":0, "climber":0})
        
        m_climber = r_spec["climber"] * (1.0 if terrain in ["mountain", "high_mountain"] else 0.0)
        m_sprint = r_spec["sprint"] * (1.0 if terrain == "flat" else 0.0)
        m_tt = r_spec["tt"] * (1.0 if st_type == 3 else 0.0)

        raw = pd.DataFrame([{
            "distance": dist, "vertical_meters": vert, "temperature": temp, "stage": float(stage_num),
            "rider_avg_points": r_avg, "rider_type_avg": t_avg, "rider_recent_form": r_rec,
            "spec_oneday": r_spec["oneday"], "spec_gc": r_spec["gc"], "spec_tt": r_spec["tt"], "spec_sprint": r_spec["sprint"], "spec_climber": r_spec["climber"],
            "match_climber": m_climber, "match_sprint": m_sprint, "match_tt": m_tt
        }])
        s_nums = scaler.transform(raw[NUM_COLS])[0]
        feat = torch.tensor([[float(st_type), *s_nums, *[float(terrain == t) for t in ["flat", "semi_hilly", "hilly", "mountain", "high_mountain"]]]], dtype=torch.float32).to(device)
        with torch.no_grad():
            p = model(torch.tensor([int(team_id)]).to(device), feat).item()
        results.append({"rider": matched or name, "points": round(p, 2)})
    return sorted(results, key=lambda x: x["points"], reverse=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", nargs=2, metavar=("RACE", "YEAR"))
    parser.add_argument("--data", default="grand_tours.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.train:
        print("Loading data..."); df = load_data(args.data); print("Adding Features..."); df, le_t, scaler = build_features(df)
        train_df, val_df = df[df["year"] < 2025], df[df["year"] >= 2025]
        model = WielermanagerModel(df["team_id"].nunique(), len(FEATURE_COLS)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_v, tl, vl = float("inf"), DataLoader(StageDataset(train_df), batch_size=512, shuffle=True), DataLoader(StageDataset(val_df), batch_size=512)
        print("Training...");
        for e in range(50):
            model.train()
            for t, f, l in tl: opt.zero_grad(); p = model(t.to(device), f.to(device)); loss = weighted_mse(p, l.to(device)); loss.backward(); opt.step()
            model.eval(); v_l = 0
            with torch.no_grad():
                for t, f, l in vl: v_l += weighted_mse(model(t.to(device), f.to(device)), l.to(device)).item()
            v_l /= len(vl)
            if v_l < best_v: best_v = v_l; torch.save(model.state_dict(), "best_model.pt"); pickle.dump((le_t, scaler, df), open("encoders.pkl", "wb"))
            if (e+1)%10==0: print(f"Epoch {e+1} Val Loss: {v_l:.4f}")
    if args.predict:
        slug, yr = args.predict
        le_t, scaler, df = pickle.load(open("encoders.pkl", "rb"))
        model = WielermanagerModel(df["team_id"].nunique(), len(FEATURE_COLS)).to(device)
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
        print(f"\n--- Predicting {slug} {yr} ---")
        sl, r = Startlist(slug, yr), Race(slug, yr)
        riders, s_ids = sl.get_riders(), r.get_stage_ids()
        all_p = []
        for sid in s_ids:
            print(f"S{sid}...", end=" ", flush=True)
            info = Stage(slug, int(yr), sid).get_profile()
            res = predict_stage(model, df, le_t, scaler, slug, sid, info, riders, device)
            all_p.append({"stage": sid, "terrain": info.get("terrain"), "top10": res[:10]})
            print(f"({info.get('terrain')}) Top: {res[0]['rider']} ({res[0]['points']}pts)")
        json.dump(all_p, open(f"predictions_{slug}_{yr}.json", "w"), indent=2)
