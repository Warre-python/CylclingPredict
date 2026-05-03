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

FEATURE_COLS = [
    "stage_type",
    "distance_scaled", "vertical_meters_scaled", "temperature_scaled", "stage_scaled",
    "rider_avg_points_scaled", "rider_type_avg_scaled", "rider_type_top3_rate_scaled", "rider_recent_form_scaled",
    "spec_oneday_scaled", "spec_gc_scaled", "spec_tt_scaled", "spec_sprint_scaled", "spec_climber_scaled",
    "is_terrain_flat", "is_terrain_semi_hilly", "is_terrain_hilly", "is_terrain_mountain", "is_terrain_high_mountain",
]

NUM_COLS = [
    "distance", "vertical_meters", "temperature", "stage",
    "rider_avg_points", "rider_type_avg", "rider_type_top3_rate", "rider_recent_form",
    "spec_oneday", "spec_gc", "spec_tt", "spec_sprint", "spec_climber"
]

def position_to_points(position):
    if position is None: return 0.0
    try: return float(POINTS_MAP.get(int(position), 0))
    except (ValueError, TypeError): return 0.0

def infer_stage_type(stage_data: dict) -> int:
    st = stage_data.get("stage_type", "road").lower()
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
        name = r["name"]
        s = r.get("specialties", {})
        specs[name] = {
            "oneday": s.get("onedayraces", 0),
            "gc": s.get("gc", 0),
            "tt": s.get("tt", 0),
            "sprint": s.get("sprint", 0),
            "climber": s.get("climber", 0)
        }
    return specs

def load_data(path="grand_tours.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for race_entry in raw:
        race, year = race_entry["race"], int(race_entry["year"])
        for stage_data in race_entry["stages"]:
            stage_num = stage_data["stage"]
            dist = stage_data.get("distance_km") or float(str(stage_data.get("profile", {}).get("distance", "150")).split()[0])
            vert = stage_data.get("vertical_meters") or stage_data.get("profile", {}).get("vertical_meters", 0)
            terrain = stage_data.get("terrain", "unknown")
            weather = stage_data.get("weather", {})
            temp = weather.get("temperature_c", 20.0) if isinstance(weather, dict) else 20.0
            st_type = infer_stage_type(stage_data)
            for result in stage_data.get("results", []):
                rider_name, team, pos = result.get("rider_name"), result.get("team"), result.get("position")
                if not rider_name: continue
                rows.append({
                    "race": race, "year": year, "stage": stage_num, "stage_type": st_type,
                    "distance": float(dist), "vertical_meters": float(vert or 0), "terrain": terrain, "temperature": float(temp),
                    "rider_name": rider_name, "team": team or "unknown", "position": pos, "points": position_to_points(pos),
                })
    return pd.DataFrame(rows)

def build_features(df: pd.DataFrame):
    le_rider, le_team, le_race = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df["rider_id"] = le_rider.fit_transform(df["rider_name"])
    df["team_id"] = le_team.fit_transform(df["team"])
    df["race_id"] = le_race.fit_transform(df["race"])
    
    specs = load_rider_specs()
    for col in ["oneday", "gc", "tt", "sprint", "climber"]:
        df[f"spec_{col}"] = df["rider_name"].map(lambda x: specs.get(x, {}).get(col, 0)).fillna(0)

    df = df.sort_values(["year", "stage"]).reset_index(drop=True)
    df["rider_avg_points"] = df.groupby("rider_id")["points"].transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    df["rider_type_avg"] = df.groupby(["rider_id", "stage_type"])["points"].transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    df["is_top3"] = df["position"].apply(lambda x: 1 if str(x).isdigit() and int(x) <= 3 else 0).astype(float)
    df["rider_type_top3_rate"] = df.groupby(["rider_id", "stage_type"])["is_top3"].transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    df["rider_recent_form"] = df.groupby("rider_id")["points"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(0))
    for t in ["flat", "semi_hilly", "hilly", "mountain", "high_mountain"]:
        df[f"is_terrain_{t}"] = (df["terrain"] == t).astype(float)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[NUM_COLS])
    for i, col in enumerate(NUM_COLS): df[f"{col}_scaled"] = scaled_data[:, i]
    return df, le_rider, le_team, le_race, scaler

class StageDataset(Dataset):
    def __init__(self, df):
        self.r = torch.tensor(df["rider_id"].values, dtype=torch.long)
        self.t = torch.tensor(df["team_id"].values, dtype=torch.long)
        self.rc = torch.tensor(df["race_id"].values, dtype=torch.long)
        self.f = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        self.l = torch.tensor(df["points"].values, dtype=torch.float32)
    def __len__(self): return len(self.l)
    def __getitem__(self, idx): return self.r[idx], self.t[idx], self.rc[idx], self.f[idx], self.l[idx]

class WielermanagerModel(nn.Module):
    def __init__(self, n_riders, n_teams, n_races, n_features):
        super().__init__()
        self.r_emb = nn.Embedding(n_riders + 1, 8)
        self.t_emb = nn.Embedding(n_teams + 1, 4)
        self.rc_emb = nn.Embedding(n_races + 1, 4)
        self.net = nn.Sequential(
            nn.Linear(8+4+4+n_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, r, t, rc, f):
        x = torch.cat([self.r_emb(r), self.t_emb(t), self.rc_emb(rc), f], dim=1)
        return torch.sigmoid(self.net(x).squeeze(1)) * 60.0

def weighted_mse(preds, labels):
    weights = torch.where(labels > 0, torch.tensor(5.0), torch.tensor(1.0)).to(preds.device)
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

def predict_stage(model, df, le_rider, le_team, le_race, scaler, race_name, stage_num, info, riders, device):
    model.eval()
    known = list(le_rider.classes_)
    specs = load_rider_specs()
    st_type = infer_stage_type(info)
    dist, vert, temp, terrain = float(info.get("distance_km", 150)), float(info.get("vertical_meters", 1000)), float(info.get("weather", {}).get("temperature_c", 20) if isinstance(info.get("weather"), dict) else 20), info.get("terrain", "flat")
    try: race_id = le_race.transform([race_name])[0]
    except: race_id = 0
    results = []
    for name in riders:
        matched = find_rider_name(name, known)
        if not matched: continue
        r_id = le_rider.transform([matched])[0]
        r_rows = df[df["rider_id"] == r_id]
        t_rows = r_rows[r_rows["stage_type"] == st_type]
        r_avg = r_rows["points"].mean() if len(r_rows) else 0.0
        t_avg = t_rows["points"].mean() if len(t_rows) else r_avg
        t_top3 = (t_rows["position"].apply(lambda x: 1 if str(x).isdigit() and int(x) <= 3 else 0)).mean() if len(t_rows) else 0.0
        r_rec = r_rows["points"].tail(5).mean() if len(r_rows) else 0.0
        r_spec = specs.get(matched, {"oneday":0, "gc":0, "tt":0, "sprint":0, "climber":0})
        raw = pd.DataFrame([{
            "distance": dist, "vertical_meters": vert, "temperature": temp, "stage": float(stage_num),
            "rider_avg_points": r_avg, "rider_type_avg": t_avg, "rider_type_top3_rate": t_top3, "rider_recent_form": r_rec,
            "spec_oneday": r_spec["oneday"], "spec_gc": r_spec["gc"], "spec_tt": r_spec["tt"], "spec_sprint": r_spec["sprint"], "spec_climber": r_spec["climber"]
        }])
        s_nums = scaler.transform(raw[NUM_COLS])[0]
        t_hot = [float(terrain == t) for t in ["flat", "semi_hilly", "hilly", "mountain", "high_mountain"]]
        feat = torch.tensor([[float(st_type), *s_nums, *t_hot]], dtype=torch.float32).to(device)
        team = r_rows["team"].iloc[-1] if len(r_rows) else "unknown"
        try: t_id = le_team.transform([team])[0]
        except: t_id = 0
        with torch.no_grad():
            p = model(torch.tensor([r_id]).to(device), torch.tensor([t_id]).to(device), torch.tensor([race_id]).to(device), feat).item()
        results.append({"rider": matched, "points": round(p, 2)})
    return sorted(results, key=lambda x: x["points"], reverse=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", nargs=2, metavar=("RACE", "YEAR"))
    parser.add_argument("--data", default="grand_tours.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.train:
        df = load_data(args.data)
        df, le_r, le_t, le_rc, scaler = build_features(df)
        train_df, val_df = df[df["year"] < 2025], df[df["year"] >= 2025]
        model = WielermanagerModel(df["rider_id"].nunique(), df["team_id"].nunique(), df["race_id"].nunique(), len(FEATURE_COLS)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_v = float("inf")
        tl, vl = DataLoader(StageDataset(train_df), batch_size=256, shuffle=True), DataLoader(StageDataset(val_df), batch_size=256)
        for e in range(50):
            model.train()
            for r, t, rc, f, l in tl:
                opt.zero_grad(); p = model(r.to(device), t.to(device), rc.to(device), f.to(device)); loss = weighted_mse(p, l.to(device)); loss.backward(); opt.step()
            model.eval(); v_l = 0
            with torch.no_grad():
                for r, t, rc, f, l in vl: v_l += weighted_mse(model(r.to(device), t.to(device), rc.to(device), f.to(device)), l.to(device)).item()
            v_l /= len(vl)
            if v_l < best_v: best_v = v_l; torch.save(model.state_dict(), "best_model.pt"); pickle.dump((le_r, le_t, le_rc, scaler, df), open("encoders.pkl", "wb"))
            if (e+1)%10==0: print(f"Epoch {e+1} Val Loss: {v_l:.4f}")
    if args.predict:
        slug, yr = args.predict
        le_r, le_t, le_rc, scaler, df = pickle.load(open("encoders.pkl", "rb"))
        model = WielermanagerModel(df["rider_id"].nunique(), df["team_id"].nunique(), df["race_id"].nunique(), len(FEATURE_COLS)).to(device)
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
        print(f"\n--- Predicting {slug} {yr} ---")
        sl, r = Startlist(slug, yr), Race(slug, yr)
        riders, s_ids = sl.get_riders(), r.get_stage_ids()
        all_p = []
        for sid in s_ids:
            print(f"S{sid}...", end=" ", flush=True)
            res = predict_stage(model, df, le_r, le_t, le_rc, scaler, slug, sid, Stage(slug, int(yr), sid).get_profile(), riders, device)
            all_p.append({"stage": sid, "top10": res[:10]})
            print(f"Top: {res[0]['rider']} ({res[0]['points']}pts)")
        json.dump(all_p, open(f"predictions_{slug}_{yr}.json", "w"), indent=2)
