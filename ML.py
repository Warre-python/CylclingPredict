import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from scraper import Startlist, Stage, Race

# ── Wielermanager scoring ──────────────────────────────────────────────────────
POINTS_MAP = {
    1: 60, 2: 45, 3: 35, 4: 28, 5: 22,
    6: 18, 7: 15, 8: 12, 9: 10, 10: 8,
    11: 6, 12: 5, 13: 4, 14: 3, 15: 2,
}

FEATURE_COLS = [
    "stage_type",
    "distance_scaled",
    "stage",
    "rider_avg_points",
    "rider_type_avg",
    "rider_type_top3_rate",
    "rider_recent_form",
    "is_stage_type_0",
    "is_stage_type_1", 
    "is_stage_type_2",
    "is_stage_type_3",
    "is_stage_type_4",
]

class StageDataset(Dataset):
    def __init__(self, df):
        self.rider_ids = torch.tensor(df["rider_id"].values, dtype=torch.long)
        self.team_ids = torch.tensor(df["team_id"].values, dtype=torch.long)
        self.race_ids = torch.tensor(df["race_id"].values, dtype=torch.long)
        self.features = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        self.labels = torch.tensor(df["points"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.rider_ids[idx],
            self.team_ids[idx],
            self.race_ids[idx],
            self.features[idx],
            self.labels[idx],
        )

def position_to_points(position):
    if position is None:
        return 0.0
    return float(POINTS_MAP.get(position, 0))

# ── Stage type inference from profile ─────────────────────────────────────────
STAGE_TYPE_MAP = {
    "flat": 0, "hilly": 1, "mountain": 2, "itt": 3, "ttt": 4,
}

def infer_stage_type(profile: dict) -> int:
    raw = str(profile).lower()
    if "time trial" in raw or "itt" in raw or "ttt" in raw:
        return 3 if "individual" in raw else 4
    if "mountain" in raw:
        return 2
    if "hilly" in raw or "hill" in raw:
        return 1
    return 0  # default flat

# ── Flatten JSON into rows ─────────────────────────────────────────────────────
def load_data(path="grand_tours.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for race_entry in raw:
        race = race_entry["race"]
        year = int(race_entry["year"])

        for stage_data in race_entry["stages"]:
            stage_num = stage_data["stage"]
            profile = stage_data.get("profile", {})
            stage_type = infer_stage_type(profile)

            # Parse distance
            distance_str = profile.get("distance", "0").replace("km", "").strip()
            try:
                distance = float(distance_str.split()[0])
            except:
                distance = 0.0

            for result in stage_data.get("results", []):
                position = result.get("position")
                rider_name = result.get("rider_name")
                team = result.get("team")

                if not rider_name:
                    continue

                rows.append({
                    "race": race,
                    "year": year,
                    "stage": stage_num,
                    "stage_type": stage_type,
                    "distance": distance,
                    "rider_name": rider_name,
                    "team": team or "unknown",
                    "position": position,
                    "points": position_to_points(position),
                })

    return pd.DataFrame(rows)

# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    le_rider = LabelEncoder()
    le_team = LabelEncoder()
    le_race = LabelEncoder()

    df["rider_id"] = le_rider.fit_transform(df["rider_name"])
    df["team_id"] = le_team.fit_transform(df["team"])
    df["race_id"] = le_race.fit_transform(df["race"])

    # Rider historical avg points (computed before current row = no leakage)
    df = df.sort_values(["year", "stage"]).reset_index(drop=True)
    df["rider_avg_points"] = (
        df.groupby("rider_id")["points"]
        .transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    )

    # Rider avg points by stage type
    df["rider_type_avg"] = (
        df.groupby(["rider_id", "stage_type"])["points"]
        .transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    )

    scaler = StandardScaler()
    df["distance_scaled"] = scaler.fit_transform(df[["distance"]])

    return df, le_rider, le_team, le_race, scaler

# ── Dataset ────────────────────────────────────────────────────────────────────
class StageDataset(Dataset):
    def __init__(self, df):
        self.rider_ids = torch.tensor(df["rider_id"].values, dtype=torch.long)
        self.team_ids = torch.tensor(df["team_id"].values, dtype=torch.long)
        self.race_ids = torch.tensor(df["race_id"].values, dtype=torch.long)
        self.features = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
        self.labels = torch.tensor(df["points"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.rider_ids[idx],
            self.team_ids[idx],
            self.race_ids[idx],
            self.features[idx],
            self.labels[idx],
        )

# ── Model ──────────────────────────────────────────────────────────────────────
class WielermanagerModel(nn.Module):
    def __init__(self, n_riders, n_teams, n_races, n_features, embed_dim=16):
        super().__init__()

        # Embeddings capture latent rider/team/race skill
        self.rider_emb = nn.Embedding(n_riders, embed_dim)
        self.team_emb = nn.Embedding(n_teams, embed_dim // 2)
        self.race_emb = nn.Embedding(n_races, embed_dim // 2)

        input_dim = embed_dim + embed_dim // 2 + embed_dim // 2 + n_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),  # points are always >= 0
        )

    def forward(self, rider_ids, team_ids, race_ids, features):
        r = self.rider_emb(rider_ids)
        t = self.team_emb(team_ids)
        rc = self.race_emb(race_ids)
        x = torch.cat([r, t, rc, features], dim=1)
        return self.net(x).squeeze(1)

# ── Training ───────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for rider_ids, team_ids, race_ids, features, labels in loader:
        rider_ids = rider_ids.to(device)
        team_ids = team_ids.to(device)
        race_ids = race_ids.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(rider_ids, team_ids, race_ids, features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for rider_ids, team_ids, race_ids, features, labels in loader:
            rider_ids = rider_ids.to(device)
            team_ids = team_ids.to(device)
            race_ids = race_ids.to(device)
            features = features.to(device)
            labels = labels.to(device)
            preds = model(rider_ids, team_ids, race_ids, features)
            total_loss += criterion(preds, labels).item()
    return total_loss / len(loader)

from difflib import get_close_matches

def find_rider_name(name: str, known_names: list) -> str | None:
    # Try direct match first
    if name in known_names:
        return name
    # Try reversing Firstname Lastname -> Lastname Firstname
    parts = name.strip().split()
    if len(parts) == 2:
        reversed_name = f"{parts[1]} {parts[0]}"
        if reversed_name in known_names:
            return reversed_name
        matches = get_close_matches(reversed_name, known_names, n=1, cutoff=0.75)
        if matches:
            return matches[0]
    # Fuzzy fallback
    matches = get_close_matches(name, known_names, n=1, cutoff=0.75)
    return matches[0] if matches else None

# ── Predict for an upcoming stage ─────────────────────────────────────────────
def predict_stage(
    model, df, le_rider, le_team, le_race, scaler,
    race_name, stage_num, stage_type_str, distance_km,
    rider_names, device
):
    model.eval()
    known_names = list(le_rider.classes_)
    stage_type = STAGE_TYPE_MAP.get(stage_type_str.lower(), 0)
    distance_scaled = scaler.transform(pd.DataFrame({"distance": [distance_km]}))["distance"].values[0]

    try:
        race_id = le_race.transform([race_name])[0]
    except ValueError:
        race_id = 0

    results = []
    for name in rider_names:
        matched = find_rider_name(name, known_names)
        if not matched:
            continue

        rider_id = le_rider.transform([matched])[0]
        rider_rows = df[df["rider_id"] == rider_id]
        type_rows = rider_rows[rider_rows["stage_type"] == stage_type]

        rider_avg = rider_rows["points"].mean() if len(rider_rows) else 0.0
        rider_type_avg = type_rows["points"].mean() if len(type_rows) else rider_avg
        rider_type_top3 = type_rows["is_top3"].mean() if len(type_rows) else 0.0
        rider_recent = rider_rows["points"].tail(5).mean() if len(rider_rows) else 0.0

        for val in [rider_avg, rider_type_avg, rider_type_top3, rider_recent]:
            if np.isnan(val):
                val = 0.0

        one_hot = [float(stage_type == st) for st in range(5)]

        feature_vec = [
            stage_type,
            distance_scaled,
            stage_num,
            rider_avg,
            rider_type_avg,
            rider_type_top3,
            rider_recent,
            *one_hot,
        ]

        team_name = rider_rows["team"].iloc[-1] if len(rider_rows) else "unknown"
        try:
            team_id = le_team.transform([team_name])[0]
        except ValueError:
            team_id = 0

        features = torch.tensor([feature_vec], dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_points = model(
                torch.tensor([rider_id]).to(device),
                torch.tensor([team_id]).to(device),
                torch.tensor([race_id]).to(device),
                features,
            ).item()

        results.append({"rider": matched, "predicted_points": round(pred_points, 2)})

    return sorted(results, key=lambda x: x["predicted_points"], reverse=True)

def predict_race(
    model, df, le_rider, le_team, le_race, scaler, device,
    race_slug, year, known_names
):
    print(f"\nFetching startlist for {race_slug} {year}...")
    startlist = Startlist(race_slug, year)
    rider_names = startlist.get_riders()
    print(f"Found {len(rider_names)} riders on startlist")

    print(f"Fetching stages for {race_slug} {year}...")
    race = Race(race_slug, year)
    stage_count = race.get_stage_count()
    print(f"Found {stage_count} stages\n")

    all_predictions = []

    for stage_num in range(1, stage_count + 1):
        print(f"  Predicting stage {stage_num}...")
        stage = Stage(race_slug, year, stage_num)
        profile = stage.get_profile()
        stage_type = infer_stage_type(profile)
        stage_type_str = {v: k for k, v in STAGE_TYPE_MAP.items()}.get(stage_type, "flat")

        distance_str = profile.get("distance", "0").replace("km", "").strip()
        try:
            distance_km = float(distance_str.split()[0])
        except:
            distance_km = 150.0

        predictions = predict_stage(
            model=model,
            df=df,
            le_rider=le_rider,
            le_team=le_team,
            le_race=le_race,
            scaler=scaler,
            race_name=race_slug,
            stage_num=stage_num,
            stage_type_str=stage_type_str,
            distance_km=distance_km,
            rider_names=rider_names,
            device=device,
        )

        all_predictions.append({
            "stage": stage_num,
            "stage_type": stage_type_str,
            "distance_km": distance_km,
            "profile": profile,
            "top10": predictions[:10],
        })

        print(f"    Type: {stage_type_str} | Distance: {distance_km}km")
        print(f"    Top 3: " + ", ".join(
            f"{p['rider']} ({p['predicted_points']}pts)"
            for p in predictions[:3]
        ))

    # Temporarily add this in predict_race to debug
    startlist = Startlist(race_slug, year)
    rider_names = startlist.get_riders()
    print("Sample startlist names:", rider_names[:10])

    stage = Stage(race_slug, year, 1)
    profile = stage.get_profile()
    print("Stage 1 profile:", profile)

    return all_predictions


def save_predictions(predictions, race_slug, year, path=None):
    import json
    path = path or f"predictions_{race_slug}_{year}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\nSaved predictions to {path}")

def build_features(df: pd.DataFrame):
    le_rider = LabelEncoder()
    le_team = LabelEncoder()
    le_race = LabelEncoder()

    df["rider_id"] = le_rider.fit_transform(df["rider_name"])
    df["team_id"] = le_team.fit_transform(df["team"])
    df["race_id"] = le_race.fit_transform(df["race"])

    df = df.sort_values(["year", "stage"]).reset_index(drop=True)

    # Overall avg points (lagged)
    df["rider_avg_points"] = (
        df.groupby("rider_id")["points"]
        .transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    )

    # Avg points per stage type (lagged) — KEY feature
    df["rider_type_avg"] = (
        df.groupby(["rider_id", "stage_type"])["points"]
        .transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    )

    # Win rate per stage type
    df["is_top3"] = (df["position"] <= 3).astype(float)
    df["rider_type_top3_rate"] = (
        df.groupby(["rider_id", "stage_type"])["is_top3"]
        .transform(lambda x: x.shift(1).expanding().mean().fillna(0))
    )

    # Recent form: avg points in last 5 results
    df["rider_recent_form"] = (
        df.groupby("rider_id")["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(0))
    )

    # Stage type one-hot (so model can't ignore it)
    for st in range(5):
        df[f"is_stage_type_{st}"] = (df["stage_type"] == st).astype(float)

    scaler = StandardScaler()
    df["distance_scaled"] = scaler.fit_transform(df[["distance"]])

    return df, le_rider, le_team, le_race, scaler

def weighted_mse(preds, labels):
    weights = torch.where(labels > 0, torch.tensor(10.0), torch.tensor(1.0)).to(preds.device)
    return (weights * (preds - labels) ** 2).mean()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    

    # Load + process
    print("Loading data...")
    df = load_data("grand_tours.json")
    df, le_rider, le_team, le_race, scaler = build_features(df)
    print(f"Dataset: {len(df)} rows, {df['rider_name'].nunique()} unique riders")

    

    # Split by year so val = most recent season (no leakage)
    train_df = df[df["year"] < 2025]
    val_df = df[df["year"] >= 2025]

    train_ds = StageDataset(train_df)
    val_ds = StageDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # Build model
    model = WielermanagerModel(
        n_riders=df["rider_id"].nunique(),
        n_teams=df["team_id"].nunique(),
        n_races=df["race_id"].nunique(),
        n_features=len(FEATURE_COLS),  # was hardcoded 5, now 12
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = weighted_mse
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # Train
    print("\nTraining...")
    best_val_loss = float("inf")
    for epoch in range(50):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # Save encoders for reuse
    with open("encoders.pkl", "wb") as f:
        pickle.dump((le_rider, le_team, le_race, scaler, df), f)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("Model saved to best_model.pt")

    # ── Example prediction ─────────────────────────────────────────────────────
    # Load trained model + encoders
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    known_names = list(le_rider.classes_)

    # Predict full race automatically
    predictions = predict_race(
        model=model,
        df=df,
        le_rider=le_rider,
        le_team=le_team,
        le_race=le_race,
        scaler=scaler,
        device=device,
        race_slug="giro-d-italia",
        year=2025,
        known_names=known_names,
    )

    save_predictions(predictions, "giro-d-italia", 2026)