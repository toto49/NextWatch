#!/usr/bin/env python3
"""Exporte l'artefact cluster (groupes de goûts) pour l'inférence produit.

Paramètres = ceux de ``notbook/MODEL_CLUSTER.ipynb`` (pas Optuna).

Usage :
  python scripts/export_cluster.py

Sortie :
  artifacts/cluster.joblib
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_PATH = ARTIFACTS_DIR / "cluster.joblib"

# === Aligné sur notbook/MODEL_CLUSTER.ipynb ===
RANDOM_STATE = 42
LIKED_THRESHOLD = 4
TRAIN_FRAC = 0.65
MIN_SUPPORT = 22
N_CLUSTERS = 8
MIN_GROUP_CANDIDATES = 5
CF_N_USER = 200  # témoin notebook (conservé dans l'artefact, non utilisé en reco produit)
# Poids démo (alignés sur le UserCF du notebook : 0.2 sexe + 0.8 âge/95)
DEMO_SEX_WEIGHT = 0.2
AGE_SCALE = 95.0
# Amplifie âge/sexe à l'assignation de cluster (sinon les genres écrasent la démo)
DEMO_ASSIGN_WEIGHT = 4.0

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def _load_df() -> pd.DataFrame:
    notes = pd.read_csv(DATA_DIR / "u.data.csv")
    films = pd.read_csv(DATA_DIR / "u.item.csv")
    users = pd.read_csv(DATA_DIR / "u.user.csv", dtype={"zip_code": "string"}).drop(
        columns=["occupation"], errors="ignore"
    )
    df = notes.merge(users, on="user_id", how="left").merge(
        films, left_on="item_id", right_on="movie_id", how="left"
    )
    df = df.loc[df["unknown"] != 1].drop(columns=["unknown"]).copy()
    df["genres"] = df[GENRES].apply(
        lambda row: [g for g in GENRES if row[g] == 1], axis=1
    )
    return df


def _build_user_genre_profiles(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    tm = ratings[["user_id", "item_id", "rating", "age", "gender"]].merge(
        movies[GENRES], left_on="item_id", right_index=True, how="left"
    )
    cols = {
        g: tm.loc[tm[g] == 1].groupby("user_id")["rating"].mean() for g in GENRES
    }
    prof = pd.DataFrame(cols)
    demo = ratings.drop_duplicates("user_id").set_index("user_id")[["age", "gender"]]
    prof["age"] = demo["age"]
    prof["gender_m"] = (demo["gender"] == "M").astype(float)
    return prof


def build_artifact() -> dict:
    df = _load_df()
    train = df.sample(frac=TRAIN_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)

    movies = df.drop_duplicates("item_id").set_index("item_id")
    film_genre_X = movies[GENRES].astype(float)
    movie_title = {int(i): str(t) for i, t in movies["title"].items()}
    title_to_id = {t.lower(): i for i, t in movie_title.items()}

    film_stats = (
        train.groupby("item_id")["rating"]
        .agg(n_ratings="count", mean_rating="mean")
        .reset_index()
    )
    eligible = film_stats.loc[film_stats["n_ratings"] >= MIN_SUPPORT].set_index(
        "item_id"
    )
    eligible_ids = set(int(i) for i in eligible.index)
    # Volume brut (comme pop_count du notebook) pour le repli popularité
    pop_count = {int(mid): int(n) for mid, n in eligible["n_ratings"].items()}

    liked_train = train.loc[train["rating"] >= LIKED_THRESHOLD].copy()

    profiles = _build_user_genre_profiles(train, movies)
    feat_cols = GENRES + ["age", "gender_m"]
    X_raw = profiles[feat_cols].fillna(0.0)
    scaler = StandardScaler().fit(X_raw.values)
    X_scaled = scaler.transform(X_raw.values)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    user_cluster = {
        int(uid): int(lab)
        for uid, lab in zip(X_raw.index.astype(int), labels)
    }

    liked = liked_train.copy()
    liked["cluster"] = liked["user_id"].map(user_cluster)
    cluster_film_scores: dict[int, dict[int, float]] = {}
    for cid, grp in liked.dropna(subset=["cluster"]).groupby("cluster"):
        s = grp.groupby("item_id")["rating"].sum()
        cluster_film_scores[int(cid)] = {
            int(mid): float(sc)
            for mid, sc in s.items()
            if int(mid) in eligible_ids
        }

    # Profil démo des fans (pour re-ranker âge/sexe à l'inférence)
    film_liker_demo: dict[int, tuple[float, float, int]] = {}
    for mid, grp in liked_train.groupby("item_id"):
        mid = int(mid)
        if mid not in eligible_ids:
            continue
        film_liker_demo[mid] = (
            float(grp["age"].mean()),
            float((grp["gender"] == "M").mean()),
            int(len(grp)),
        )

    artifact = {
        "model": "user_cluster",
        "genres": list(GENRES),
        "scaler": scaler,
        "kmeans_centers": np.asarray(kmeans.cluster_centers_, dtype=float),
        "cluster_film_scores": cluster_film_scores,
        "film_liker_demo": film_liker_demo,
        "pop_count": pop_count,
        "film_genre_matrix": {
            int(i): film_genre_X.loc[i].to_numpy(dtype=float).tolist()
            for i in film_genre_X.index
        },
        "movie_title": movie_title,
        "title_to_id": title_to_id,
        "eligible_ids": eligible_ids,
        "params": {
            "RANDOM_STATE": RANDOM_STATE,
            "LIKED_THRESHOLD": LIKED_THRESHOLD,
            "TRAIN_FRAC": TRAIN_FRAC,
            "MIN_SUPPORT": MIN_SUPPORT,
            "N_CLUSTERS": N_CLUSTERS,
            "MIN_GROUP_CANDIDATES": MIN_GROUP_CANDIDATES,
            "CF_N_USER": CF_N_USER,
            "SEED_PROXY": float(LIKED_THRESHOLD),
            "DEMO_SEX_WEIGHT": DEMO_SEX_WEIGHT,
            "AGE_SCALE": AGE_SCALE,
            "DEMO_ASSIGN_WEIGHT": DEMO_ASSIGN_WEIGHT,
        },
        "meta": {
            "source": "notbook/MODEL_CLUSTER.ipynb",
            "n_clusters": N_CLUSTERS,
            "n_users_profiles": int(len(profiles)),
            "n_eligible": len(eligible_ids),
            "n_titles": len(movie_title),
            "cluster_sizes": {
                int(c): int(n) for c, n in pd.Series(labels).value_counts().items()
            },
        },
    }
    return artifact


def main() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    joblib.dump(artifact, OUT_PATH, compress=3)
    meta = artifact["meta"]
    p = artifact["params"]
    print(
        f"Exporté → {OUT_PATH}\n"
        f"  model=user_cluster (MODEL_CLUSTER.ipynb) | "
        f"K={p['N_CLUSTERS']} | MIN_SUPPORT={p['MIN_SUPPORT']} | "
        f"aimé≥{p['LIKED_THRESHOLD']} | TRAIN_FRAC={p['TRAIN_FRAC']}\n"
        f"  eligible={meta['n_eligible']} | titres={meta['n_titles']} | "
        f"tailles={meta['cluster_sizes']}"
    )
    return OUT_PATH


if __name__ == "__main__":
    main()
