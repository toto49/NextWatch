"""Construit et exporte l'artefact User–CF (joblib) pour l'inférence produit.

Usage :
  python scripts/export_usercf.py

Sortie :
  artifacts/usercf.joblib
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_PATH = ARTIFACTS_DIR / "usercf.joblib"

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

RANDOM_STATE = 42
DEFAULTS = {
    "TRAIN_FRAC": 0.65,
    "MIN_SUPPORT": 34,
    "CF_N_USER": 180,
    "DEMO_SEX_WEIGHT": 0.05,
    "AGE_SCALE": 30.0,
    "RATING_POWER": 0.75,
}


def _load_params() -> dict:
    path = ROOT / "results" / "best_hyperparams_cf_optuna.json"
    params = dict(DEFAULTS)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for k in DEFAULTS:
            if k in raw:
                params[k] = raw[k]
        print(f"Hyperparams depuis {path.name}")
    else:
        print("Pas de JSON Optuna CF — valeurs par défaut.")
    return params


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


def build_artifact() -> dict:
    params = _load_params()
    train_frac = float(params["TRAIN_FRAC"])
    min_support = int(params["MIN_SUPPORT"])
    cf_n_user = int(params["CF_N_USER"])
    demo_sex_weight = float(params["DEMO_SEX_WEIGHT"])
    age_scale = float(params["AGE_SCALE"])
    rating_power = float(params["RATING_POWER"])

    df = _load_df()
    train = df.sample(frac=train_frac, random_state=RANDOM_STATE).reset_index(drop=True)

    film_stats = (
        train.groupby("item_id")["rating"]
        .agg(n_ratings="count", mean_rating="mean")
        .reset_index()
    )
    eligible = film_stats.loc[film_stats["n_ratings"] >= min_support].set_index(
        "item_id"
    )

    user_mean = train.groupby("user_id")["rating"].mean()
    train = train.copy()
    train["user_mean"] = train["user_id"].map(user_mean)
    liked_train = train.loc[train["rating"] > train["user_mean"]]

    liked_by_user = (
        liked_train.groupby("user_id")[["item_id", "rating"]]
        .apply(lambda x: list(zip(x["item_id"].tolist(), x["rating"].tolist())))
        .to_dict()
    )
    fans_of = {
        int(mid): list(
            zip(
                grp["user_id"].astype(int).tolist(),
                grp["rating"].astype(float).tolist(),
                grp["age"].astype(float).tolist(),
                grp["gender"].tolist(),
            )
        )
        for mid, grp in liked_train.groupby("item_id")
    }

    movies = df.drop_duplicates("item_id").set_index("item_id")
    movie_title = {int(i): str(t) for i, t in movies["title"].items()}
    title_to_id = {t.lower(): i for i, t in movie_title.items()}

    artifact = {
        "fans_of": fans_of,
        "liked_by_user": liked_by_user,
        "movie_title": movie_title,
        "title_to_id": title_to_id,
        "eligible_ids": set(int(i) for i in eligible.index),
        "params": {
            "CF_N_USER": cf_n_user,
            "DEMO_SEX_WEIGHT": demo_sex_weight,
            "AGE_SCALE": age_scale,
            "RATING_POWER": rating_power,
        },
        "meta": {
            "random_state": RANDOM_STATE,
            "train_frac": train_frac,
            "min_support": min_support,
            "n_fans_films": len(fans_of),
            "n_likers": len(liked_by_user),
            "n_titles": len(movie_title),
        },
    }
    return artifact


def main() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    joblib.dump(artifact, OUT_PATH, compress=3)
    meta = artifact["meta"]
    print(
        f"Exporté → {OUT_PATH}\n"
        f"  films (fans_of)={meta['n_fans_films']} | "
        f"users likers={meta['n_likers']} | titres={meta['n_titles']}"
    )
    return OUT_PATH


if __name__ == "__main__":
    main()
