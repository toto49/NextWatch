"""Construit et exporte l'artefact User–CF (joblib) pour l'inférence produit.

Usage :
  python scripts/Models/export_usercf.py

Sortie :
  artifacts/usercf.joblib
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation_protocol import (  # noqa: E402
    dataset_fingerprint,
    liked_interactions,
    load_dataframe,
    protocol_metadata,
    runtime_versions,
    split_dataframe,
)

ARTIFACTS_DIR = ROOT / "artifacts"
OUT_PATH = ARTIFACTS_DIR / "usercf.joblib"

DEFAULTS = {
    "MIN_SUPPORT": 34,
    "CF_N_USER": 180,
    "DEMO_SEX_WEIGHT": 0.05,
    "AGE_SCALE": 30.0,
    "RATING_POWER": 0.75,
}


def _load_params() -> dict:
    path = ROOT / "results" / "best_hyperparams_cf_optuna.json"
    params = dict(DEFAULTS)
    source_meta = {}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for k in DEFAULTS:
            if k in raw:
                params[k] = raw[k]
        source_meta = {
            key: raw[key]
            for key in ("validation_score", "metric", "model", "n_trials")
            if key in raw
        }
        if "protocol" in raw:
            source_meta["protocol"] = raw["protocol"]
        print(f"Hyperparams depuis {path.name}")
    else:
        print("Pas de JSON Optuna CF — valeurs par défaut.")
    return {"params": params, "source_meta": source_meta}


def build_artifact() -> dict:
    loaded = _load_params()
    params = loaded["params"]
    min_support = int(params["MIN_SUPPORT"])
    cf_n_user = int(params["CF_N_USER"])
    demo_sex_weight = float(params["DEMO_SEX_WEIGHT"])
    age_scale = float(params["AGE_SCALE"])
    rating_power = float(params["RATING_POWER"])

    df = load_dataframe()
    split = split_dataframe(df)
    train_validation = split.train_validation

    film_stats = (
        train_validation.groupby("item_id")["rating"]
        .agg(n_ratings="count", mean_rating="mean")
        .reset_index()
    )
    eligible_ids = {
        int(movie_id)
        for movie_id in film_stats.loc[
            film_stats["n_ratings"] >= min_support, "item_id"
        ]
    }
    liked_train = liked_interactions(train_validation, train_validation)
    liked_train = liked_train.loc[liked_train["item_id"].isin(eligible_ids)]

    liked_by_user = (
        liked_train.groupby("user_id")[["item_id", "rating"]]
        .apply(
            lambda rows: list(
                zip(
                    rows["item_id"].astype(int).tolist(),
                    rows["rating"].astype(float).tolist(),
                )
            )
        )
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
        "eligible_ids": eligible_ids,
        "params": {
            "MIN_SUPPORT": min_support,
            "CF_N_USER": cf_n_user,
            "DEMO_SEX_WEIGHT": demo_sex_weight,
            "AGE_SCALE": age_scale,
            "RATING_POWER": rating_power,
        },
        "meta": {
            "protocol": protocol_metadata(),
            "dataset_sha256": dataset_fingerprint(),
            "runtime_versions": runtime_versions(),
            "training_split": "train+validation",
            "test_used": False,
            "tuning": loaded["source_meta"],
            "n_fans_films": len(fans_of),
            "n_likers": len(liked_by_user),
            "n_titles": len(movie_title),
            "n_eligible": len(eligible_ids),
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
