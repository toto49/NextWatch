#!/usr/bin/env python3
"""Reconstruit et exporte le modèle Cluster pur sélectionné sur validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation_protocol import (  # noqa: E402
    GENRES,
    RANDOM_STATE,
    dataset_fingerprint,
    liked_interactions,
    load_dataframe,
    protocol_metadata,
    runtime_versions,
    split_dataframe,
)

ARTIFACTS_DIR = ROOT / "artifacts"
OUT_PATH = ARTIFACTS_DIR / "cluster.joblib"
PARAMS_PATH = ROOT / "results" / "best_hyperparams_cluster_optuna.json"
TUNED_PARAMETERS = (
    "MIN_SUPPORT",
    "N_CLUSTERS",
    "MIN_GROUP_CANDIDATES",
    "SEED_PROXY",
    "RATING_POWER",
    "SCORE_MODE",
    "GENRE_BOOST",
)


def _build_user_profiles(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    tagged = ratings[["user_id", "item_id", "rating"]].merge(
        movies[GENRES], left_on="item_id", right_index=True, how="left"
    )
    profiles = pd.DataFrame(
        {
            genre: tagged.loc[tagged[genre] == 1]
            .groupby("user_id")["rating"]
            .mean()
            for genre in GENRES
        }
    )
    demos = ratings.drop_duplicates("user_id").set_index("user_id")[["age", "gender"]]
    profiles["age"] = demos["age"]
    profiles["gender_m"] = (demos["gender"] == "M").astype(float)
    return profiles


def _load_parameters() -> tuple[dict, dict]:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Hyperparamètres absents : {PARAMS_PATH}. Lance d'abord le tuner."
        )
    result = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    missing = [name for name in TUNED_PARAMETERS if name not in result]
    if missing:
        raise ValueError(
            "JSON Optuna incomplet, paramètres manquants : " + ", ".join(missing)
        )
    params = {name: result[name] for name in TUNED_PARAMETERS}
    params["RANDOM_STATE"] = RANDOM_STATE
    return params, result


def build_artifact() -> dict:
    params, tuning_result = _load_parameters()
    data = load_dataframe()
    split = split_dataframe(data)
    training = split.train_validation

    # Les métadonnées de catalogue ne contiennent pas la cible et peuvent être
    # exposées intégralement, même si les notes test restent strictement isolées.
    movies = data.drop_duplicates("item_id").set_index("item_id")
    film_genres = movies[GENRES].astype(float)
    movie_title = {int(movie_id): str(title) for movie_id, title in movies["title"].items()}
    title_to_id = {title.lower(): movie_id for movie_id, title in movie_title.items()}

    film_stats = training.groupby("item_id")["rating"].agg(
        n_ratings="count", mean_rating="mean"
    )
    eligible = film_stats.loc[
        film_stats["n_ratings"] >= int(params["MIN_SUPPORT"])
    ]
    eligible_ids = {int(movie_id) for movie_id in eligible.index}
    pop_count = {
        int(movie_id): int(row.n_ratings) for movie_id, row in eligible.iterrows()
    }

    profiles = _build_user_profiles(training, movies)
    feature_columns = GENRES + ["age", "gender_m"]
    profile_values = profiles[feature_columns].fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler().fit(profile_values)
    kmeans = KMeans(
        n_clusters=int(params["N_CLUSTERS"]),
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = kmeans.fit_predict(scaler.transform(profile_values))
    user_cluster = {
        int(user_id): int(cluster)
        for user_id, cluster in zip(profiles.index.astype(int), labels)
    }

    liked = liked_interactions(training, training)
    liked["cluster"] = liked["user_id"].map(user_cluster)
    liked = liked.dropna(subset=["cluster"])
    liked["weighted_rating"] = np.power(
        liked["rating"].to_numpy(dtype=float),
        float(params["RATING_POWER"]),
    )
    cluster_film_scores: dict[int, dict[int, float]] = {}
    for cluster, group in liked.groupby("cluster"):
        grouped = group.groupby("item_id")["weighted_rating"]
        scores = (
            grouped.mean()
            if params["SCORE_MODE"] == "mean"
            else grouped.sum()
        )
        cluster_film_scores[int(cluster)] = {
            int(movie_id): float(score)
            for movie_id, score in scores.items()
            if int(movie_id) in eligible_ids
        }

    film_liker_demo: dict[int, tuple[float, float, int]] = {}
    for movie_id, group in liked.groupby("item_id"):
        movie_id = int(movie_id)
        if movie_id in eligible_ids:
            film_liker_demo[movie_id] = (
                float(group["age"].mean()),
                float((group["gender"] == "M").mean()),
                int(len(group)),
            )

    return {
        "model": "user_cluster",
        "genres": list(GENRES),
        "scaler": scaler,
        "kmeans_centers": np.asarray(kmeans.cluster_centers_, dtype=float),
        "cluster_film_scores": cluster_film_scores,
        "film_liker_demo": film_liker_demo,
        "pop_count": pop_count,
        "film_genre_matrix": {
            int(movie_id): film_genres.loc[movie_id].to_numpy(dtype=float).tolist()
            for movie_id in film_genres.index
        },
        "movie_title": movie_title,
        "title_to_id": title_to_id,
        "eligible_ids": eligible_ids,
        "params": params,
        "meta": {
            "source": str(PARAMS_PATH.relative_to(ROOT)),
            "model": "cluster_pure",
            "training_split": "train+validation",
            "test_used": False,
            "protocol": tuning_result.get("protocol", protocol_metadata()),
            "dataset_sha256": dataset_fingerprint(),
            "runtime_versions": runtime_versions(),
            "validation_score": tuning_result.get("validation_score"),
            "metric": tuning_result.get("metric"),
            "n_clusters": int(params["N_CLUSTERS"]),
            "n_users_profiles": int(len(profiles)),
            "n_eligible": int(len(eligible_ids)),
            "n_titles": int(len(movie_title)),
            "cluster_sizes": {
                int(cluster): int(size)
                for cluster, size in pd.Series(labels).value_counts().items()
            },
        },
    }


def main() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    joblib.dump(artifact, OUT_PATH, compress=3)
    params = artifact["params"]
    meta = artifact["meta"]
    print(
        f"Exporté → {OUT_PATH}\n"
        f"  modèle=cluster_pure | entraînement=train+validation | test intact\n"
        f"  K={params['N_CLUSTERS']} | MIN_SUPPORT={params['MIN_SUPPORT']} | "
        f"SCORE_MODE={params['SCORE_MODE']} | RATING_POWER={params['RATING_POWER']}\n"
        f"  profils={meta['n_users_profiles']} | eligible={meta['n_eligible']} | "
        f"titres={meta['n_titles']}"
    )
    return OUT_PATH


if __name__ == "__main__":
    main()
