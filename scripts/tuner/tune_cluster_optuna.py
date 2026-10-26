#!/usr/bin/env python3
"""Optimise le modèle Cluster pur avec le protocole produit commun."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation_protocol import (  # noqa: E402
    GENRES,
    RANDOM_STATE,
    build_eval_cases,
    liked_interactions,
    load_dataframe,
    precision_at_k,
    protocol_metadata,
    split_dataframe,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

REC_K = 5
N_TRIALS_DEFAULT = 300
_STATE: dict | None = None


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


def _prepare(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    movies = pd.concat([train, validation], ignore_index=True).drop_duplicates(
        "item_id"
    ).set_index("item_id")
    film_genres = movies[GENRES].astype(float)
    film_stats = train.groupby("item_id")["rating"].agg(
        n_ratings="count", mean_rating="mean"
    )
    liked_train = liked_interactions(train, train)
    cases = build_eval_cases(train, validation)

    profiles = _build_user_profiles(train, movies)
    feature_columns = GENRES + ["age", "gender_m"]
    profile_values = profiles[feature_columns].fillna(0.0)

    seed_lists: list[list[int]] = []
    seed_masks: list[np.ndarray] = []
    relevant: list[set[int]] = []
    ages: list[float] = []
    genders_m: list[float] = []
    for row in cases.itertuples(index=False):
        seeds = [int(movie_id) for movie_id in row.query_ids]
        mask = np.zeros(len(GENRES), dtype=bool)
        present = [movie_id for movie_id in seeds if movie_id in film_genres.index]
        if present:
            mask = film_genres.loc[present].to_numpy(dtype=float).sum(axis=0) > 0
        seed_lists.append(seeds)
        seed_masks.append(mask)
        relevant.append({int(movie_id) for movie_id in row.relevant_ids})
        ages.append(float(row.age))
        genders_m.append(1.0 if row.gender == "M" else 0.0)

    return {
        "film_stats": film_stats,
        "film_genres": film_genres,
        "liked_train": liked_train[["user_id", "item_id", "rating"]].copy(),
        "profile_user_ids": profile_values.index.astype(int).to_numpy(),
        "profile_values": profile_values.to_numpy(dtype=float),
        "seed_lists": seed_lists,
        "seed_masks": np.asarray(seed_masks, dtype=bool),
        "relevant": relevant,
        "ages": np.asarray(ages, dtype=float),
        "genders_m": np.asarray(genders_m, dtype=float),
        "n_cases": len(cases),
        "n_profiles": len(profiles),
    }


def _init_worker(train_dict: dict, validation_dict: dict) -> None:
    global _STATE
    _STATE = _prepare(pd.DataFrame(train_dict), pd.DataFrame(validation_dict))


def _evaluate_config(config: dict) -> float:
    assert _STATE is not None

    n_clusters = int(config["N_CLUSTERS"])
    if (
        _STATE["n_cases"] == 0
        or n_clusters < 2
        or n_clusters > _STATE["n_profiles"]
    ):
        return 0.0

    min_support = int(config["MIN_SUPPORT"])
    eligible = _STATE["film_stats"].loc[
        _STATE["film_stats"]["n_ratings"] >= min_support
    ]
    if eligible.empty:
        return 0.0
    eligible_ids = {int(movie_id) for movie_id in eligible.index}
    popularity = sorted(
        ((int(movie_id), int(row.n_ratings)) for movie_id, row in eligible.iterrows()),
        key=lambda item: (-item[1], item[0]),
    )

    scaler = StandardScaler().fit(_STATE["profile_values"])
    scaled_profiles = scaler.transform(_STATE["profile_values"])
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = kmeans.fit_predict(scaled_profiles)
    user_cluster = {
        int(user_id): int(cluster)
        for user_id, cluster in zip(_STATE["profile_user_ids"], labels)
    }

    liked = _STATE["liked_train"].copy()
    liked["cluster"] = liked["user_id"].map(user_cluster)
    liked = liked.dropna(subset=["cluster"])
    liked["weighted_rating"] = np.power(
        liked["rating"].to_numpy(dtype=float),
        float(config["RATING_POWER"]),
    )
    grouped_scores: dict[int, dict[int, float]] = {}
    for cluster, group in liked.groupby("cluster"):
        ratings = group.groupby("item_id")["weighted_rating"]
        scores = ratings.mean() if config["SCORE_MODE"] == "mean" else ratings.sum()
        grouped_scores[int(cluster)] = {
            int(movie_id): float(score)
            for movie_id, score in scores.items()
            if int(movie_id) in eligible_ids
        }

    queries = np.tile(np.asarray(scaler.mean_, dtype=float), (_STATE["n_cases"], 1))
    for index, mask in enumerate(_STATE["seed_masks"]):
        queries[index, : len(GENRES)][mask] = float(config["SEED_PROXY"])
    queries[:, len(GENRES)] = _STATE["ages"]
    queries[:, len(GENRES) + 1] = _STATE["genders_m"]
    scaled_queries = scaler.transform(queries)
    assigned = np.argmin(
        np.linalg.norm(
            scaled_queries[:, None, :] - kmeans.cluster_centers_[None, :, :],
            axis=2,
        ),
        axis=1,
    )

    genre_boost = float(config["GENRE_BOOST"])
    min_group = int(config["MIN_GROUP_CANDIDATES"])
    precisions: list[float] = []
    for index, cluster in enumerate(assigned):
        seeds = set(_STATE["seed_lists"][index])
        seed_mask = _STATE["seed_masks"][index]
        ranked: list[tuple[int, int, float]] = []
        for movie_id, score in grouped_scores.get(int(cluster), {}).items():
            if movie_id in seeds:
                continue
            genres = _STATE["film_genres"].loc[movie_id].to_numpy(dtype=float) > 0
            overlap = int(np.sum(genres & seed_mask))
            # Le tri lexicographique reproduit exactement l'inférence produit.
            boosted_score = score * (genre_boost**overlap)
            ranked.append((movie_id, overlap, boosted_score))
        ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
        recommendations = [movie_id for movie_id, _, _ in ranked[:REC_K]]

        if len(recommendations) < min_group:
            for movie_id, _ in popularity:
                if movie_id not in seeds and movie_id not in recommendations:
                    recommendations.append(movie_id)
                if len(recommendations) >= REC_K:
                    break

        precisions.append(
            precision_at_k(recommendations, _STATE["relevant"][index], k=REC_K)
        )
    return float(np.mean(precisions))


def _suggest_config(trial: optuna.Trial) -> dict:
    return {
        "MIN_SUPPORT": trial.suggest_int("MIN_SUPPORT", 5, 45),
        "N_CLUSTERS": trial.suggest_int("N_CLUSTERS", 4, 24),
        "MIN_GROUP_CANDIDATES": trial.suggest_int(
            "MIN_GROUP_CANDIDATES", 1, REC_K
        ),
        "SEED_PROXY": trial.suggest_float("SEED_PROXY", 2.5, 5.0, step=0.25),
        "RATING_POWER": trial.suggest_float(
            "RATING_POWER", 0.5, 3.0, step=0.25
        ),
        "SCORE_MODE": trial.suggest_categorical("SCORE_MODE", ["sum", "mean"]),
        "GENRE_BOOST": trial.suggest_float("GENRE_BOOST", 1.0, 1.6, step=0.05),
    }


def main(n_trials: int = N_TRIALS_DEFAULT, out_path: Path | None = None) -> dict:
    split = split_dataframe(load_dataframe())
    train = split.train
    validation = split.validation
    train_dict = train.to_dict(orient="list")
    validation_dict = validation.to_dict(orient="list")

    preview = _prepare(train, validation)
    print(
        "=== Optuna Cluster pur ===\n"
        f"Cas validation={preview['n_cases']} | profils train={preview['n_profiles']} | "
        f"essais={n_trials}"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE,
            n_startup_trials=min(40, max(1, n_trials)),
        ),
    )
    max_workers = min(os.cpu_count() or 4, max(1, n_trials))
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(train_dict, validation_dict),
    ) as executor:
        for start in range(0, n_trials, max_workers):
            batch_size = min(max_workers, n_trials - start)
            trials = [study.ask() for _ in range(batch_size)]
            scores = list(
                executor.map(
                    _evaluate_config,
                    [_suggest_config(trial) for trial in trials],
                )
            )
            for trial, score in zip(trials, scores):
                study.tell(trial, score)
            print(
                f"[{start + batch_size}/{n_trials}] "
                f"meilleur P@{REC_K} validation={study.best_value:.4f}"
            )

    result = dict(study.best_params)
    result.update(
        {
            "metric": f"precision@{REC_K}",
            "validation_score": float(study.best_value),
            "n_trials": int(n_trials),
            "best_trial_number": int(study.best_trial.number),
            "protocol": protocol_metadata(),
            "training_split": "train",
            "optimization_split": "validation",
            "model": "cluster_pure",
        }
    )
    destination = out_path or (
        ROOT / "results" / "best_hyperparams_cluster_optuna.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Résultats exportés dans : {destination}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna pour le modèle Cluster pur.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS_DEFAULT,
        help=f"Nombre d'essais (défaut : {N_TRIALS_DEFAULT}).",
    )
    arguments = parser.parse_args()
    main(n_trials=arguments.n_trials)
