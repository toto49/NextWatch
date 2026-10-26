#!/usr/bin/env python3
"""Optimise le modèle UserCF servi, exclusivement sur la validation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation_protocol import (  # noqa: E402
    build_eval_cases,
    liked_interactions,
    load_dataframe,
    precision_at_k,
    protocol_metadata,
    split_dataframe,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 300
_TRAIN: pd.DataFrame | None = None
_EVAL_CASES: pd.DataFrame | None = None


def _init_worker(train_dict: dict, cases_dict: dict) -> None:
    global _TRAIN, _EVAL_CASES
    _TRAIN = pd.DataFrame(train_dict)
    _EVAL_CASES = pd.DataFrame(cases_dict)


def _evaluate_config(cfg: dict) -> float:
    """Calcule Precision@5 avec exactement le chemin UserCF du produit."""
    assert _TRAIN is not None
    assert _EVAL_CASES is not None

    min_support = int(cfg["MIN_SUPPORT"])
    cf_n_user = int(cfg["CF_N_USER"])
    demo_sex_weight = float(cfg["DEMO_SEX_WEIGHT"])
    age_scale = float(cfg["AGE_SCALE"])
    rating_power = float(cfg["RATING_POWER"])

    counts = _TRAIN.groupby("item_id").size()
    eligible_ids = {
        int(movie_id) for movie_id in counts[counts >= min_support].index
    }
    liked_train = liked_interactions(_TRAIN, _TRAIN)
    liked_train = liked_train.loc[liked_train["item_id"].isin(eligible_ids)]

    liked_by_user = (
        liked_train.groupby("user_id")[["item_id", "rating"]]
        .apply(
            lambda rows: list(
                zip(
                    rows["item_id"].astype(int),
                    rows["rating"].astype(float),
                )
            )
        )
        .to_dict()
    )
    fans_of = {
        int(movie_id): list(
            zip(
                group["user_id"].astype(int),
                group["rating"].astype(float),
                group["age"].astype(float),
                group["gender"].astype(str),
            )
        )
        for movie_id, group in liked_train.groupby("item_id")
    }

    scores_by_case: list[float] = []
    for row in _EVAL_CASES.itertuples(index=False):
        seeds = {int(movie_id) for movie_id in row.query_ids}
        relevant = {int(movie_id) for movie_id in row.relevant_ids}
        if not relevant:
            continue

        neighbors: list[tuple[int, float]] = []
        for query_id in seeds:
            for user_id, rating, age, gender in fans_of.get(query_id, []):
                sex = 1.0 if str(gender) == str(row.gender) else 0.0
                age_term = 1.0 - min(
                    abs(float(row.age) - float(age)) / age_scale,
                    1.0,
                )
                similarity = (
                    demo_sex_weight * sex
                    + (1.0 - demo_sex_weight) * age_term
                )
                weighted_rating = (rating**rating_power) / (5.0**rating_power)
                neighbors.append((user_id, similarity * weighted_rating))

        neighbors.sort(key=lambda pair: -pair[1])
        candidates: dict[int, float] = {}
        for user_id, similarity in neighbors[:cf_n_user]:
            for movie_id, rating in liked_by_user.get(user_id, []):
                if movie_id in seeds:
                    continue
                candidates[movie_id] = candidates.get(movie_id, 0.0) + (
                    similarity * (rating**rating_power)
                )

        recommended = [
            movie_id
            for movie_id, _ in sorted(
                candidates.items(), key=lambda pair: (-pair[1], pair[0])
            )[:5]
        ]
        scores_by_case.append(precision_at_k(recommended, relevant, k=5))

    return float(np.mean(scores_by_case)) if scores_by_case else 0.0


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "MIN_SUPPORT": trial.suggest_int("MIN_SUPPORT", 5, 35),
        "CF_N_USER": trial.suggest_int("CF_N_USER", 30, 300, step=10),
        "DEMO_SEX_WEIGHT": trial.suggest_float(
            "DEMO_SEX_WEIGHT", 0.05, 0.60, step=0.05
        ),
        "AGE_SCALE": trial.suggest_int("AGE_SCALE", 20, 90, step=5),
        "RATING_POWER": trial.suggest_float(
            "RATING_POWER", 0.5, 2.0, step=0.25
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"Nombre d'essais Optuna (défaut : {N_TRIALS}).",
    )
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials doit être supérieur ou égal à 1.")

    split = split_dataframe(load_dataframe())
    eval_cases = build_eval_cases(split.train, split.validation)
    train_dict = split.train.to_dict(orient="list")
    cases_dict = eval_cases.to_dict(orient="list")

    sampler = optuna.samplers.TPESampler(
        seed=protocol_metadata()["random_state"],
        n_startup_trials=min(50, args.n_trials),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    max_workers = min(os.cpu_count() or 4, args.n_trials)

    print(
        f"Optimisation UserCF : {args.n_trials} essais, "
        f"{len(eval_cases)} cas validation, {max_workers} processus"
    )
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(train_dict, cases_dict),
    ) as executor:
        for start in range(0, args.n_trials, max_workers):
            batch_size = min(max_workers, args.n_trials - start)
            trials = [study.ask() for _ in range(batch_size)]
            configs = [_suggest(trial) for trial in trials]
            values = list(executor.map(_evaluate_config, configs))
            for trial, value in zip(trials, values):
                study.tell(trial, value)
            print(
                f"[{start + batch_size}/{args.n_trials}] "
                f"meilleure validation P@5={study.best_value:.4f}"
            )

    result = {
        **study.best_params,
        "validation_score": float(study.best_value),
        "metric": "precision_at_5",
        "model": "usercf",
        "n_trials": args.n_trials,
        "protocol": protocol_metadata(),
    }
    out_path = ROOT / "results" / "best_hyperparams_cf_optuna.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Résultats exportés dans : {out_path}")
    return out_path


if __name__ == "__main__":
    main()