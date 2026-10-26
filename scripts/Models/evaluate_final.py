#!/usr/bin/env python3
"""Évaluation finale commune de la baseline, Cluster et User–CF.

Ce script est le seul point où le split test est consommé. Les deux modèles
sont évalués par l'API réellement servie dans ``nextwatch_recommend``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nextwatch_recommend  # noqa: E402
from scripts.evaluation_protocol import (  # noqa: E402
    GENRES,
    build_eval_cases,
    dataset_fingerprint,
    load_dataframe,
    precision_at_k,
    protocol_metadata,
    runtime_versions,
    split_dataframe,
)

RESULTS_DIR = ROOT / "results"
MAX_K = 10
BASELINE_MIN_RATINGS = 5


def _validate_artifact(model: nextwatch_recommend.ModelName) -> None:
    """Refuse un artefact non traçable vers le protocole final commun."""
    artifact = nextwatch_recommend.load(model)
    meta = artifact.get("meta")
    label = "Cluster" if model == "cluster" else "UserCF"
    if not isinstance(meta, dict):
        raise RuntimeError(
            f"Artefact {label} incompatible : métadonnées 'meta' absentes."
        )

    errors: list[str] = []
    if meta.get("training_split") != "train+validation":
        errors.append(
            "training_split doit valoir 'train+validation' "
            f"(reçu {meta.get('training_split')!r})"
        )
    if meta.get("test_used") is not False:
        errors.append(f"test_used doit valoir False (reçu {meta.get('test_used')!r})")
    expected_dataset = dataset_fingerprint()
    if meta.get("dataset_sha256") != expected_dataset:
        errors.append("dataset_sha256 ne correspond pas aux données courantes")
    if meta.get("runtime_versions") != runtime_versions():
        errors.append("runtime_versions ne correspond pas à l'environnement courant")

    expected = protocol_metadata()
    artifact_protocol = meta.get("protocol")
    if not isinstance(artifact_protocol, dict):
        errors.append("meta.protocol est absent ou invalide")
    else:
        mismatches = [
            f"{key}={artifact_protocol.get(key)!r} (attendu {value!r})"
            for key, value in expected.items()
            if artifact_protocol.get(key) != value
        ]
        if mismatches:
            errors.append("meta.protocol diffère : " + "; ".join(mismatches))

    if errors:
        raise RuntimeError(
            f"Artefact {label} incompatible avec le protocole commun :\n- "
            + "\n- ".join(errors)
            + "\nRégénérez-le avec le script d'export correspondant avant "
            "l'évaluation finale."
        )


def _full_protocol() -> dict:
    return {
        **protocol_metadata(),
        "reference_split": "train+validation",
        "heldout_split": "test",
        "case_builder": "scripts.evaluation_protocol.build_eval_cases",
        "same_cases_for_all_methods": True,
        "model_inference": "nextwatch_recommend.recommend",
        "metrics": {
            "P@5": "hits_in_top_5 / 5",
            "P@10": "hits_in_top_10 / 10",
            "recs_top5": "first_5_recommendation_ids",
        },
        "baseline": {
            "candidate_genres": "at_least_one_genre_shared_with_seeds",
            "minimum_reference_ratings": BASELINE_MIN_RATINGS,
            "ranking": "mean_rating_desc_then_item_id_asc",
            "excluded_items": "seeds_only",
        },
        "artifact_requirements": {
            "training_split": "train+validation",
            "test_used": False,
            "protocol": "common_protocol_metadata",
        },
    }


def _build_baseline(reference: pd.DataFrame) -> Callable[[list[int]], list[int]]:
    stats = reference.groupby("item_id")["rating"].agg(["count", "mean"])
    eligible = stats.loc[stats["count"] >= BASELINE_MIN_RATINGS].copy()
    eligible.index = eligible.index.astype(int)

    movie_genres = (
        reference.drop_duplicates("item_id")
        .set_index("item_id")[GENRES]
        .astype(bool)
    )
    movie_genres.index = movie_genres.index.astype(int)
    eligible = eligible.join(movie_genres, how="inner")

    def recommend_baseline(seeds: list[int]) -> list[int]:
        seed_rows = movie_genres.reindex(seeds).fillna(False)
        seed_mask = seed_rows.any(axis=0).to_numpy(dtype=bool)
        if not seed_mask.any():
            return []

        overlaps = eligible[GENRES].to_numpy(dtype=bool)[:, seed_mask].any(axis=1)
        candidates = eligible.loc[overlaps].copy()
        candidates = candidates.loc[~candidates.index.isin(set(seeds))]
        candidates = candidates.reset_index(names="item_id")
        candidates = candidates.sort_values(
            ["mean", "item_id"], ascending=[False, True], kind="mergesort"
        )
        return [int(movie_id) for movie_id in candidates["item_id"].iloc[:MAX_K]]

    return recommend_baseline


def _evaluate(
    cases: pd.DataFrame,
    recommend_fn: Callable[[pd.Series], list[int]],
) -> dict[str, list]:
    p5: list[float] = []
    p10: list[float] = []
    recs_top5: list[list[int]] = []
    for _, case in cases.iterrows():
        recommendations = [int(x) for x in recommend_fn(case)[:MAX_K]]
        relevant = {int(x) for x in case["relevant_ids"]}
        p5.append(float(precision_at_k(recommendations, relevant, 5)))
        p10.append(float(precision_at_k(recommendations, relevant, 10)))
        recs_top5.append(recommendations[:5])
    return {"P@5": p5, "P@10": p10, "recs_top5": recs_top5}


def _bundle(
    source: str,
    method_name: str,
    values: dict[str, list],
    *,
    n_cases: int,
    protocol: dict,
    film_n_ratings: dict[str, int],
) -> dict:
    return {
        "source": source,
        "n_cases": n_cases,
        "random_state": protocol["random_state"],
        "train_frac": protocol["train_frac"],
        "protocol": protocol,
        "film_n_ratings": film_n_ratings,
        "methods": {method_name: values},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # Valider avant de construire les cas test évite toute évaluation avec un
    # artefact entraîné ou réglé selon un protocole différent.
    _validate_artifact("cluster")
    _validate_artifact("cf")

    split = split_dataframe(load_dataframe())
    reference = split.train_validation
    cases = build_eval_cases(reference, split.test)
    if cases.empty:
        raise RuntimeError("Aucun cas final évaluable n'a été construit.")

    baseline = _build_baseline(reference)
    evaluations = {
        "baseline_overlap_mean": _evaluate(
            cases, lambda case: baseline(case["query_ids"])
        ),
        "user_cluster_groups": _evaluate(
            cases,
            lambda case: nextwatch_recommend.recommend(
                age=case["age"],
                gender=case["gender"],
                liked_movies=case["query_ids"],
                model="cluster",
                k=MAX_K,
                return_ids=True,
            ),
        ),
        "user_user_cf": _evaluate(
            cases,
            lambda case: nextwatch_recommend.recommend(
                age=case["age"],
                gender=case["gender"],
                liked_movies=case["query_ids"],
                model="cf",
                k=MAX_K,
                return_ids=True,
            ),
        ),
    }

    n_cases = int(len(cases))
    protocol = _full_protocol()
    film_n_ratings = {
        str(int(movie_id)): int(count)
        for movie_id, count in reference.groupby("item_id").size().items()
    }
    bundles = {
        "eval_baseline.json": _bundle(
            "baseline",
            "baseline_overlap_mean",
            evaluations["baseline_overlap_mean"],
            n_cases=n_cases,
            protocol=protocol,
            film_n_ratings=film_n_ratings,
        ),
        "eval_cluster.json": _bundle(
            "cluster",
            "user_cluster_groups",
            evaluations["user_cluster_groups"],
            n_cases=n_cases,
            protocol=protocol,
            film_n_ratings=film_n_ratings,
        ),
        "eval_cf.json": _bundle(
            "cf",
            "user_user_cf",
            evaluations["user_user_cf"],
            n_cases=n_cases,
            protocol=protocol,
            film_n_ratings=film_n_ratings,
        ),
    }

    summary = {
        "n_cases": n_cases,
        "protocol": protocol,
        "scores": {
            method: {
                "P@5": float(np.mean(values["P@5"])),
                "P@10": float(np.mean(values["P@10"])),
            }
            for method, values in evaluations.items()
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, payload in bundles.items():
        _write_json(RESULTS_DIR / filename, payload)
    _write_json(RESULTS_DIR / "final_evaluation.json", summary)

    print(f"Évaluation finale écrite dans {RESULTS_DIR} ({n_cases} cas).")
    for method, scores in summary["scores"].items():
        print(f"  {method}: P@5={scores['P@5']:.4f} | P@10={scores['P@10']:.4f}")


if __name__ == "__main__":
    main()
