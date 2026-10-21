#!/usr/bin/env python3
"""Fine-tune MODEL_CLUSTER (groupes d'utilisateurs par goûts) avec Optuna.

Même recommandeur que ``notbook/MODEL_CLUSTER.ipynb`` :
profils (moyenne de notes par genre + âge / sexe), KMeans, classement des
films aimés dans le groupe, repli popularité si le groupe est trop pauvre.

Métrique : Precision@5 en mode produit (âge + sexe + 1–3 films, cold-start).
``TRAIN_FRAC``, ``LIKED_THRESHOLD``, ``MAX_SEEDS`` et ``RANDOM_STATE`` sont
fixés pour que les essais restent comparables au notebook.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# Protocole du notebook (ne pas tuner : ça changerait les cas d'évaluation).
FIXED_TRAIN_FRAC = 0.65
FIXED_RANDOM_STATE = 42
FIXED_MAX_SEEDS = 3
FIXED_LIKED_THRESHOLD = 4
REC_K = 10  # comme evaluate_product ; la métrique ne lit que le top 5

ROOT = Path(__file__).resolve().parents[1]
N_TRIALS_DEFAULT = 300

_STATE: dict | None = None


def find_data_dir() -> Path:
    for p in [Path("dataset"), Path("../dataset"), ROOT / "dataset"]:
        if (p / "u.data.csv").exists():
            return p
    raise FileNotFoundError("Répertoire dataset/ introuvable.")


def load_dataframe() -> pd.DataFrame:
    data_dir = find_data_dir()
    notes = pd.read_csv(data_dir / "u.data.csv")
    films = pd.read_csv(data_dir / "u.item.csv")
    users = pd.read_csv(data_dir / "u.user.csv").drop(
        columns=["occupation"], errors="ignore"
    )
    df = notes.merge(users, on="user_id", how="left").merge(
        films, left_on="item_id", right_on="movie_id", how="left"
    )
    return df.loc[df["unknown"] != 1].drop(columns=["unknown"]).copy()


def _build_user_genre_profiles(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Moyenne de notes par genre (users × GENRES) + âge / sexe."""
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


def _query_vector(seeds, age, gender, film_genre_X: pd.DataFrame, scaler: StandardScaler):
    """Profil cold-start : proxy LIKED_THRESHOLD sur les genres des seeds."""
    genre_means = np.zeros(len(GENRES), dtype=float)
    seed_list = [s for s in seeds if s in film_genre_X.index]
    if seed_list:
        proxy = float(FIXED_LIKED_THRESHOLD)
        for j, g in enumerate(GENRES):
            present = film_genre_X.loc[seed_list, g].to_numpy()
            if present.sum() > 0:
                genre_means[j] = proxy
    age_val = 30.0 if age is None else float(age)
    gender_m = 1.0 if (gender or "M") == "M" else 0.0
    raw = np.concatenate([genre_means, [age_val, gender_m]])
    return scaler.transform(raw.reshape(1, -1))[0]


def _prepare(df: pd.DataFrame) -> dict:
    """Split, cas produit et profils : indépendants des hypers tunés."""
    train = df.sample(frac=FIXED_TRAIN_FRAC, random_state=FIXED_RANDOM_STATE)
    test = df.drop(train.index)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    movies = df.drop_duplicates("item_id").set_index("item_id")
    film_genre_X = movies[GENRES].astype(float)

    film_stats = (
        train.groupby("item_id")["rating"]
        .agg(n_ratings="count", mean_rating="mean")
        .reset_index()
    )
    train_counts = train.groupby("item_id").size()

    liked_train = train.loc[train["rating"] >= FIXED_LIKED_THRESHOLD].copy()
    liked_test = test.loc[test["rating"] >= FIXED_LIKED_THRESHOLD]
    cand = liked_train.copy()
    cand["pop"] = cand["item_id"].map(train_counts)

    query_map: dict = {}
    for uid, grp in cand.groupby("user_id", sort=True):
        sorted_u = grp.sort_values(["rating", "pop"], ascending=[False, False])
        query_map[uid] = list(sorted_u["item_id"].unique()[:FIXED_MAX_SEEDS])

    user_demo = train.drop_duplicates("user_id").set_index("user_id")[["age", "gender"]]
    relevant = liked_test.groupby("user_id")["item_id"].apply(set)
    eval_cases = pd.concat(
        [
            pd.Series(query_map, name="query_ids"),
            user_demo,
            relevant.rename("relevant_ids"),
        ],
        axis=1,
        join="inner",
    ).reset_index(drop=True)
    eval_cases["relevant_ids"] = [
        set(rel) - set(qids)
        for rel, qids in zip(eval_cases["relevant_ids"], eval_cases["query_ids"])
    ]
    eval_cases = eval_cases.loc[eval_cases["relevant_ids"].map(len) > 0].reset_index(
        drop=True
    )

    profiles = _build_user_genre_profiles(train, movies)
    feat_cols = GENRES + ["age", "gender_m"]
    X_raw = profiles[feat_cols].fillna(0.0)
    scaler = StandardScaler().fit(X_raw.values)
    queries = np.vstack(
        [
            _query_vector(qids, age, gender, film_genre_X, scaler)
            for qids, age, gender in zip(
                eval_cases["query_ids"], eval_cases["age"], eval_cases["gender"]
            )
        ]
    )

    return {
        "film_stats": film_stats,
        "liked_train": liked_train[["user_id", "item_id", "rating"]].copy(),
        "user_ids": X_raw.index.astype(int).to_numpy(),
        "X_scaled": scaler.transform(X_raw.values),
        "queries": queries,
        "seed_sets": [set(int(x) for x in q) for q in eval_cases["query_ids"]],
        "relevants": [set(int(x) for x in rel) for rel in eval_cases["relevant_ids"]],
        "n_cases": int(len(eval_cases)),
    }


def _init_worker(df_dict: dict) -> None:
    global _STATE
    _STATE = _prepare(pd.DataFrame(df_dict))


def _precision_at_5(recommended: list, relevant: set) -> float:
    top = recommended[:5]
    if not top:
        return 0.0
    return sum(1 for mid in top if mid in relevant) / 5.0


def _eval_trial_config(cfg: dict) -> float:
    """Precision@5 de recommend_user_cluster (cold-start produit)."""
    global _STATE
    assert _STATE is not None

    min_support = int(cfg["MIN_SUPPORT"])
    n_clusters = int(cfg["N_CLUSTERS"])
    min_group = int(cfg["MIN_GROUP_CANDIDATES"])

    X_scaled = _STATE["X_scaled"]
    if n_clusters < 2 or n_clusters > len(X_scaled):
        return 0.0

    film_stats = _STATE["film_stats"]
    eligible = film_stats.loc[film_stats["n_ratings"] >= min_support].set_index(
        "item_id"
    )
    if len(eligible) == 0 or _STATE["n_cases"] == 0:
        return 0.0
    eligible_ids = set(int(i) for i in eligible.index)
    pop_count = eligible["n_ratings"]

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=FIXED_RANDOM_STATE, n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    user_cluster = {
        int(uid): int(lab) for uid, lab in zip(_STATE["user_ids"], labels)
    }

    liked = _STATE["liked_train"].copy()
    liked["cluster"] = liked["user_id"].map(user_cluster)
    cluster_film_scores: dict[int, dict] = {}
    for cid, grp in liked.dropna(subset=["cluster"]).groupby("cluster"):
        cluster_film_scores[int(cid)] = grp.groupby("item_id")["rating"].sum().to_dict()

    centers = kmeans.cluster_centers_
    queries = _STATE["queries"]
    dists = np.linalg.norm(centers - queries[:, None, :], axis=2)
    assigned = np.argmin(dists, axis=1)

    # Listes déjà triées : le repli popularité ne dépend que du support.
    pop_ranked = sorted(
        ((int(mid), int(n)) for mid, n in pop_count.items()),
        key=lambda x: (-x[1], x[0]),
    )
    ranked_by_cluster: dict[int, list] = {}
    for cid, scores in cluster_film_scores.items():
        ranked = [
            (int(mid), sc)
            for mid, sc in scores.items()
            if int(mid) in eligible_ids
        ]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        ranked_by_cluster[cid] = ranked

    p5 = []
    for i, cid in enumerate(assigned):
        seen = _STATE["seed_sets"][i]
        out = []
        for mid, _sc in ranked_by_cluster.get(int(cid), ()):
            if mid not in seen:
                out.append(mid)
            if len(out) >= REC_K:
                break
        if len(out) < min_group:
            for mid, _n in pop_ranked:
                if mid not in seen and mid not in out:
                    out.append(mid)
                if len(out) >= REC_K:
                    break
        p5.append(_precision_at_5(out[:REC_K], _STATE["relevants"][i]))

    return float(np.mean(p5))


def main(n_trials: int = N_TRIALS_DEFAULT, out_path: Path | None = None) -> dict:
    print("=== Chargement des données (MODEL_CLUSTER — groupes users) ===")
    df = load_dataframe()
    df_dict = df.to_dict(orient="list")

    # Un passage pour annoncer le nombre de cas (les workers refont le split).
    preview = _prepare(df)
    print(
        f"Cas d'évaluation : {preview['n_cases']} | "
        f"TRAIN_FRAC={FIXED_TRAIN_FRAC} | aimé ≥ {FIXED_LIKED_THRESHOLD}"
    )
    print(
        f"=== Optuna MODEL_CLUSTER ({n_trials} essais) — "
        f"P@5 user_cluster_groups ==="
    )

    sampler = optuna.samplers.TPESampler(seed=FIXED_RANDOM_STATE, n_startup_trials=20)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    max_workers = os.cpu_count() or 4
    print(f"Parallélisation sur {max_workers} cœurs CPU")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(df_dict,),
    ) as executor:
        for i in range(0, n_trials, max_workers):
            batch_size = min(max_workers, n_trials - i)
            trials = [study.ask() for _ in range(batch_size)]
            configs = [
                {
                    "MIN_SUPPORT": t.suggest_int("MIN_SUPPORT", 8, 40),
                    "N_CLUSTERS": t.suggest_int("N_CLUSTERS", 4, 20),
                    "MIN_GROUP_CANDIDATES": t.suggest_int(
                        "MIN_GROUP_CANDIDATES", 1, 10
                    ),
                }
                for t in trials
            ]
            results = list(executor.map(_eval_trial_config, configs))
            for trial, score in zip(trials, results):
                study.tell(trial, score)

            completed = min(i + max_workers, n_trials)
            print(
                f"[{completed}/{n_trials}] Meilleur P@5 actuel : "
                f"{study.best_value:.4f}"
            )

    best = dict(study.best_params)
    best["LIKED_THRESHOLD"] = FIXED_LIKED_THRESHOLD
    best["MAX_SEEDS"] = FIXED_MAX_SEEDS
    best["TRAIN_FRAC"] = FIXED_TRAIN_FRAC
    best["RANDOM_STATE"] = FIXED_RANDOM_STATE
    best["metric"] = "P@5_product_user_cluster"
    best["best_P@5"] = study.best_value

    print("\n=== MEILLEURS HYPERPARAMÈTRES (MODEL_CLUSTER) ===")
    print(f"P@5 Max : {study.best_value:.4f}")
    print(json.dumps(best, indent=2))

    out_path = out_path or (ROOT / "results" / "best_hyperparams_cluster_optuna.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(f"\nRésultats exportés dans : {out_path}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna sur le clustering par groupes de goûts (MODEL_CLUSTER)."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS_DEFAULT,
        help=f"Nombre d'essais Optuna (défaut {N_TRIALS_DEFAULT}).",
    )
    args = parser.parse_args()
    main(n_trials=args.n_trials)
