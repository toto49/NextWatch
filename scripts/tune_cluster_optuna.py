#!/usr/bin/env python3
"""Fine-tune MODEL_CLUSTER (groupes d'utilisateurs + fusion UserCF) avec Optuna.

Pipeline (aligné sur ``notbook/MODEL_CLUSTER.ipynb``) :
1. Profils users = moyennes de notes par genre + âge / sexe
2. KMeans → ranking des films aimés dans le groupe
3. UserCF démographique (témoin du notebook)
4. Fusion RRF cluster ⊕ UserCF (leviers Optuna)

Métrique : Precision@5 cold-start produit (âge + sexe + 1–3 films).
Protocole fixe : TRAIN_FRAC=0.65, aimé ≥ 4, MAX_SEEDS=3, seed=42.
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

FIXED_TRAIN_FRAC = 0.65
FIXED_RANDOM_STATE = 42
FIXED_MAX_SEEDS = 3
FIXED_LIKED_THRESHOLD = 4
REC_K = 5
POOL = 80  # taille des listes avant fusion RRF

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


def _prepare(df: pd.DataFrame) -> dict:
    train = df.sample(frac=FIXED_TRAIN_FRAC, random_state=FIXED_RANDOM_STATE)
    test = df.drop(train.index)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    movies = df.drop_duplicates("item_id").set_index("item_id")
    film_genre_X = movies[GENRES].astype(float)
    movie_genres = movies["genres"].to_dict()

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

    liked_by_user = (
        liked_train.groupby("user_id")[["item_id", "rating"]]
        .apply(lambda x: list(zip(x["item_id"].to_numpy(), x["rating"].to_numpy())))
        .to_dict()
    )
    fans_of = {
        int(mid): list(
            zip(
                grp["user_id"].to_numpy(),
                grp["rating"].to_numpy(),
                grp["age"].to_numpy(),
                grp["gender"].to_numpy(),
            )
        )
        for mid, grp in liked_train.groupby("item_id")
    }

    seed_genre_masks = []
    ages, genders_m, gender_labels = [], [], []
    seed_sets, relevants, seed_lists = [], [], []
    for qids, age, gender, rel in zip(
        eval_cases["query_ids"],
        eval_cases["age"],
        eval_cases["gender"],
        eval_cases["relevant_ids"],
    ):
        seeds = [int(x) for x in qids]
        seed_lists.append(seeds)
        seed_sets.append(set(seeds))
        relevants.append({int(x) for x in rel})
        ages.append(float(age))
        genders_m.append(1.0 if gender == "M" else 0.0)
        gender_labels.append(gender)
        mask = np.zeros(len(GENRES), dtype=float)
        present = [s for s in seeds if s in film_genre_X.index]
        if present:
            mask = (film_genre_X.loc[present].to_numpy().sum(axis=0) > 0).astype(float)
        seed_genre_masks.append(mask)

    return {
        "film_stats": film_stats,
        "liked_train": liked_train[["user_id", "item_id", "rating"]].copy(),
        "user_ids": X_raw.index.astype(int).to_numpy(),
        "X_raw": X_raw.to_numpy(dtype=float),
        "n_genres": len(GENRES),
        "film_genre_X": film_genre_X,
        "movie_genres": movie_genres,
        "liked_by_user": liked_by_user,
        "fans_of": fans_of,
        "seed_lists": seed_lists,
        "seed_sets": seed_sets,
        "seed_genre_masks": np.vstack(seed_genre_masks),
        "ages": np.asarray(ages, dtype=float),
        "genders_m": np.asarray(genders_m, dtype=float),
        "gender_labels": gender_labels,
        "relevants": relevants,
        "n_cases": int(len(eval_cases)),
        "n_users_profiles": int(len(X_raw)),
    }


def _init_worker(df_dict: dict) -> None:
    global _STATE
    _STATE = _prepare(pd.DataFrame(df_dict))


def _rrf_fuse(lists: list[list[int]], weights: list[float], rrf_k: float, k: int) -> list[int]:
    scores: dict[int, float] = {}
    for lst, w in zip(lists, weights):
        if w <= 0:
            continue
        for rank, mid in enumerate(lst):
            scores[mid] = scores.get(mid, 0.0) + w / (rrf_k + rank + 1)
    return [m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def _user_cf_list(
    seeds: list[int],
    seen: set[int],
    age: float,
    gender: str,
    *,
    cf_n_user: int,
    demo_sex_weight: float,
    age_scale: float,
    genre_boost: float,
    movie_genres: dict,
    fans_of: dict,
    liked_by_user: dict,
    pool: int,
) -> list[int]:
    neighbors = []
    for qid in seeds:
        for uid, rat, a, g in fans_of.get(qid, ()):
            sex = 1.0 if g == gender else 0.0
            age_term = 1.0 - min(abs(age - float(a)) / age_scale, 1.0)
            sim = (
                demo_sex_weight * sex + (1.0 - demo_sex_weight) * age_term
            ) * float(rat) / 5.0
            neighbors.append((int(uid), sim))
    neighbors.sort(key=lambda x: -x[1])
    neighbors = neighbors[:cf_n_user]

    scores: dict[int, float] = {}
    seed_g = set()
    for qid in seeds:
        seed_g.update(movie_genres.get(qid, []))
    for vid, sim in neighbors:
        for mid, rat in liked_by_user.get(vid, ()):
            mid = int(mid)
            if mid in seen:
                continue
            scores[mid] = scores.get(mid, 0.0) + float(sim) * float(rat)

    if genre_boost > 1.0 and seed_g:
        for mid in list(scores):
            if seed_g & set(movie_genres.get(mid, [])):
                scores[mid] *= genre_boost

    return [m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:pool]]


def _eval_trial_config(cfg: dict) -> float:
    global _STATE
    assert _STATE is not None

    min_support = int(cfg["MIN_SUPPORT"])
    n_clusters = int(cfg["N_CLUSTERS"])
    min_group = int(cfg["MIN_GROUP_CANDIDATES"])
    demo_weight = float(cfg["DEMO_WEIGHT"])
    seed_proxy = float(cfg["SEED_PROXY"])
    rating_power = float(cfg["RATING_POWER"])
    genre_boost = float(cfg["GENRE_BOOST"])
    pop_blend = float(cfg["POP_BLEND"])
    score_mode = cfg["SCORE_MODE"]
    cf_n_user = int(cfg["CF_N_USER"])
    demo_sex_weight = float(cfg["DEMO_SEX_WEIGHT"])
    age_scale = float(cfg["AGE_SCALE"])
    w_cluster = float(cfg["W_CLUSTER"])
    w_cf = float(cfg["W_CF"])
    rrf_k = float(cfg["RRF_K"])

    # Normalise les poids de fusion
    w_sum = w_cluster + w_cf
    if w_sum <= 0:
        return 0.0
    w_cluster /= w_sum
    w_cf /= w_sum

    X_base = _STATE["X_raw"]
    n_g = _STATE["n_genres"]
    n_users = _STATE["n_users_profiles"]
    if n_clusters < 2 or n_clusters > n_users:
        return 0.0

    film_stats = _STATE["film_stats"]
    eligible = film_stats.loc[film_stats["n_ratings"] >= min_support].set_index(
        "item_id"
    )
    if len(eligible) == 0 or _STATE["n_cases"] == 0:
        return 0.0
    eligible_ids = set(int(i) for i in eligible.index)
    pop_count = eligible["n_ratings"].astype(float)
    pop_max = float(pop_count.max()) or 1.0
    pop_norm = {int(mid): float(n) / pop_max for mid, n in pop_count.items()}
    pop_ranked = sorted(pop_norm.items(), key=lambda x: (-x[1], x[0]))

    # --- Clustering ---
    X = X_base.copy()
    X[:, n_g:] *= demo_weight
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=FIXED_RANDOM_STATE, n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    user_cluster = {
        int(uid): int(lab) for uid, lab in zip(_STATE["user_ids"], labels)
    }

    liked = _STATE["liked_train"].copy()
    liked["cluster"] = liked["user_id"].map(user_cluster)
    liked = liked.dropna(subset=["cluster"])
    liked["cluster"] = liked["cluster"].astype(int)
    liked["w"] = np.power(liked["rating"].to_numpy(dtype=float), rating_power)

    cluster_film_scores: dict[int, dict[int, float]] = {}
    for cid, grp in liked.groupby("cluster"):
        agg = (
            grp.groupby("item_id")["w"].mean()
            if score_mode == "mean"
            else grp.groupby("item_id")["w"].sum()
        )
        cluster_film_scores[int(cid)] = {
            int(mid): float(sc)
            for mid, sc in agg.items()
            if int(mid) in eligible_ids
        }

    masks = _STATE["seed_genre_masks"]
    queries = np.zeros((_STATE["n_cases"], n_g + 2), dtype=float)
    queries[:, :n_g] = masks * seed_proxy
    queries[:, n_g] = _STATE["ages"] * demo_weight
    queries[:, n_g + 1] = _STATE["genders_m"] * demo_weight
    queries = scaler.transform(queries)

    centers = kmeans.cluster_centers_
    assigned = np.argmin(
        np.linalg.norm(centers[None, :, :] - queries[:, None, :], axis=2),
        axis=1,
    )

    ranked_by_cluster = {
        cid: sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        for cid, scores in cluster_film_scores.items()
    }

    movie_genres = _STATE["movie_genres"]
    fans_of = _STATE["fans_of"]
    liked_by_user = _STATE["liked_by_user"]
    pool_size = max(POOL, REC_K * 15)

    p5 = []
    for i, cid in enumerate(assigned):
        seen = _STATE["seed_sets"][i]
        seeds = _STATE["seed_lists"][i]
        seed_g = {GENRES[j] for j in range(len(GENRES)) if masks[i, j] > 0}

        # Liste cluster (re-score genre + popularité)
        entries = []
        for mid, sc in ranked_by_cluster.get(int(cid), [])[:pool_size]:
            if mid in seen:
                continue
            boost = (
                genre_boost
                if (genre_boost > 1.0 and seed_g & set(movie_genres.get(mid, [])))
                else 1.0
            )
            entries.append((mid, sc * boost, pop_norm.get(mid, 0.0)))

        if not entries:
            cluster_list: list[int] = []
        elif pop_blend <= 0.0:
            entries.sort(key=lambda x: (-x[1], x[0]))
            cluster_list = [m for m, _, _ in entries[:pool_size]]
        else:
            max_sc = max(sc for _, sc, _ in entries) or 1.0
            blended = [
                (mid, (1.0 - pop_blend) * (sc / max_sc) + pop_blend * pop)
                for mid, sc, pop in entries
            ]
            blended.sort(key=lambda x: (-x[1], x[0]))
            cluster_list = [m for m, _ in blended[:pool_size]]

        if len(cluster_list) < min_group:
            for mid, _ in pop_ranked:
                if mid not in seen and mid not in cluster_list:
                    cluster_list.append(mid)
                if len(cluster_list) >= pool_size:
                    break

        # Liste UserCF
        cf_list = _user_cf_list(
            seeds,
            seen,
            float(_STATE["ages"][i]),
            _STATE["gender_labels"][i],
            cf_n_user=cf_n_user,
            demo_sex_weight=demo_sex_weight,
            age_scale=age_scale,
            genre_boost=genre_boost,
            movie_genres=movie_genres,
            fans_of=fans_of,
            liked_by_user=liked_by_user,
            pool=pool_size,
        )

        fused = _rrf_fuse(
            [cluster_list, cf_list],
            [w_cluster, w_cf],
            rrf_k=rrf_k,
            k=REC_K,
        )
        hit = sum(1 for m in fused if m in _STATE["relevants"][i]) / float(REC_K)
        p5.append(hit)

    return float(np.mean(p5))


def _suggest_config(trial: optuna.Trial) -> dict:
    return {
        "MIN_SUPPORT": trial.suggest_int("MIN_SUPPORT", 8, 45),
        "N_CLUSTERS": trial.suggest_int("N_CLUSTERS", 4, 24),
        "MIN_GROUP_CANDIDATES": trial.suggest_int("MIN_GROUP_CANDIDATES", 1, 12),
        "DEMO_WEIGHT": trial.suggest_float("DEMO_WEIGHT", 0.0, 2.5, step=0.1),
        "SEED_PROXY": trial.suggest_float("SEED_PROXY", 3.0, 5.0, step=0.25),
        "RATING_POWER": trial.suggest_float("RATING_POWER", 0.5, 2.5, step=0.25),
        "GENRE_BOOST": trial.suggest_float("GENRE_BOOST", 1.0, 1.6, step=0.05),
        "POP_BLEND": trial.suggest_float("POP_BLEND", 0.0, 0.5, step=0.05),
        "SCORE_MODE": trial.suggest_categorical("SCORE_MODE", ["sum", "mean"]),
        "CF_N_USER": trial.suggest_int("CF_N_USER", 40, 350, step=10),
        "DEMO_SEX_WEIGHT": trial.suggest_float("DEMO_SEX_WEIGHT", 0.05, 0.6, step=0.05),
        "AGE_SCALE": trial.suggest_int("AGE_SCALE", 20, 95, step=5),
        "W_CLUSTER": trial.suggest_float("W_CLUSTER", 0.05, 1.0),
        "W_CF": trial.suggest_float("W_CF", 0.05, 1.0),
        "RRF_K": trial.suggest_int("RRF_K", 1, 60),
    }


def main(n_trials: int = N_TRIALS_DEFAULT, out_path: Path | None = None) -> dict:
    print("=== Chargement des données (MODEL_CLUSTER — groupes + UserCF) ===")
    df = load_dataframe()
    df_dict = df.to_dict(orient="list")

    preview = _prepare(df)
    print(
        f"Cas d'évaluation : {preview['n_cases']} | "
        f"profils users : {preview['n_users_profiles']} | "
        f"TRAIN_FRAC={FIXED_TRAIN_FRAC} | aimé ≥ {FIXED_LIKED_THRESHOLD}"
    )
    print(
        f"=== Optuna MODEL_CLUSTER ({n_trials} essais) — "
        f"P@5 fusion cluster⊕UserCF ==="
    )

    sampler = optuna.samplers.TPESampler(seed=FIXED_RANDOM_STATE, n_startup_trials=40)
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
            configs = [_suggest_config(t) for t in trials]
            results = list(executor.map(_eval_trial_config, configs))
            for trial, score in zip(trials, results):
                study.tell(trial, score)

            completed = min(i + max_workers, n_trials)
            batch = np.asarray(results, dtype=float)
            print(
                f"[{completed}/{n_trials}] "
                f"batch P@5 min={batch.min():.4f} mean={batch.mean():.4f} "
                f"max={batch.max():.4f} | meilleur={study.best_value:.4f}"
            )

    bt = study.best_trial
    best = dict(study.best_params)
    # Poids normalisés (lisibles dans le JSON)
    ws, wc = best["W_CLUSTER"], best["W_CF"]
    s = ws + wc
    best["W_CLUSTER_NORM"] = ws / s
    best["W_CF_NORM"] = wc / s
    best["LIKED_THRESHOLD"] = FIXED_LIKED_THRESHOLD
    best["MAX_SEEDS"] = FIXED_MAX_SEEDS
    best["TRAIN_FRAC"] = FIXED_TRAIN_FRAC
    best["RANDOM_STATE"] = FIXED_RANDOM_STATE
    best["metric"] = "P@5_product_cluster_usercf_rrf"
    best["best_P@5"] = study.best_value
    best["n_trials"] = n_trials
    best["best_trial_number"] = bt.number

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
        description="Optuna MODEL_CLUSTER (groupes de goûts + fusion UserCF)."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS_DEFAULT,
        help=f"Nombre d'essais Optuna (défaut {N_TRIALS_DEFAULT}).",
    )
    args = parser.parse_args()
    main(n_trials=args.n_trials)
