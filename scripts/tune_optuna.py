#!/usr/bin/env python3
"""Fine-tune MODEL_CF (User-CF, Film-CF et Hybride) avec Optuna.

Métrique : Precision@5 en mode produit avec exclusion stricte des films vus.
Ajout d'hyperparamètres avancés (RRF_K, SUBLINEAR_TF, RATING_POWER, MIN_GENRE_MATCH, MAX_SEEDS).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Protocole d'évaluation fixe
FIXED_TRAIN_FRAC = 0.65
RANDOM_STATE = 42
N_TRIALS = 300

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

_DF: pd.DataFrame | None = None


def find_data_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    for p in [Path("dataset"), Path("../dataset"), root / "dataset"]:
        if (p / "u.data.csv").exists():
            return p
    raise FileNotFoundError("Répertoire dataset/ introuvable.")


def _init_worker(df_dict: dict):
    global _DF
    _DF = pd.DataFrame(df_dict)


def _eval_trial_config(cfg: dict) -> dict[str, float]:
    """Évalue la synergie User-CF, Film-CF et Hybride sur P@5."""
    global _DF
    assert _DF is not None

    min_support = int(cfg["MIN_SUPPORT"])
    cf_n_user = int(cfg["CF_N_USER"])
    cf_n_film = int(cfg["CF_N_FILM"])
    hybrid_alpha = float(cfg["HYBRID_ALPHA"])
    hybrid_pool = int(cfg["HYBRID_POOL"])
    demo_sex_weight = float(cfg["DEMO_SEX_WEIGHT"])
    age_scale = float(cfg["AGE_SCALE"])
    genre_boost = float(cfg["GENRE_BOOST"])
    
    # Nouveaux hyperparamètres
    max_seeds = int(cfg["MAX_SEEDS"])
    rrf_k = float(cfg["RRF_K"])
    sublinear_tf = bool(cfg["SUBLINEAR_TF"])
    rating_power = float(cfg["RATING_POWER"])
    min_genre_match = int(cfg["MIN_GENRE_MATCH"])

    # 1. Split Train / Test (0.65)
    train_idx = _DF.sample(
        frac=FIXED_TRAIN_FRAC, random_state=RANDOM_STATE
    ).index
    train = _DF.loc[train_idx].reset_index(drop=True)
    test = _DF.drop(train_idx).reset_index(drop=True)

    # 2. Support minimum
    film_counts = train.groupby("item_id").size()
    eligible_items = set(film_counts[film_counts >= min_support].index)

    user_mean = train.groupby("user_id")["rating"].mean()
    train = train.copy()
    test = test.copy()
    train["user_mean"] = train["user_id"].map(user_mean)
    test["user_mean"] = test["user_id"].map(user_mean)

    liked_train = train.loc[
        (train["rating"] > train["user_mean"])
        & (train["item_id"].isin(eligible_items))
    ]
    liked_test = test.loc[
        (test["rating"] > test["user_mean"])
        & (test["item_id"].isin(eligible_items))
    ]

    cand_A = liked_train.copy()
    cand_A["pop"] = cand_A["item_id"].map(film_counts)

    def _get_top_seeds(df_u):
        sorted_u = df_u.sort_values(["rating", "pop"], ascending=[False, False])
        return list(sorted_u["item_id"].unique()[:max_seeds])

    seeds_per_user = cand_A.groupby("user_id", group_keys=False).apply(
        _get_top_seeds
    )
    user_demo = train.drop_duplicates("user_id").set_index("user_id")[
        ["age", "gender"]
    ]
    relevant = liked_test.groupby("user_id")["item_id"].apply(set)
    liked_lists = liked_train.groupby("user_id")["item_id"].apply(set)

    eval_cases = pd.concat(
        [
            seeds_per_user.rename("query_ids"),
            user_demo,
            relevant.rename("rel"),
            liked_lists.rename("liked_train_ids"),
        ],
        axis=1,
        join="inner",
    ).reset_index()
    eval_cases["rel"] = eval_cases.apply(
        lambda r: r["rel"] - set(r["query_ids"]), axis=1
    )
    eval_cases = eval_cases.loc[
        eval_cases["rel"].map(len) > 0
    ].reset_index(drop=True)

    if len(eval_cases) == 0:
        return {"hybrid_p5": 0.0, "user_p5": 0.0, "film_p5": 0.0}

    # 3. Matrice Film-CF (TF-IDF avec sublinear_tf)
    users_u = sorted(train["user_id"].unique())
    items_u = sorted(
        train[train["item_id"].isin(eligible_items)]["item_id"].unique()
    )
    if not items_u:
        return {"hybrid_p5": 0.0, "user_p5": 0.0, "film_p5": 0.0}

    u_ix = {u: i for i, u in enumerate(users_u)}
    i_ix = {it: i for i, it in enumerate(items_u)}

    train_filtered = train[train["item_id"].isin(eligible_items)]
    mat_ui = csr_matrix(
        (
            train_filtered["rating"].astype(float).values,
            (
                train_filtered["user_id"].map(u_ix).values,
                train_filtered["item_id"].map(i_ix).values,
            ),
        ),
        shape=(len(users_u), len(items_u)),
    )
    mat_iu = mat_ui.T.tocsr()

    tfidf = TfidfTransformer(sublinear_tf=sublinear_tf)
    mat_iu_tfidf = tfidf.fit_transform(mat_iu)

    n_ask_i = min(cf_n_film + 1, len(items_u))
    nn_item = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_item.fit(mat_iu_tfidf)

    d_i, ix_i = nn_item.kneighbors(mat_iu_tfidf, n_neighbors=n_ask_i)
    item_ids_arr = np.array(items_u)

    CF_ITEM = {}
    for i, mid in enumerate(items_u):
        CF_ITEM[mid] = [
            (int(item_ids_arr[j]), 1.0 - float(d))
            for d, j in zip(d_i[i], ix_i[i])
            if int(item_ids_arr[j]) != mid
        ]

    # 4. Structures User-CF
    liked_by_user = (
        liked_train.groupby("user_id")[["item_id", "rating"]]
        .apply(lambda x: list(zip(x["item_id"], x["rating"])))
        .to_dict()
    )

    fans_of = {
        mid: list(
            zip(
                grp["user_id"].to_numpy(),
                grp["rating"].to_numpy(),
                grp["age"].to_numpy(),
                grp["gender"].to_numpy(),
            )
        )
        for mid, grp in liked_train.groupby("item_id")
    }

    movies = _DF.drop_duplicates("item_id").set_index("item_id")
    movie_genres = movies["genres"].to_dict()

    def _get_seen(query_ids, liked_train_ids):
        seeds = set(int(x) for x in query_ids)
        liked = set(liked_train_ids) if liked_train_ids is not None else set()
        return seeds.union(liked), seeds

    # Fonctions de recommandation
    def recommend_film_cf(query_ids, liked_train_ids, k=5):
        seen, seeds = _get_seen(query_ids, liked_train_ids)
        scores = {}
        for qid in seeds:
            for mid, sim in CF_ITEM.get(qid, []):
                if mid in seen:
                    continue
                scores[mid] = scores.get(mid, 0.0) + sim
        return [
            m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        ]

    def recommend_user_cf(query_ids, age, gender, liked_train_ids, k=5):
        seen, seeds = _get_seen(query_ids, liked_train_ids)
        age_val = 30.0 if age is None else float(age)
        gender_val = "M" if gender is None else gender

        neighbors = []
        for qid in seeds:
            for uid, rat, a, g in fans_of.get(qid, []):
                sex = 1.0 if gender_val == g else 0.0
                age_term = 1.0 - min(abs(age_val - float(a)) / age_scale, 1.0)
                demo = demo_sex_weight * sex + (1.0 - demo_sex_weight) * age_term
                neighbors.append((int(uid), demo * (float(rat) ** rating_power) / (5.0 ** rating_power)))

        neighbors.sort(key=lambda x: -x[1])
        neighbors = neighbors[:cf_n_user]

        scores = {}
        for vid, sim in neighbors:
            for mid, rat in liked_by_user.get(vid, []):
                if mid in seen:
                    continue
                scores[mid] = scores.get(mid, 0.0) + float(sim) * (float(rat) ** rating_power)

        return [
            m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        ]

    def recommend_hybrid(query_ids, age, gender, liked_train_ids, k=5):
        seen, seeds = _get_seen(query_ids, liked_train_ids)
        film_list = recommend_film_cf(
            query_ids, liked_train_ids, k=hybrid_pool
        )
        user_list = recommend_user_cf(
            query_ids, age, gender, liked_train_ids, k=hybrid_pool
        )

        scores = {}
        # Application de la formule RRF : 1 / (k_rrf + rank)
        for rank, mid in enumerate(film_list):
            scores[mid] = scores.get(mid, 0.0) + hybrid_alpha / (rrf_k + rank)
        for rank, mid in enumerate(user_list):
            scores[mid] = scores.get(mid, 0.0) + (1.0 - hybrid_alpha) / (rrf_k + rank)

        union_g = set()
        for qid in seeds:
            union_g.update(movie_genres.get(qid, []))
        for mid in list(scores.keys()):
            common_genres = len(union_g & set(movie_genres.get(mid, [])))
            if common_genres >= min_genre_match:
                scores[mid] *= genre_boost

        return [
            m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        ]

    # Évaluation des 3 approches
    hybrid_p5, user_p5, film_p5 = [], [], []
    for _, row in eval_cases.iterrows():
        qids = row["query_ids"]
        age = row["age"]
        gender = row["gender"]
        liked = row["liked_train_ids"]
        rel = row["rel"]

        rec_h = recommend_hybrid(qids, age, gender, liked, k=5)
        rec_u = recommend_user_cf(qids, age, gender, liked, k=5)
        rec_f = recommend_film_cf(qids, liked, k=5)

        hybrid_p5.append(sum(1 for m in rec_h if m in rel) / 5.0)
        user_p5.append(sum(1 for m in rec_u if m in rel) / 5.0)
        film_p5.append(sum(1 for m in rec_f if m in rel) / 5.0)

    return {
        "hybrid_p5": float(np.mean(hybrid_p5)),
        "user_p5": float(np.mean(user_p5)),
        "film_p5": float(np.mean(film_p5)),
    }


def _worker_wrapper(cfg: dict) -> dict[str, float]:
    return _eval_trial_config(cfg)


if __name__ == "__main__":
    print("=== Chargement des données (MODEL_CF) ===")
    data_dir = find_data_dir()

    notes = pd.read_csv(data_dir / "u.data.csv")
    films = pd.read_csv(data_dir / "u.item.csv")
    users = pd.read_csv(data_dir / "u.user.csv").drop(
        columns=["occupation"], errors="ignore"
    )

    df_raw = notes.merge(users, on="user_id", how="left").merge(
        films, left_on="item_id", right_on="movie_id", how="left"
    )
    df = df_raw.loc[df_raw["unknown"] != 1].drop(columns=["unknown"]).copy()
    df["genres"] = df[GENRES].apply(
        lambda row: [g for g in GENRES if row[g] == 1], axis=1
    )
    df_dict = df.to_dict(orient="list")

    print(
        f"=== Lancement de l'optimisation Optuna ({N_TRIALS} essais) — "
        f"TRAIN_FRAC fixé={FIXED_TRAIN_FRAC} ==="
    )

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=50)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    max_workers = os.cpu_count() or 4
    print(f"Parallélisation sur {max_workers} cœurs CPU")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(df_dict,),
    ) as executor:

        for i in range(0, N_TRIALS, max_workers):
            batch_size = min(max_workers, N_TRIALS - i)
            trials = [study.ask() for _ in range(batch_size)]

            configs = []
            for t in trials:
                cfg = {
                    "MIN_SUPPORT": t.suggest_int("MIN_SUPPORT", 5, 35),
                    "CF_N_USER": t.suggest_int("CF_N_USER", 30, 300, step=10),
                    "CF_N_FILM": t.suggest_int("CF_N_FILM", 20, 180, step=10),
                    "HYBRID_ALPHA": t.suggest_float("HYBRID_ALPHA", 0.10, 0.90, step=0.05),
                    "HYBRID_POOL": t.suggest_int("HYBRID_POOL", 50, 300, step=10),
                    "DEMO_SEX_WEIGHT": t.suggest_float("DEMO_SEX_WEIGHT", 0.05, 0.60, step=0.05),
                    "AGE_SCALE": t.suggest_int("AGE_SCALE", 20, 90, step=5),
                    "GENRE_BOOST": t.suggest_float("GENRE_BOOST", 1.05, 1.40, step=0.05),
                    # Nouveaux hyperparamètres
                    "MAX_SEEDS": t.suggest_int("MAX_SEEDS", 2, 7),
                    "RRF_K": t.suggest_int("RRF_K", 1, 60),
                    "SUBLINEAR_TF": t.suggest_categorical("SUBLINEAR_TF", [True, False]),
                    "RATING_POWER": t.suggest_float("RATING_POWER", 0.5, 2.0, step=0.25),
                    "MIN_GENRE_MATCH": t.suggest_int("MIN_GENRE_MATCH", 1, 3),
                }
                configs.append(cfg)

            metrics_list = list(executor.map(_worker_wrapper, configs))

            for trial, metrics in zip(trials, metrics_list):
                trial.set_user_attr("user_p5", metrics["user_p5"])
                trial.set_user_attr("film_p5", metrics["film_p5"])
                trial.set_user_attr("hybrid_p5", metrics["hybrid_p5"])
                
                study.tell(trial, metrics["hybrid_p5"])

            completed = min(i + max_workers, N_TRIALS)
            best_t = study.best_trial
            print(
                f"[{completed}/{N_TRIALS}] Top Hybride P@5 : {study.best_value:.4f} "
                f"(User-CF: {best_t.user_attrs.get('user_p5', 0):.4f} | "
                f"Film-CF: {best_t.user_attrs.get('film_p5', 0):.4f})"
            )

    bt = study.best_trial
    best = dict(study.best_params)
    best["TRAIN_FRAC"] = FIXED_TRAIN_FRAC
    best["best_hybrid_P@5"] = bt.value
    best["associated_user_P@5"] = bt.user_attrs.get("user_p5")
    best["associated_film_P@5"] = bt.user_attrs.get("film_p5")

    print("\n=== MEILLEUR CONFIGURATION ET PERFORMANCES ===")
    print(f"P@5 Hybride Max  : {best['best_hybrid_P@5']:.4f}")
    print(f"P@5 User-CF seul : {best['associated_user_P@5']:.4f}")
    print(f"P@5 Film-CF seul : {best['associated_film_P@5']:.4f}")
    print(json.dumps(best, indent=2))

    out_path = Path("results/best_hyperparams_cf_optuna.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"\nRésultats exportés dans : {out_path}")