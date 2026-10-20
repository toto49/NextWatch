#!/usr/bin/env python3
"""Fine-tune NextWatch hyperparamètres (Precision@5 mode produit) - Version Multi-cœurs.

Usage:
  python scripts/tune_hyperparams.py
  python scripts/tune_hyperparams.py --n-iter 200
  python scripts/tune_hyperparams.py --max-cases 150 --n-iter 30
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# --- Constantes fixes ---
LIKED_THRESHOLD = 4
RANDOM_STATE = 42

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

# --- Espaces de recherche (élargis) ---
SEARCH_SPACE = {
    "TRAIN_FRAC": [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
    "MIN_SUPPORT": [3, 5, 8, 10, 15, 20, 30, 50],
    "CF_N_NEIGHBORS": [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200],
    "HYBRID_ALPHA": [round(x * 0.1, 1) for x in range(0, 11)],  # 0.0 .. 1.0
    "HYBRID_POOL": [20, 40, 60, 80, 100, 150, 200, 300],
    "DEMO_SEX_WEIGHT": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "AGE_SCALE": [20, 30, 40, 50, 60, 80, 100],
}

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def find_data_dir() -> Path:
    for p in [ROOT / "dataset", Path("dataset"), Path("../dataset")]:
        if (p / "u.data.csv").exists():
            return p.resolve()
    raise FileNotFoundError("dataset/ introuvable (u.data.csv)")


def load_df(data_dir: Path) -> pd.DataFrame:
    notes = pd.read_csv(data_dir / "u.data.csv")
    films = pd.read_csv(data_dir / "u.item.csv")
    users = pd.read_csv(
        data_dir / "u.user.csv", dtype={"zip_code": "string"}
    ).drop(columns=["occupation"])
    df_raw = notes.merge(users, on="user_id", how="left").merge(
        films, left_on="item_id", right_on="movie_id", how="left"
    )
    df = df_raw.loc[df_raw["unknown"] != 1].drop(columns=["unknown"]).copy()
    df["genres"] = df[GENRES].apply(
        lambda row: [g for g in GENRES if row[g] == 1], axis=1
    )
    return df


def precision_at_k(recommended, relevant, k: int) -> float:
    top = recommended[:k]
    if not top:
        return 0.0
    return sum(1 for m in top if m in relevant) / k


def sample_configs(
    mode: str, n_iter: int, rng: np.random.Generator
) -> list[dict]:
    keys = list(SEARCH_SPACE.keys())
    if mode == "grid":
        combos = list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
        return [dict(zip(keys, c)) for c in combos]

    configs = []
    seen = set()
    attempts = 0
    max_attempts = n_iter * 20
    while len(configs) < n_iter and attempts < max_attempts:
        attempts += 1
        cfg = {
            k: SEARCH_SPACE[k][int(rng.integers(0, len(SEARCH_SPACE[k])))]
            for k in keys
        }
        key = tuple(cfg[k] for k in keys)
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs


class PipelineCache:
    """Mémoïsation localisée par processus fils."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.movies = df.drop_duplicates("item_id").set_index("item_id")
        self.movie_genres = self.movies["genres"].to_dict()
        self.all_movie_ids = self.movies.index.to_numpy()
        self._split: dict = {}
        self._stats: dict = {}
        self._cf: dict = {}

    def get_split(self, train_frac: float):
        key = float(train_frac)
        if key not in self._split:
            train = self.df.sample(frac=key, random_state=RANDOM_STATE)
            test = self.df.drop(train.index)
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
            liked_train = train.loc[train["rating"] >= LIKED_THRESHOLD]
            liked_test = test.loc[test["rating"] >= LIKED_THRESHOLD]
            tc = train.groupby("item_id").size()
            cand = liked_train.copy()
            cand["pop"] = cand["item_id"].map(tc)
            idx = (
                cand.sort_values(["rating", "pop"], ascending=[False, False])
                .groupby("user_id", sort=False)
                .first()
                .reset_index()
            )
            idx = idx.rename(
                columns={"item_id": "query_id", "rating": "query_rating"}
            )[["user_id", "query_id", "query_rating", "age", "gender"]]
            liked_lists = liked_train.groupby("user_id")["item_id"].apply(set)
            rel = (
                liked_test.groupby("user_id")["item_id"]
                .apply(set)
                .rename("relevant_ids")
                .reset_index()
            )
            ev = idx.merge(rel, on="user_id")
            ev["liked_train_ids"] = ev["user_id"].map(liked_lists)
            ev["relevant_ids"] = ev.apply(
                lambda r: r["relevant_ids"] - {r["query_id"]}, axis=1
            )
            ev = ev.loc[ev["relevant_ids"].map(len) > 0].reset_index(drop=True)
            self._split[key] = (train, test, ev)
        return self._split[key]

    def get_stats(self, train_frac: float, min_support: int):
        key = (float(train_frac), int(min_support))
        if key not in self._stats:
            train, _, _ = self.get_split(train_frac)
            film_stats = (
                train.groupby("item_id")["rating"]
                .agg(n_ratings="count", mean_rating="mean")
                .reset_index()
            )
            eligible = film_stats.loc[
                film_stats["n_ratings"] >= min_support
            ].set_index("item_id")
            self._stats[key] = eligible["mean_rating"]
        return self._stats[key]

    def get_cf(self, train_frac: float, n_neighbors: int):
        key = (float(train_frac), int(n_neighbors))
        if key not in self._cf:
            train, _, _ = self.get_split(train_frac)
            users_u = sorted(train["user_id"].unique())
            items_u = sorted(train["item_id"].unique())
            u_ix = {u: i for i, u in enumerate(users_u)}
            i_ix = {it: i for i, it in enumerate(items_u)}
            mat_ui = csr_matrix(
                (
                    train["rating"].astype(float).values,
                    (
                        train["user_id"].map(u_ix).values,
                        train["item_id"].map(i_ix).values,
                    ),
                ),
                shape=(len(users_u), len(items_u)),
            )
            mat_iu = mat_ui.T.tocsr()
            nn_item = NearestNeighbors(metric="cosine", algorithm="brute")
            nn_item.fit(mat_iu)
            item_ids_arr = np.array(items_u)
            n_ask = min(n_neighbors + 1, len(items_u))
            d_i, ix_i = nn_item.kneighbors(mat_iu, n_neighbors=n_ask)
            cf_item = {}
            for i, mid in enumerate(items_u):
                cf_item[mid] = [
                    (int(item_ids_arr[j]), 1.0 - float(d))
                    for d, j in zip(d_i[i], ix_i[i])
                    if int(item_ids_arr[j]) != mid
                ]
            liked_by_user = (
                train.loc[train["rating"] >= LIKED_THRESHOLD]
                .groupby("user_id")[["item_id", "rating"]]
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
                for mid, grp in train.loc[
                    train["rating"] >= LIKED_THRESHOLD
                ].groupby("item_id")
            }
            self._cf[key] = {
                "train": train,
                "CF_ITEM": cf_item,
                "liked_by_user": liked_by_user,
                "fans_of": fans_of,
            }
        return self._cf[key]


def make_recommenders(cache: PipelineCache, cfg: dict):
    train_frac = float(cfg["TRAIN_FRAC"])
    min_support = int(cfg["MIN_SUPPORT"])
    n_neighbors = int(cfg["CF_N_NEIGHBORS"])
    hybrid_alpha = float(cfg["HYBRID_ALPHA"])
    hybrid_pool = int(cfg["HYBRID_POOL"])
    demo_sex_w = float(cfg["DEMO_SEX_WEIGHT"])
    age_scale = float(cfg["AGE_SCALE"])

    train, _, eval_cases = cache.get_split(train_frac)
    mean_rating = cache.get_stats(train_frac, min_support)
    cf = cache.get_cf(train_frac, n_neighbors)
    CF_ITEM = cf["CF_ITEM"]
    liked_by_user = cf["liked_by_user"]
    fans_of = cf["fans_of"]
    movie_genres = cache.movie_genres
    all_movie_ids = cache.all_movie_ids

    def _seen(query_id, liked_train_ids):
        s = set(liked_train_ids) if liked_train_ids is not None else set()
        s.add(int(query_id))
        return s

    def _demo_sim(age_a, gender_a, age_b, gender_b):
        sex = 1.0 if gender_a == gender_b else 0.0
        age_term = 1.0 - min(abs(float(age_a) - float(age_b)) / age_scale, 1.0)
        return demo_sex_w * sex + (1.0 - demo_sex_w) * age_term

    def recommend_dummy(
        query_id,
        k=10,
        age=None,
        gender=None,
        liked_train_ids=None,
        user_id=None,
    ):
        seen = _seen(query_id, liked_train_ids)
        pool = np.array([m for m in all_movie_ids if m not in seen])
        local = np.random.default_rng(RANDOM_STATE + int(query_id))
        if len(pool) == 0:
            return []
        return local.choice(
            pool, size=min(k, len(pool)), replace=False
        ).tolist()

    def recommend_baseline(
        query_id,
        k=10,
        age=None,
        gender=None,
        liked_train_ids=None,
        user_id=None,
    ):
        seen = _seen(query_id, liked_train_ids)
        gq = set(movie_genres.get(query_id, []))
        if not gq:
            return []
        scores = [
            (mid, float(m))
            for mid, m in mean_rating.items()
            if mid not in seen and gq & set(movie_genres.get(mid, []))
        ]
        scores.sort(key=lambda x: (-x[1], x[0]))
        return [m for m, _ in scores[:k]]

    def recommend_user_cf(
        query_id,
        k=10,
        age=None,
        gender=None,
        liked_train_ids=None,
        user_id=None,
    ):
        seen = _seen(query_id, liked_train_ids)
        age = 30.0 if age is None else float(age)
        gender = "M" if gender is None else gender
        neighbors = []
        for uid, rat, a, g in fans_of.get(int(query_id), []):
            demo = _demo_sim(age, gender, a, g)
            neighbors.append((int(uid), demo * float(rat) / 5.0))
        neighbors.sort(key=lambda x: -x[1])
        neighbors = neighbors[:n_neighbors]
        scores = {}
        for vid, sim in neighbors:
            for mid, rat in liked_by_user.get(vid, []):
                if mid in seen:
                    continue
                scores[mid] = scores.get(mid, 0.0) + float(sim) * float(rat)
        return [
            m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        ]

    def recommend_film_cf(
        query_id,
        k=10,
        age=None,
        gender=None,
        liked_train_ids=None,
        user_id=None,
    ):
        seen = _seen(query_id, liked_train_ids)
        seeds = (
            set(liked_train_ids) if liked_train_ids is not None else {query_id}
        )
        seeds.add(int(query_id))
        scores = {}
        for aid in seeds:
            for mid, sim in CF_ITEM.get(aid, []):
                if mid in seen:
                    continue
                scores[mid] = (
                    scores.get(mid, 0.0) + float(LIKED_THRESHOLD) * sim
                )
        return [
            m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        ]

    def recommend_hybrid(
        query_id,
        k=10,
        age=None,
        gender=None,
        liked_train_ids=None,
        user_id=None,
    ):
        pool = max(hybrid_pool, k)
        film_list = recommend_film_cf(
            query_id,
            k=pool,
            age=age,
            gender=gender,
            liked_train_ids=liked_train_ids,
            user_id=user_id,
        )
        user_list = recommend_user_cf(
            query_id,
            k=pool,
            age=age,
            gender=gender,
            liked_train_ids=liked_train_ids,
            user_id=user_id,
        )
        scores = {}
        for rank, mid in enumerate(film_list):
            scores[mid] = scores.get(mid, 0.0) + hybrid_alpha / (1.0 + rank)
        for rank, mid in enumerate(user_list):
            scores[mid] = scores.get(mid, 0.0) + (1.0 - hybrid_alpha) / (
                1.0 + rank
            )
        return [
            m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        ]

    methods = {
        "dummy_random": recommend_dummy,
        "baseline_overlap_mean": recommend_baseline,
        "model_user_user_cf": recommend_user_cf,
        "model_film_film_cf": recommend_film_cf,
        "model_hybrid_film_user": recommend_hybrid,
    }
    return methods, eval_cases


def evaluate_product(recommend_fn, cases, max_k=10):
    rows = []
    for _, row in cases.iterrows():
        rec = recommend_fn(
            int(row["query_id"]),
            k=max_k,
            age=row["age"],
            gender=row["gender"],
            liked_train_ids=row["liked_train_ids"],
            user_id=None,
        )
        relevant = row["relevant_ids"]
        rows.append(
            {
                "P@5": precision_at_k(rec, relevant, 5),
                "P@10": precision_at_k(rec, relevant, 10),
            }
        )
    return pd.DataFrame(rows).mean()


def _eval_single_config(args_tuple):
    """Fonction isolée exécutée en parallèle par processus fils."""
    cfg, df_dict, max_cases = args_tuple
    df = pd.DataFrame(df_dict)
    cache = PipelineCache(df)

    methods, eval_cases = make_recommenders(cache, cfg)
    if max_cases is not None:
        eval_cases = eval_cases.head(max_cases)

    n_cases = len(eval_cases)
    results = []
    for method, fn in methods.items():
        metrics = evaluate_product(fn, eval_cases)
        results.append(
            {
                **cfg,
                "method": method,
                "n_cases": n_cases,
                "P@5": float(metrics["P@5"]),
                "P@10": float(metrics["P@10"]),
            }
        )

    # Nettoyage mémoire explicite du sous-processus
    del cache, methods, eval_cases
    gc.collect()

    return results


def run_tune(args):
    data_dir = find_data_dir()
    print(f"Dataset : {data_dir}")
    df = load_df(data_dir)
    print(
        f"Notes : {len(df):,} | users={df['user_id'].nunique()} |"
        f" films={df['item_id'].nunique()}"
    )

    rng = np.random.default_rng(RANDOM_STATE)
    configs = sample_configs(args.mode, args.n_iter, rng)

    # Détection automatique des cœurs CPU disponibles (max 10 pour préserver le système)
    max_workers = min(os.cpu_count() or 4, 10)
    print(
        f"Mode={args.mode} | configs={len(configs)} | Parallélisation sur"
        f" {max_workers} cœurs"
    )

    df_dict = df.to_dict(orient="list")
    tasks = [(cfg, df_dict, args.max_cases) for cfg in configs]

    rows = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_eval_single_config, task) for task in tasks
        ]

        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            rows.extend(res)
            if i % 10 == 0 or i == 1 or i == len(configs):
                elapsed = time.time() - t0
                print(
                    f"[{i}/{len(configs)}] accomplis en {elapsed:.1f}s"
                    f" ({elapsed/i:.2f}s/config)"
                )

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "tune_hyperparams.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nCSV → {csv_path}")

    # Meilleur par méthode (P@5)
    best_by_method = {}
    for method, sub in results.groupby("method"):
        best = sub.sort_values(["P@5", "P@10"], ascending=False).iloc[0]
        params = {}
        for k in SEARCH_SPACE:
            v = best[k]
            if k in (
                "TRAIN_FRAC",
                "HYBRID_ALPH",
                "DEMO_SEX_WEIGHT",
                "AGE_SCALE",
            ):
                params[k] = float(v)
            else:
                params[k] = int(v)
        best_by_method[method] = {
            "P@5": float(best["P@5"]),
            "P@10": float(best["P@10"]),
            "params": params,
        }

    # Meilleur global = hybride
    hybrid_best = best_by_method.get("model_hybrid_film_user", {})
    payload = {
        "metric": "P@5_product",
        "liked_threshold": LIKED_THRESHOLD,
        "random_state": RANDOM_STATE,
        "n_configs": len(configs),
        "mode": args.mode,
        "best_by_method": best_by_method,
        "recommended_for_notebook": hybrid_best.get("params", {}),
    }
    json_path = RESULTS_DIR / "best_hyperparams.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON → {json_path}")

    print("\n=== Meilleur P@5 produit par méthode ===")
    summary = (
        results.sort_values(
            ["method", "P@5", "P@10"], ascending=[True, False, False]
        )
        .groupby("method", as_index=False)
        .first()[["method", "P@5", "P@10"] + list(SEARCH_SPACE.keys())]
    )
    print(summary.round(4).to_string(index=False))
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune hyperparamètres NextWatch"
    )
    p.add_argument("--mode", choices=["random", "grid"], default="random")
    p.add_argument(
        "--n-iter",
        type=int,
        default=200,
        help="Tirages random (ignoré si grid)",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limiter le nombre de cas d'eval (smoke test)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "grid":
        n_full = int(np.prod([len(v) for v in SEARCH_SPACE.values()]))
        print(f"ATTENTION: grille complète = {n_full:,} configs")
        if n_full > 5000 and args.max_cases is None:
            print("Conseil: utilise --max-cases ou --mode random (défaut).")
    run_tune(args)


if __name__ == "__main__":
    main()