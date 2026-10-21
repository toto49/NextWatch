#!/usr/bin/env python3
"""Fine-tune MODEL_CLUSTER (experts sexe/âge/catégories + fusion soft) avec Optuna.

Métrique : Precision@5 en mode produit (âge + sexe + 1–3 films).
TRAIN_FRAC est fixé (même protocole que le notebook) pour ne pas biaiser le score.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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

# Aligné notebook MODEL_CLUSTER — ne pas tuner (sinon P@5 artificiel)
FIXED_TRAIN_FRAC = 0.7
FIXED_RANDOM_STATE = 42
FIXED_LIKED_THRESHOLD = 4
FIXED_MAX_SEEDS = 3

_DF: pd.DataFrame | None = None


def find_data_dir() -> Path:
  root = Path(__file__).resolve().parents[1]
  for p in [Path("dataset"), Path("../dataset"), root / "dataset"]:
    if (p / "u.data.csv").exists():
      return p
  raise FileNotFoundError("Repertoire dataset/ introuvable.")


def _init_worker(df_dict: dict):
  global _DF
  _DF = pd.DataFrame(df_dict)


def _eval_trial_config(cfg: dict) -> float:
  """Évalue la fusion cluster en Precision@5 produit."""
  global _DF
  assert _DF is not None

  min_support = int(cfg["MIN_SUPPORT"])
  cf_n_user = int(cfg["CF_N_USER"])
  age_band = float(cfg["AGE_BAND"])
  min_seg_users = int(cfg["MIN_SEG_USERS"])
  n_clusters = int(cfg["N_CLUSTERS"])
  expert_pool = int(cfg["EXPERT_POOL"])
  w_sex = float(cfg["W_SEX"])
  w_age = float(cfg["W_AGE"])
  w_cat = float(cfg["W_CAT"])
  genre_boost = float(cfg["GENRE_BOOST"])

  train_idx = _DF.sample(
      frac=FIXED_TRAIN_FRAC, random_state=FIXED_RANDOM_STATE
  ).index
  train = _DF.loc[train_idx].reset_index(drop=True)
  test = _DF.drop(train_idx).reset_index(drop=True)

  film_stats = (
      train.groupby("item_id")["rating"]
      .agg(n_ratings="count", mean_rating="mean")
      .reset_index()
  )
  eligible = film_stats.loc[film_stats["n_ratings"] >= min_support].set_index(
      "item_id"
  )
  if len(eligible) == 0:
    return 0.0
  mean_rating = eligible["mean_rating"]

  liked_train = train.loc[train["rating"] >= FIXED_LIKED_THRESHOLD]
  liked_test = test.loc[test["rating"] >= FIXED_LIKED_THRESHOLD]
  train_counts = train.groupby("item_id").size()

  cand_A = liked_train.copy()
  cand_A["pop"] = cand_A["item_id"].map(train_counts)

  def _get_top_seeds(df_u):
    sorted_u = df_u.sort_values(["rating", "pop"], ascending=[False, False])
    return list(sorted_u["item_id"].unique()[:FIXED_MAX_SEEDS])

  seeds_per_user = cand_A.groupby("user_id", group_keys=False).apply(
      _get_top_seeds
  )
  user_demo = train.drop_duplicates("user_id").set_index("user_id")[
      ["age", "gender"]
  ]
  relevant = liked_test.groupby("user_id")["item_id"].apply(set)

  eval_cases = pd.concat(
      [seeds_per_user.rename("query_ids"), user_demo, relevant.rename("rel")],
      axis=1,
      join="inner",
  ).reset_index()
  eval_cases["rel"] = eval_cases.apply(
      lambda r: r["rel"] - set(r["query_ids"]), axis=1
  )
  eval_cases = eval_cases.loc[eval_cases["rel"].map(len) > 0].reset_index(
      drop=True
  )
  if len(eval_cases) == 0:
    return 0.0

  movies = _DF.drop_duplicates("item_id").set_index("item_id")
  movie_genres = movies["genres"].to_dict()
  film_genre_X = movies[GENRES].astype(float)
  all_movie_ids = film_genre_X.index.to_numpy()

  liked_by_user = (
      train.loc[train["rating"] >= FIXED_LIKED_THRESHOLD]
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
          train["rating"] >= FIXED_LIKED_THRESHOLD
      ].groupby("item_id")
  }

  train_movie_ids = train["item_id"].unique()
  X_km = film_genre_X.loc[film_genre_X.index.isin(train_movie_ids)]
  if len(X_km) < n_clusters:
    return 0.0
  kmeans = KMeans(
      n_clusters=n_clusters, random_state=FIXED_RANDOM_STATE, n_init=10
  )
  cluster_labels = kmeans.fit_predict(X_km.values)
  film_cluster = {
      int(mid): int(lab) for mid, lab in zip(X_km.index, cluster_labels)
  }
  missing = [m for m in all_movie_ids if int(m) not in film_cluster]
  if missing:
    pred = kmeans.predict(film_genre_X.loc[missing].values)
    for mid, lab in zip(missing, pred):
      film_cluster[int(mid)] = int(lab)

  def _seen(query_ids):
    return set(int(x) for x in query_ids)

  def _user_cf_from_neighbors(neighbors, seen, k):
    neighbors = neighbors[:cf_n_user]
    scores = {}
    for vid, sim in neighbors:
      for mid, rat in liked_by_user.get(vid, []):
        if mid in seen:
          continue
        scores[mid] = scores.get(mid, 0.0) + float(sim) * float(rat)
    return [
        m
        for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
    ]

  def _collect_fans(seeds, predicate):
    neighbors = []
    for qid in seeds:
      for uid, rat, a, g in fans_of.get(qid, []):
        w = predicate(int(uid), float(rat), float(a), g)
        if w is None:
          continue
        neighbors.append((int(uid), w))
    neighbors.sort(key=lambda x: -x[1])
    return neighbors

  def recommend_user_cf_global(query_ids, k, age, gender):
    seen = _seen(query_ids)
    seeds = seen

    def pred(uid, rat, a, g):
      sex = 1.0 if g == gender else 0.0
      age_term = 1.0 - min(abs(float(age) - a) / 95.0, 1.0)
      return (0.2 * sex + 0.8 * age_term) * rat / 5.0

    return _user_cf_from_neighbors(_collect_fans(seeds, pred), seen, k)

  def recommend_baseline(query_ids, k):
    seen = _seen(query_ids)
    seeds = seen
    union_g = set()
    for qid in seeds:
      union_g.update(movie_genres.get(qid, []))
    if not union_g:
      return []
    scores = [
        (mid, float(m))
        for mid, m in mean_rating.items()
        if mid not in seen and union_g & set(movie_genres.get(mid, []))
    ]
    scores.sort(key=lambda x: (-x[1], x[0]))
    return [m for m, _ in scores[:k]]

  def recommend_expert_sex(query_ids, k, age, gender):
    seen = _seen(query_ids)
    seeds = seen

    def pred(uid, rat, a, g):
      if g != gender:
        return None
      return rat / 5.0

    neighbors = _collect_fans(seeds, pred)
    if len(neighbors) < min_seg_users:
      return recommend_user_cf_global(query_ids, k, age, gender)
    return _user_cf_from_neighbors(neighbors, seen, k)

  def recommend_expert_age(query_ids, k, age, gender):
    seen = _seen(query_ids)
    seeds = seen

    def pred(uid, rat, a, g):
      if abs(a - float(age)) > age_band:
        return None
      return (1.0 - abs(a - float(age)) / age_band) * rat / 5.0

    neighbors = _collect_fans(seeds, pred)
    if len(neighbors) < min_seg_users:
      return recommend_user_cf_global(query_ids, k, age, gender)
    return _user_cf_from_neighbors(neighbors, seen, k)

  def recommend_expert_category(query_ids, k):
    seen = _seen(query_ids)
    seeds = seen
    seed_clusters = {
        film_cluster.get(qid) for qid in seeds if qid in film_cluster
    }
    seed_clusters.discard(None)
    if not seed_clusters:
      return recommend_baseline(query_ids, k)

    seed_vecs = film_genre_X.loc[
        [s for s in seeds if s in film_genre_X.index]
    ]
    if len(seed_vecs) == 0:
      return []
    q = seed_vecs.mean(axis=0).values.reshape(1, -1)
    cands = [
        mid
        for mid in eligible.index
        if mid not in seen and film_cluster.get(int(mid)) in seed_clusters
    ]
    if not cands:
      return recommend_baseline(query_ids, k)

    C = film_genre_X.loc[cands].values
    sims = cosine_similarity(q, C).ravel()
    means = np.array([float(mean_rating.get(mid, 0.0)) for mid in cands])
    scores = 0.7 * sims + 0.3 * (means / 5.0)
    order = np.argsort(-scores)
    return [int(cands[i]) for i in order[:k]]

  def recommend_fusion(query_ids, age, gender):
    pool = max(expert_pool, 5)
    sex_list = recommend_expert_sex(query_ids, pool, age, gender)
    age_list = recommend_expert_age(query_ids, pool, age, gender)
    cat_list = recommend_expert_category(query_ids, pool)

    scores = {}
    for rank, mid in enumerate(sex_list):
      scores[mid] = scores.get(mid, 0.0) + w_sex / (1.0 + rank)
    for rank, mid in enumerate(age_list):
      scores[mid] = scores.get(mid, 0.0) + w_age / (1.0 + rank)
    for rank, mid in enumerate(cat_list):
      scores[mid] = scores.get(mid, 0.0) + w_cat / (1.0 + rank)

    seeds = _seen(query_ids)
    union_g = set()
    for qid in seeds:
      union_g.update(movie_genres.get(qid, []))
    for mid in list(scores.keys()):
      if union_g & set(movie_genres.get(mid, [])):
        scores[mid] *= genre_boost

    return [
        m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]

  p5_scores = []
  for _, row in eval_cases.iterrows():
    rec = recommend_fusion(row["query_ids"], row["age"], row["gender"])
    hit = sum(1 for m in rec if m in row["rel"]) / 5.0
    p5_scores.append(hit)

  return float(np.mean(p5_scores))


if __name__ == "__main__":
  N_TRIALS = 300

  print("=== Chargement des données (MODEL_CLUSTER) ===")
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
      f"=== Optuna MODEL_CLUSTER ({N_TRIALS} essais) — "
      f"TRAIN_FRAC fixé={FIXED_TRAIN_FRAC} ==="
  )

  sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=30)
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
        w_sex = t.suggest_float("W_SEX", 0.05, 0.85)
        w_age = t.suggest_float("W_AGE", 0.05, max(0.06, 0.95 - w_sex))
        w_cat = max(0.0, 1.0 - w_sex - w_age)
        # renormalise au cas où (float)
        s = w_sex + w_age + w_cat
        w_sex, w_age, w_cat = w_sex / s, w_age / s, w_cat / s

        cfg = {
            "MIN_SUPPORT": t.suggest_int("MIN_SUPPORT", 8, 40),
            "CF_N_USER": t.suggest_int("CF_N_USER", 40, 350, step=10),
            "AGE_BAND": t.suggest_int("AGE_BAND", 5, 25),
            "MIN_SEG_USERS": t.suggest_int("MIN_SEG_USERS", 5, 40),
            "N_CLUSTERS": t.suggest_int("N_CLUSTERS", 6, 24),
            "EXPERT_POOL": t.suggest_int("EXPERT_POOL", 40, 200, step=10),
            "W_SEX": w_sex,
            "W_AGE": w_age,
            "W_CAT": w_cat,
            "GENRE_BOOST": t.suggest_float(
                "GENRE_BOOST", 1.0, 1.4, step=0.05
            ),
        }
        # stocke aussi les poids normalisés dans le trial pour le JSON final
        t.set_user_attr("W_SEX", w_sex)
        t.set_user_attr("W_AGE", w_age)
        t.set_user_attr("W_CAT", w_cat)
        configs.append(cfg)

      results = list(executor.map(_eval_trial_config, configs))

      for trial, score, cfg in zip(trials, results, configs):
        study.tell(trial, score)

      completed = min(i + max_workers, N_TRIALS)
      print(
          f"[{completed}/{N_TRIALS}] Meilleur P@5 actuel : "
          f"{study.best_value:.4f}"
      )

  # Reconstruit les poids normalisés depuis le meilleur trial
  best = dict(study.best_params)
  bt = study.best_trial
  best["W_SEX"] = bt.user_attrs.get("W_SEX", best.get("W_SEX"))
  best["W_AGE"] = bt.user_attrs.get("W_AGE", best.get("W_AGE"))
  best["W_CAT"] = bt.user_attrs.get("W_CAT", 1.0 - best["W_SEX"] - best["W_AGE"])
  best["TRAIN_FRAC"] = FIXED_TRAIN_FRAC
  best["metric"] = "P@5_product_cluster_fusion"
  best["best_P@5"] = study.best_value

  print("\n=== MEILLEURS HYPERPARAMÈTRES (MODEL_CLUSTER) ===")
  print(f"P@5 Max : {study.best_value:.4f}")
  print(json.dumps(best, indent=2))

  out_path = Path("results/best_hyperparams_cluster_optuna.json")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
  print(f"\nRésultats exportés dans : {out_path}")
