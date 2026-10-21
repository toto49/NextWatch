#!/usr/bin/env python3
"""Fine-tune NextWatch (hybride Film×User) avec Optuna.

TRAIN_FRAC est fixé à 0.7 pour un protocole d'éval stable (pas dans la search).
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

FIXED_TRAIN_FRAC = 0.7

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

FIXED_TRAIN_FRAC = 0.25
FIXED_MIN_SUPPORT = 19
RANDOM_STATE = 42
LIKED_THRESHOLD = 4
MAX_SEEDS = 3

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


def _eval_trial_config(args: tuple[dict, str]) -> float:
  """Évalue une config selon le modèle ciblé ('film_cf', 'user_cf', ou 'hybrid')."""
  cfg, target_model = args
  global _DF
  assert _DF is not None

  random_state = 42
  liked_threshold = 4
  max_seeds = 3

  min_support = cfg["MIN_SUPPORT"]
  cf_n_user = int(cfg["CF_N_USER"])
  cf_n_film = int(cfg["CF_N_FILM"])
  hybrid_alpha = cfg["HYBRID_ALPHA"]
  hybrid_pool = cfg["HYBRID_POOL"]
  demo_sex_weight = cfg["DEMO_SEX_WEIGHT"]
  age_scale = float(cfg["AGE_SCALE"])

  # 1. Split Train / Test (sans chevauchement) — TRAIN_FRAC fixé
  train_idx = _DF.sample(
      frac=FIXED_TRAIN_FRAC, random_state=random_state
  ).index
  train = _DF.loc[train_idx].reset_index(drop=True)
  test = _DF.drop(train_idx).reset_index(drop=True)

  liked_train = train.loc[train["rating"] >= LIKED_THRESHOLD]
  liked_test = test.loc[test["rating"] >= LIKED_THRESHOLD]
  train_counts = train.groupby("item_id").size()

  cand_A = liked_train.copy()
  cand_A["pop"] = cand_A["item_id"].map(train_counts)

  def _get_top_seeds(df_u):
    sorted_u = df_u.sort_values(["rating", "pop"], ascending=[False, False])
    return list(sorted_u["item_id"].unique()[:MAX_SEEDS])

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

  # Preparation Film-CF
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

  tfidf = TfidfTransformer()
  mat_iu_tfidf = tfidf.fit_transform(mat_iu)

  cf_n_film = int(cfg.get("CF_N_FILM", 80))
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

  # Preparation User-CF
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
      for mid, grp in train.loc[train["rating"] >= LIKED_THRESHOLD].groupby(
          "item_id"
      )
  }

  movies = _DF.drop_duplicates("item_id").set_index("item_id")
  movie_genres = movies["genres"].to_dict()

  # Fonctions de Recommandation
  def recommend_film_cf(query_ids):
    seeds = set(int(x) for x in query_ids)
    scores = {}
    for qid in seeds:
      for mid, sim in CF_ITEM.get(qid, []):
        if mid in seeds:
          continue
        scores[mid] = scores.get(mid, 0.0) + sim
    return [
        m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]

  def recommend_user_cf(query_ids, age, gender):
    seeds = set(int(x) for x in query_ids)
    age_val = 30.0 if age is None else float(age)
    gender_val = "M" if gender is None else gender
    cf_n_user = int(cfg.get("CF_N_USER", 200))
    demo_sex_weight = float(cfg.get("DEMO_SEX_WEIGHT", 0.1))
    age_scale = float(cfg.get("AGE_SCALE", 60))

    neighbors = []
    for qid in seeds:
      for uid, rat, a, g in fans_of.get(qid, []):
        sex = 1.0 if gender_val == g else 0.0
        age_term = 1.0 - min(abs(age_val - float(a)) / age_scale, 1.0)
        demo = demo_sex_weight * sex + (1.0 - demo_sex_weight) * age_term
        neighbors.append((int(uid), demo * float(rat) / 5.0))

    neighbors.sort(key=lambda x: -x[1])
    neighbors = neighbors[:cf_n_user]

    scores = {}
    for vid, sim in neighbors:
      for mid, rat in liked_by_user.get(vid, []):
        if mid in seeds:
          continue
        scores[mid] = scores.get(mid, 0.0) + float(sim) * float(rat)

    return [
        m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]

  def recommend_hybrid(query_ids, age, gender):
    pool = int(cfg.get("HYBRID_POOL", 150))
    alpha = float(cfg.get("HYBRID_ALPHA", 0.5))

    seeds = set(int(x) for x in query_ids)
    scores_f = {}
    for qid in seeds:
      for mid, sim in CF_ITEM.get(qid, []):
        if mid in seeds:
          continue
        scores_f[mid] = scores_f.get(mid, 0.0) + sim
    film_list = [
        m for m, _ in sorted(scores_f.items(), key=lambda x: -x[1])[:pool]
    ]

    age_val = 30.0 if age is None else float(age)
    gender_val = "M" if gender is None else gender
    cf_n_user = int(cfg.get("CF_N_USER", 200))
    demo_sex_weight = float(cfg.get("DEMO_SEX_WEIGHT", 0.1))
    age_scale = float(cfg.get("AGE_SCALE", 60))

    neighbors = []
    for qid in seeds:
      for uid, rat, a, g in fans_of.get(qid, []):
        sex = 1.0 if gender_val == g else 0.0
        age_term = 1.0 - min(abs(age_val - float(a)) / age_scale, 1.0)
        demo = demo_sex_weight * sex + (1.0 - demo_sex_weight) * age_term
        neighbors.append((int(uid), demo * float(rat) / 5.0))

    neighbors.sort(key=lambda x: -x[1])
    neighbors = neighbors[:cf_n_user]

    scores_u = {}
    for vid, sim in neighbors:
      for mid, rat in liked_by_user.get(vid, []):
        if mid in seeds:
          continue
        scores_u[mid] = scores_u.get(mid, 0.0) + float(sim) * float(rat)
    user_list = [
        m for m, _ in sorted(scores_u.items(), key=lambda x: -x[1])[:pool]
    ]

    scores = {}
    for rank, mid in enumerate(film_list):
      scores[mid] = scores.get(mid, 0.0) + alpha / (1.0 + rank)
    for rank, mid in enumerate(user_list):
      scores[mid] = scores.get(mid, 0.0) + (1.0 - alpha) / (1.0 + rank)

    union_g = set()
    for qid in seeds:
      union_g.update(movie_genres.get(qid, []))
    for mid in list(scores.keys()):
      if union_g & set(movie_genres.get(mid, [])):
        scores[mid] *= 1.15

    return [
        m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]

  # Évaluation selon la cible
  p5_scores = []
  for _, row in eval_cases.iterrows():
    if target_model == "film_cf":
      rec = recommend_film_cf(row["query_ids"])
    elif target_model == "user_cf":
      rec = recommend_user_cf(row["query_ids"], row["age"], row["gender"])
    else:
      rec = recommend_hybrid(row["query_ids"], row["age"], row["gender"])

    hit = sum(1 for m in rec if m in row["rel"]) / 5.0
    p5_scores.append(hit)

  return float(np.mean(p5_scores))


if __name__ == "__main__":
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
      f"=== Lancement Optuna ({N_TRIALS} essais) — "
      f"TRAIN_FRAC fixé={FIXED_TRAIN_FRAC} ==="
  )

  sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=40)
  study = optuna.create_study(direction="maximize", sampler=sampler)

  max_workers = os.cpu_count() or 4
  best_params_all = {
      "TRAIN_FRAC": FIXED_TRAIN_FRAC,
      "MIN_SUPPORT": FIXED_MIN_SUPPORT,
  }

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
            "MIN_SUPPORT": t.suggest_int("MIN_SUPPORT", 1, 40),
            "CF_N_USER": t.suggest_int("CF_N_USER", 20, 350, step=10),
            "CF_N_FILM": t.suggest_int("CF_N_FILM", 20, 200, step=10),
            "HYBRID_ALPHA": t.suggest_float(
                "HYBRID_ALPHA", 0.05, 0.95, step=0.05
            ),
            "HYBRID_POOL": t.suggest_int("HYBRID_POOL", 40, 350, step=10),
            "DEMO_SEX_WEIGHT": t.suggest_float(
                "DEMO_SEX_WEIGHT", 0.0, 0.8, step=0.10
            ),
            "AGE_SCALE": t.suggest_int("AGE_SCALE", 15, 100, step=5),
        }
        configs.append(cfg)

      results = list(executor.map(_eval_trial_config, configs))

      for trial, score in zip(trials, results):
        study.tell(trial, score)

      completed = min(i + max_workers, N_TRIALS)
      print(
          f"[{completed}/{N_TRIALS}] Meilleur P@5 actuel : {study.best_value:.4f}"
      )

  best = dict(study.best_params)
  best["TRAIN_FRAC"] = FIXED_TRAIN_FRAC
  best["best_P@5"] = study.best_value

  print("\n=== MEILLEURS HYPERPARAMÈTRES TROUVÉS ===")
  print(f"P@5 Max : {study.best_value:.4f}")
  print(json.dumps(best, indent=2))

  out_path = Path("results/best_hyperparams_optuna.json")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
  print(f"\nRésultats exportés dans : {out_path}")