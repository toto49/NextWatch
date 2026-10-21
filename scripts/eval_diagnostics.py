"""Diagnostics transverses : variance P@5 et biais de popularité.

Chaque notebook modèle exporte un bundle JSON via `save_eval_bundle`.
Le notebook `CONCLUSION` (section variance & popularité) charge tous les bundles et compare.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUCKET_ORDER = ["top 20", "top 21–100", "hors top 100"]
BUCKET_COLORS = {
    "top 20": "#ee6c4d",
    "top 21–100": "#e9c46a",
    "hors top 100": "#2a9d8f",
}


def find_results_dir() -> Path:
    """Toujours `NextWatch/results/` (pas le cwd du notebook)."""
    root = Path(__file__).resolve().parents[1]
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def precision_at_k(recommended, relevant, k: int) -> float:
    top = recommended[:k]
    if not top:
        return 0.0
    return sum(1 for m in top if m in relevant) / k


def collect_method_eval(
    recommend_fn: Callable,
    cases: pd.DataFrame,
    *,
    max_k: int = 10,
    rec_k: int = 5,
) -> dict:
    """P@5 / P@10 par cas + listes top-`rec_k` (pour l'analyse popularité)."""
    p5, p10, recs = [], [], []
    for _, row in cases.iterrows():
        rec = recommend_fn(
            row["query_ids"],
            k=max_k,
            age=row["age"],
            gender=row["gender"],
            liked_train_ids=None,
            user_id=None,
        )
        p5.append(float(precision_at_k(rec, row["relevant_ids"], 5)))
        p10.append(float(precision_at_k(rec, row["relevant_ids"], 10)))
        recs.append([int(x) for x in rec[:rec_k]])
    return {"P@5": p5, "P@10": p10, "recs_top5": recs}


def save_eval_bundle(
    source: str,
    methods: dict[str, dict],
    film_n_ratings: pd.Series | dict,
    *,
    n_cases: int,
    random_state: int | None = None,
    train_frac: float | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Écrit `results/eval_<source>.json`."""
    results_dir = results_dir or find_results_dir()
    if isinstance(film_n_ratings, pd.Series):
        pop = {str(int(k)): int(v) for k, v in film_n_ratings.items()}
    else:
        pop = {str(int(k)): int(v) for k, v in film_n_ratings.items()}

    payload = {
        "source": source,
        "n_cases": int(n_cases),
        "random_state": random_state,
        "train_frac": train_frac,
        "film_n_ratings": pop,
        "methods": methods,
    }
    path = results_dir / f"eval_{source}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_eval_bundles(results_dir: Path | None = None) -> list[dict]:
    results_dir = results_dir or find_results_dir()
    bundles = []
    for path in sorted(results_dir.glob("eval_*.json")):
        bundles.append(json.loads(path.read_text(encoding="utf-8")))
    return bundles


def flatten_methods(bundles: Iterable[dict], *, prefix_source: bool = True) -> dict[str, dict]:
    """Fusionne les méthodes de tous les bundles.

    Par défaut préfixe `source/` pour éviter les collisions
    (ex. baseline présente dans plusieurs notebooks).
    """
    out: dict[str, dict] = {}
    for b in bundles:
        src = b.get("source", "?")
        for name, data in b.get("methods", {}).items():
            key = f"{src}/{name}" if prefix_source else name
            out[key] = {**data, "_source": src, "_short": name}
    return out


def shared_film_n_ratings(bundles: list[dict]) -> pd.Series:
    """Prend le premier bundle non vide (splits identiques recommandés)."""
    for b in bundles:
        pop = b.get("film_n_ratings") or {}
        if pop:
            return pd.Series({int(k): int(v) for k, v in pop.items()}, dtype=int)
    return pd.Series(dtype=int)


def variance_table(methods: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, data in methods.items():
        p5 = np.asarray(data["P@5"], dtype=float)
        p10 = np.asarray(data.get("P@10", []), dtype=float)
        row = {
            "method": name,
            "source": data.get("_source", ""),
            "P@5": float(p5.mean()) if len(p5) else np.nan,
            "P@10": float(p10.mean()) if len(p10) else np.nan,
            "var_P@5": float(np.var(p5, ddof=1)) if len(p5) > 1 else np.nan,
            "écart-type": float(np.std(p5, ddof=1)) if len(p5) > 1 else np.nan,
            "n_cases": int(len(p5)),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("P@5", ascending=False).reset_index(drop=True)


def plot_p5_variance(var_df: pd.DataFrame, *, baseline_method: str | None = None):
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(var_df)), 4))
    x = np.arange(len(var_df))
    ax.bar(
        x,
        var_df["P@5"],
        yerr=var_df["écart-type"],
        capsize=3,
        color="#3d5a80",
    )
    if baseline_method and baseline_method in set(var_df["method"]):
        base = float(var_df.loc[var_df["method"] == baseline_method, "P@5"].iloc[0])
        ax.axhline(base, color="#ee6c4d", ls="--", label="baseline")
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(var_df["method"], rotation=25, ha="right")
    ax.set_ylabel("P@5")
    ax.set_ylim(0, min(0.85, float(var_df["P@5"].max() + var_df["écart-type"].max()) + 0.05))
    ax.set_title("P@5 ± écart-type — tous les modèles")
    plt.tight_layout()
    return fig, ax


def _pop_rank(film_n_ratings: pd.Series) -> pd.Series:
    return film_n_ratings.rank(method="min", ascending=False).astype(int)


def _pop_bucket(rank: int) -> str:
    if rank <= 20:
        return "top 20"
    if rank <= 100:
        return "top 21–100"
    return "hors top 100"


def popularity_table(
    methods: dict[str, dict],
    film_n_ratings: pd.Series,
) -> pd.DataFrame:
    pop_rank = _pop_rank(film_n_ratings)
    rows = []
    for name, data in methods.items():
        flat = [m for rec in data.get("recs_top5", []) for m in rec]
        if not flat:
            continue
        n = len(flat)
        counts = Counter(
            _pop_bucket(int(pop_rank.get(int(m), 10**9))) for m in flat
        )
        ranks = [int(pop_rank.get(int(m), np.nan)) for m in flat]
        row = {
            "method": name,
            "source": data.get("_source", ""),
            "rang pop. médian": float(np.nanmedian(ranks)),
        }
        for b in BUCKET_ORDER:
            row[b] = counts[b] / n
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_popularity(bucket_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(bucket_df)), 4.2))
    x = np.arange(len(bucket_df))
    bottom = np.zeros(len(bucket_df))
    for b in BUCKET_ORDER:
        vals = bucket_df[b].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=b, color=BUCKET_COLORS[b])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_df["method"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Part des recommandations")
    ax.set_title("Où tombent les films recommandés ? (popularité train)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig, ax
