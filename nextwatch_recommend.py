"""Inférence produit NextWatch — Cluster (MODEL_CLUSTER) ou User–CF (MODEL_CF).

Prérequis :
  python scripts/Models/export_cluster.py
  python scripts/Models/export_usercf.py

Exemple :
  from nextwatch_recommend import recommend, list_movies
  recommend(age=24, gender=\"M\", liked_movies=[50, 174], model=\"cluster\")
  recommend(age=24, gender=\"M\", liked_movies=[50, 174], model=\"cf\")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import joblib
import numpy as np

from scripts.evaluation_protocol import dataset_fingerprint, runtime_versions

ROOT = Path(__file__).resolve().parent
CLUSTER_ARTIFACT = ROOT / "artifacts" / "cluster.joblib"
CF_ARTIFACT = ROOT / "artifacts" / "usercf.joblib"

ModelName = Literal["cluster", "cf"]

_CACHE: dict[str, tuple[Path, float, dict]] = {}


def _artifact_path_for(model: ModelName) -> Path:
    if model == "cluster":
        p = CLUSTER_ARTIFACT
        hint = "python scripts/Models/export_cluster.py"
    else:
        p = CF_ARTIFACT
        hint = "python scripts/Models/export_usercf.py"
    if not p.exists():
        raise FileNotFoundError(f"Artefact introuvable : {p}\nLance d'abord : {hint}")
    return p


def load(model: ModelName = "cluster", path: Path | str | None = None) -> dict:
    """Charge (ou recharge si mtime changé) l'artefact du modèle demandé."""
    p = Path(path) if path is not None else _artifact_path_for(model)
    if not p.exists():
        raise FileNotFoundError(f"Artefact introuvable : {p}")
    mtime = p.stat().st_mtime
    key = str(p.resolve())
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == p and cached[1] == mtime:
        return cached[2]
    artifact = joblib.load(p)
    meta = artifact.get("meta", {})
    if meta.get("dataset_sha256") != dataset_fingerprint():
        raise RuntimeError(
            f"Artefact incompatible avec les données courantes : {p}. "
            "Régénérez les artefacts."
        )
    if meta.get("runtime_versions") != runtime_versions():
        raise RuntimeError(
            f"Artefact incompatible avec l'environnement Python courant : {p}. "
            "Réinstallez requirements.txt puis régénérez les artefacts."
        )
    _CACHE[key] = (p, mtime, artifact)
    return artifact


def list_movies(
    *,
    with_ids: bool = False,
    model: ModelName = "cluster",
) -> list[str] | list[tuple[int, str]]:
    """Catalogue de films (titres triés)."""
    # Les deux artefacts partagent le même catalogue ; cluster en priorité.
    try:
        artifact = load(model)
    except FileNotFoundError:
        other: ModelName = "cf" if model == "cluster" else "cluster"
        artifact = load(other)

    movie_title: dict[int, str] = artifact["movie_title"]
    items = sorted(movie_title.items(), key=lambda x: x[1].lower())
    if with_ids:
        return [(int(mid), title) for mid, title in items]
    return [title for _, title in items]


def _resolve_movie_ids(liked_movies: Iterable, artifact: dict) -> list[int]:
    title_to_id = artifact["title_to_id"]
    movie_title = artifact["movie_title"]
    known_ids = set(movie_title.keys())
    out: list[int] = []
    for m in liked_movies:
        if isinstance(m, (int, float)) or (isinstance(m, str) and m.isdigit()):
            mid = int(m)
            if mid not in known_ids:
                raise ValueError(f"item_id inconnu : {mid}")
            out.append(mid)
            continue
        key = str(m).strip().lower()
        if key in title_to_id:
            out.append(title_to_id[key])
            continue
        hits = [i for t, i in title_to_id.items() if key in t]
        if len(hits) == 1:
            out.append(hits[0])
        elif len(hits) == 0:
            raise ValueError(f"Film introuvable : {m!r}")
        else:
            examples = [movie_title[i] for i in hits[:5]]
            raise ValueError(
                f"Titre ambigu {m!r} ({len(hits)} matches). Précise, ex. : {examples}"
            )
    if not out:
        raise ValueError("liked_movies ne doit pas être vide")
    return out


def _normalize_gender(gender: str) -> str:
    gender_input = str(gender).strip().casefold()
    gender_values = {"m": "M", "homme": "M", "f": "F", "femme": "F"}
    if gender_input not in gender_values:
        raise ValueError("gender doit être 'M', 'F', 'Homme' ou 'Femme'")
    return gender_values[gender_input]


def _recommend_user_cluster(
    seeds: list[int],
    age: float,
    gender: str,
    artifact: dict,
    k: int,
) -> list[int]:
    """Aligné sur ``recommend_user_cluster`` de ``MODEL_CLUSTER.ipynb``."""
    params = artifact["params"]
    genres: list[str] = artifact["genres"]
    n_g = len(genres)
    seed_proxy = float(params.get("SEED_PROXY", params.get("LIKED_THRESHOLD", 4)))
    min_group = int(params["MIN_GROUP_CANDIDATES"])
    genre_boost = float(params.get("GENRE_BOOST", 1.0))

    film_genre_matrix: dict[int, list[float]] = artifact["film_genre_matrix"]
    pop_count: dict[int, int] = artifact["pop_count"]
    cluster_film_scores: dict[int, dict[int, float]] = artifact["cluster_film_scores"]
    eligible_ids: set[int] = set(artifact["eligible_ids"])
    centers = np.asarray(artifact["kmeans_centers"], dtype=float)
    scaler = artifact["scaler"]

    seen = set(seeds)

    # Cold-start : genres des seeds = proxy ; reste = moyenne train (comme le notebook)
    raw = np.asarray(scaler.mean_, dtype=float).copy()
    seed_mask = np.zeros(n_g, dtype=bool)
    for qid in seeds:
        vec = film_genre_matrix.get(qid)
        if vec is None:
            continue
        seed_mask |= np.asarray(vec, dtype=float) > 0
    raw[:n_g][seed_mask] = seed_proxy
    raw[n_g] = float(age)
    raw[n_g + 1] = 1.0 if gender == "M" else 0.0
    query = scaler.transform(raw.reshape(1, -1))[0]
    cid = int(np.argmin(np.linalg.norm(centers - query, axis=1)))

    # Ranking : overlap genres seeds, puis score cluster (somme de notes)
    ranked = []
    for mid, sc in cluster_film_scores.get(cid, {}).items():
        if mid in seen or mid not in eligible_ids:
            continue
        vec = film_genre_matrix.get(mid)
        overlap = 0
        if vec is not None and seed_mask.any():
            overlap = int(np.sum((np.asarray(vec, dtype=float) > 0) & seed_mask))
        ranked.append((mid, overlap, float(sc) * (genre_boost**overlap)))
    ranked.sort(key=lambda x: (-x[1], -x[2], x[0]))
    out = [m for m, _, _ in ranked[:k]]

    if len(out) < min_group:
        pop = sorted(
            ((mid, n) for mid, n in pop_count.items() if mid not in seen),
            key=lambda x: (-x[1], x[0]),
        )
        for mid, _ in pop:
            if mid not in out:
                out.append(mid)
            if len(out) >= k:
                break
    return out[:k]


def _recommend_usercf(
    age: float,
    gender: str,
    seeds: list[int],
    artifact: dict,
    k: int,
) -> list[int]:
    """Aligné sur l'artefact User–CF (``MODEL_CF`` / ``export_usercf.py``)."""
    params = artifact["params"]
    cf_n_user = int(params["CF_N_USER"])
    demo_sex_weight = float(params["DEMO_SEX_WEIGHT"])
    age_scale = float(params["AGE_SCALE"])
    rating_power = float(params["RATING_POWER"])
    eligible_ids = set(artifact["eligible_ids"])

    seen = set(seeds)
    neighbors: list[tuple[int, float]] = []
    for qid in seeds:
        for uid, rat, a, g in artifact["fans_of"].get(qid, []):
            sex = 1.0 if g == gender else 0.0
            age_term = 1.0 - min(abs(age - float(a)) / age_scale, 1.0)
            sim = demo_sex_weight * sex + (1.0 - demo_sex_weight) * age_term
            weighted_rat = (float(rat) ** rating_power) / (5.0**rating_power)
            neighbors.append((int(uid), sim * weighted_rat))
    neighbors.sort(key=lambda x: -x[1])
    neighbors = neighbors[:cf_n_user]

    scores: dict[int, float] = {}
    for vid, sim in neighbors:
        for mid, rat in artifact["liked_by_user"].get(vid, []):
            mid = int(mid)
            if mid in seen or mid not in eligible_ids:
                continue
            scores[mid] = scores.get(mid, 0.0) + float(sim) * (
                float(rat) ** rating_power
            )
    return [m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def recommend(
    age: float | int,
    gender: str,
    liked_movies: Iterable,
    *,
    model: ModelName = "cluster",
    k: int = 5,
    return_ids: bool = False,
    artifact_path: Path | str | None = None,
) -> list[str] | list[int]:
    """Top-k selon ``model`` : ``\"cluster\"`` (MODEL_CLUSTER) ou ``\"cf\"`` (User–CF)."""
    if k < 1:
        raise ValueError("k doit être ≥ 1")
    if model not in ("cluster", "cf"):
        raise ValueError("model doit être 'cluster' ou 'cf'")

    artifact = load(model, path=artifact_path)
    gender_val = _normalize_gender(gender)
    age_val = float(age)
    seeds = _resolve_movie_ids(liked_movies, artifact)

    if model == "cluster":
        ranked = _recommend_user_cluster(seeds, age_val, gender_val, artifact, k)
    else:
        ranked = _recommend_usercf(age_val, gender_val, seeds, artifact, k)

    movie_title = artifact["movie_title"]
    if return_ids:
        return ranked
    return [movie_title[m] for m in ranked]


if __name__ == "__main__":
    for m in ("cluster", "cf"):
        try:
            titles = recommend(age=24, gender="M", liked_movies=[50, 174], model=m)
        except FileNotFoundError as e:
            print(f"[{m}] {e}")
            continue
        print(f"\n[{m}] age=24, gender=M, films=[50, 174]")
        for i, t in enumerate(titles, 1):
            print(f"  {i}. {t}")
