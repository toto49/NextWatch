"""Inférence produit NextWatch (User–CF).

Prérequis : lancer une fois
  python scripts/export_usercf.py

Exemple :
  from nextwatch_recommend import recommend, list_movies
  recommend(age=24, gender=\"M\", liked_movies=[50, 174])
  # → [\"Raiders of the Lost Ark (1981)\", ...]
  list_movies()           # tous les titres (triés)
  list_movies(with_ids=True)  # [(id, titre), ...]
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = ROOT / "artifacts" / "usercf.joblib"

_ARTIFACT: dict | None = None


def load(path: Path | str | None = None) -> dict:
    """Charge (ou recharge) l'artefact joblib."""
    global _ARTIFACT
    p = Path(path) if path is not None else DEFAULT_ARTIFACT
    if not p.exists():
        raise FileNotFoundError(
            f"Artefact introuvable : {p}\n"
            "Lance d'abord : python scripts/export_usercf.py"
        )
    _ARTIFACT = joblib.load(p)
    return _ARTIFACT


def _get_artifact() -> dict:
    if _ARTIFACT is None:
        return load()
    return _ARTIFACT


def list_movies(
    *,
    with_ids: bool = False,
    artifact_path: Path | str | None = None,
) -> list[str] | list[tuple[int, str]]:
    """Retourne tout le catalogue de films (titres triés alphabétiquement).

    Parameters
    ----------
    with_ids : si True, renvoie ``[(item_id, titre), ...]`` au lieu des titres seuls.
    """
    if artifact_path is not None:
        artifact = load(artifact_path)
    else:
        artifact = _get_artifact()

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
        # match partiel sur le titre
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


def _demo_sim(
    age_a: float,
    gender_a: str,
    age_b: float,
    gender_b: str,
    *,
    demo_sex_weight: float,
    age_scale: float,
) -> float:
    sex = 1.0 if gender_a == gender_b else 0.0
    age_term = 1.0 - min(abs(float(age_a) - float(age_b)) / age_scale, 1.0)
    return demo_sex_weight * sex + (1.0 - demo_sex_weight) * age_term


def recommend(
    age: float | int,
    gender: str,
    liked_movies: Iterable,
    *,
    return_ids: bool = False,
    artifact_path: Path | str | None = None,
) -> list[str] | list[int]:
    """Recommande k films à partir de l'âge, du sexe et de 1–3 films aimés.

    Parameters
    ----------
    age : âge de l'utilisateur
    gender : \"M\", \"F\", \"Homme\" ou \"Femme\"
    liked_movies : item_id (int) et/ou titres (str), 1 à 3 typiquement
    k : nombre de recommandations (défaut 5)
    return_ids : si True, renvoie des item_id au lieu des titres
    """
    k: int = 5

    if artifact_path is not None:
        artifact = load(artifact_path)
    else:
        artifact = _get_artifact()

    params = artifact["params"]
    cf_n_user = int(params["CF_N_USER"])
    demo_sex_weight = float(params["DEMO_SEX_WEIGHT"])
    age_scale = float(params["AGE_SCALE"])
    rating_power = float(params["RATING_POWER"])

    gender_input = str(gender).strip().casefold()
    gender_values = {
        "m": "M",
        "homme": "M",
        "f": "F",
        "femme": "F",
    }
    if gender_input not in gender_values:
        raise ValueError("gender doit être 'M', 'F', 'Homme' ou 'Femme'")
    gender_val = gender_values[gender_input]
    age_val = float(age)

    seeds = _resolve_movie_ids(liked_movies, artifact)
    seen = set(seeds)

    fans_of = artifact["fans_of"]
    liked_by_user = artifact["liked_by_user"]
    movie_title = artifact["movie_title"]

    neighbors: list[tuple[int, float]] = []
    for qid in seeds:
        for uid, rat, a, g in fans_of.get(qid, []):
            sim = _demo_sim(
                age_val,
                gender_val,
                a,
                g,
                demo_sex_weight=demo_sex_weight,
                age_scale=age_scale,
            )
            weighted_rat = (float(rat) ** rating_power) / (5.0**rating_power)
            neighbors.append((int(uid), sim * weighted_rat))
    neighbors.sort(key=lambda x: -x[1])
    neighbors = neighbors[:cf_n_user]

    scores: dict[int, float] = {}
    for vid, sim in neighbors:
        for mid, rat in liked_by_user.get(vid, []):
            mid = int(mid)
            if mid in seen:
                continue
            scores[mid] = scores.get(mid, 0.0) + float(sim) * (
                float(rat) ** rating_power
            )

    ranked = [m for m, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]
    if return_ids:
        return ranked
    return [movie_title[m] for m in ranked]


if __name__ == "__main__":
    # Démo rapide
    titles = recommend(age=24, gender="M", liked_movies=[50, 174])
    print("Input : age=24, gender=M, films=[50, 174]")
    for i, t in enumerate(titles, 1):
        print(f"  {i}. {t}")
