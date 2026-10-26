"""Protocole d'évaluation commun aux modèles NextWatch.

Le protocole simule le produit : un profil (âge, sexe) et 1 à 3 films aimés
tirés aléatoirement de manière reproductible. Les hyperparamètres sont choisis
sur la validation ; le test final reste intact jusqu'à l'évaluation finale.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

RANDOM_STATE = 42
TRAIN_FRAC = 0.65
VALIDATION_FRAC = 0.15
TEST_FRAC = 0.20
MAX_SEEDS = 3

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


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_validation(self) -> pd.DataFrame:
        return pd.concat([self.train, self.validation], ignore_index=True)


def load_dataframe() -> pd.DataFrame:
    """Charge et joint MovieLens en retirant uniquement le genre ``unknown``."""
    notes = pd.read_csv(DATA_DIR / "u.data.csv")
    films = pd.read_csv(DATA_DIR / "u.item.csv")
    users = pd.read_csv(
        DATA_DIR / "u.user.csv", dtype={"zip_code": "string"}
    ).drop(columns=["occupation"], errors="ignore")
    df = notes.merge(users, on="user_id", how="left").merge(
        films, left_on="item_id", right_on="movie_id", how="left"
    )
    df = df.loc[df["unknown"] != 1].drop(columns=["unknown"]).copy()
    df["genres"] = df[GENRES].apply(
        lambda row: [genre for genre in GENRES if row[genre] == 1], axis=1
    )
    return df


def split_dataframe(df: pd.DataFrame) -> DataSplit:
    """Split interaction-level déterministe 65 % / 15 % / 20 %."""
    if not np.isclose(TRAIN_FRAC + VALIDATION_FRAC + TEST_FRAC, 1.0):
        raise ValueError("Les fractions train/validation/test doivent totaliser 1.")

    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(df))
    train_end = int(len(df) * TRAIN_FRAC)
    validation_end = train_end + int(len(df) * VALIDATION_FRAC)

    train = df.iloc[order[:train_end]].reset_index(drop=True)
    validation = df.iloc[order[train_end:validation_end]].reset_index(drop=True)
    test = df.iloc[order[validation_end:]].reset_index(drop=True)
    return DataSplit(train=train, validation=validation, test=test)


def liked_interactions(
    reference: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Films notés strictement au-dessus de la moyenne user de référence."""
    user_mean = reference.groupby("user_id")["rating"].mean()
    out = interactions.copy()
    out["user_mean"] = out["user_id"].map(user_mean)
    return out.loc[
        out["user_mean"].notna() & (out["rating"] > out["user_mean"])
    ].copy()


def build_eval_cases(
    reference: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    max_seeds: int = MAX_SEEDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Construit les mêmes cas produit pour tous les modèles.

    Pour chaque utilisateur évaluable, 1 à ``max_seeds`` films sont tirés
    aléatoirement parmi ses films aimés de référence. Le tirage est stable par
    utilisateur et ne dépend ni de la popularité ni de la note exacte.
    """
    liked_reference = liked_interactions(reference, reference)
    liked_heldout = liked_interactions(reference, heldout)
    relevant = liked_heldout.groupby("user_id")["item_id"].apply(
        lambda values: {int(value) for value in values}
    )
    demos = reference.drop_duplicates("user_id").set_index("user_id")[
        ["age", "gender"]
    ]

    rows: list[dict] = []
    for user_id, group in liked_reference.groupby("user_id", sort=True):
        if user_id not in relevant.index or user_id not in demos.index:
            continue
        candidates = np.asarray(
            sorted({int(value) for value in group["item_id"]}), dtype=int
        )
        if len(candidates) == 0:
            continue
        user_rng = np.random.default_rng(random_state + int(user_id))
        n_seeds = int(user_rng.integers(1, min(max_seeds, len(candidates)) + 1))
        seeds = sorted(
            int(value)
            for value in user_rng.choice(candidates, size=n_seeds, replace=False)
        )
        rel = set(relevant.loc[user_id]) - set(seeds)
        if not rel:
            continue
        demo = demos.loc[user_id]
        rows.append(
            {
                "user_id": int(user_id),
                "query_ids": seeds,
                "age": float(demo["age"]),
                "gender": str(demo["gender"]),
                "relevant_ids": rel,
            }
        )
    return pd.DataFrame(rows)


def precision_at_k(recommended: list[int], relevant: set[int], k: int = 5) -> float:
    """Precision@k à dénominateur fixe, comme dans l'usage produit."""
    return sum(1 for movie_id in recommended[:k] if movie_id in relevant) / float(k)


def protocol_metadata() -> dict:
    return {
        "random_state": RANDOM_STATE,
        "train_frac": TRAIN_FRAC,
        "validation_frac": VALIDATION_FRAC,
        "test_frac": TEST_FRAC,
        "max_seeds": MAX_SEEDS,
        "liked_definition": "rating > user_mean(reference)",
        "seed_selection": "deterministic_random_1_to_3",
    }


def dataset_fingerprint() -> str:
    """SHA-256 stable des trois fichiers MovieLens utilisés."""
    digest = hashlib.sha256()
    for name in ("u.data.csv", "u.item.csv", "u.user.csv"):
        path = DATA_DIR / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    """Versions minimales nécessaires pour relire un artefact joblib."""
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "joblib": version("joblib"),
    }
