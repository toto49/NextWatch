"""Chargement et préparation des données MovieLens 100k pour les notebooks Étape."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GENRES = [
    "unknown",
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


def get_data_dir() -> Path:
    """Dossier dataset/, résolu depuis l'emplacement de ce script."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "dataset",  # notbook/ -> ../dataset
        Path("dataset").resolve(),
        Path("../dataset").resolve(),
    ]
    for path in candidates:
        if path.is_dir() and (path / "u.data.csv").exists():
            return path
    raise FileNotFoundError(
        "Dossier dataset introuvable (u.data.csv manquant). "
        f"Chemins essayés : {[str(p) for p in candidates]}"
    )


def load_prepared(*, verbose: bool = True):
    """Charge et prépare notes, films, users, et la table fusionnée df.

    Returns
    -------
    notes, films, users, df, GENRES
    """
    data_dir = get_data_dir()

    notes = pd.read_csv(data_dir / "u.data.csv")
    films = pd.read_csv(data_dir / "u.item.csv")
    users = pd.read_csv(data_dir / "u.user.csv", dtype={"zip_code": "string"})

    # Timestamp → datetime
    notes = notes.copy()
    notes["datetime"] = pd.to_datetime(notes["timestamp"], unit="s", utc=True)
    notes["date"] = notes["datetime"].dt.date
    # Period ne gère pas bien le fuseau : on convertit en UTC naïf pour le mois
    notes["month"] = notes["datetime"].dt.tz_convert(None).dt.to_period("M").astype(str)
    notes["weekday"] = notes["datetime"].dt.day_name()
    notes["hour"] = notes["datetime"].dt.hour

    # Date de sortie des films
    films = films.copy()
    films["release_date"] = pd.to_datetime(
        films["release_date"], format="%d-%b-%Y", errors="coerce"
    )
    films["release_year"] = films["release_date"].dt.year

    # Table fusionnée pour analyses croisées
    df = notes.merge(users, on="user_id", how="left").merge(
        films, left_on="item_id", right_on="movie_id", how="left"
    )

    if verbose:
        print(f"data_dir: {data_dir}")
        print("notes:", notes.shape)
        print("users:", users.shape)
        print("films:", films.shape)
        print("df fusionné:", df.shape)

    return notes, films, users, df, GENRES


if __name__ == "__main__":
    load_prepared()
