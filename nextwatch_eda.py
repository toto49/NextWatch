"""Agrégats EDA MovieLens 100k pour la page Streamlit « Les données »."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

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

AGE_BINS = [0, 17, 24, 34, 44, 54, 100]
AGE_LABELS = ["≤17", "18-24", "25-34", "35-44", "45-54", "55+"]
AGE_FOCUS_GENRES = ["Drama", "Action", "Comedy", "Romance", "Children's", "War"]
DATA_DIR = Path(__file__).resolve().parent / "dataset"


def _preference_share(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Part des notes d'un groupe qui touchent chaque genre (0–1)."""
    out = {}
    for key, sub in frame.groupby(group_col, observed=True):
        lists = sub["genres"]
        out[key] = {
            g: float(lists.map(lambda gs, genre=g: genre in gs).mean())
            for g in GENRES
        }
    return pd.DataFrame(out)


@st.cache_data(ttl=3600, max_entries=2)
def load_eda_bundle(_cache_version: int = 3) -> dict:
    notes = pd.read_csv(DATA_DIR / "u.data.csv")
    films = pd.read_csv(DATA_DIR / "u.item.csv")
    users = pd.read_csv(DATA_DIR / "u.user.csv", dtype={"zip_code": "string"})
    users = users.drop(columns=["occupation"], errors="ignore")

    df_raw = notes.merge(users, on="user_id", how="left").merge(
        films, left_on="item_id", right_on="movie_id", how="left"
    )
    df = df_raw.loc[df_raw["unknown"] != 1].drop(columns=["unknown"]).copy()
    df["genres"] = df[GENRES].apply(
        lambda row: [g for g in GENRES if row[g] == 1],
        axis=1,
    )
    df = df.drop(columns=GENRES)
    df["age_bin"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
    )

    films_clean = films.loc[films["unknown"] != 1].copy()
    n_films = int(films_clean["movie_id"].nunique())
    n_users = int(users["user_id"].nunique())
    n_notes = int(len(df))
    male_share = float((df["gender"] == "M").mean())

    notes_by_gender = (
        df["gender"]
        .map({"M": "Hommes", "F": "Femmes"})
        .value_counts()
        .reindex(["Hommes", "Femmes"])
        .fillna(0)
        .astype(int)
        .rename_axis("sexe")
        .reset_index(name="n_notes")
    )

    age_hist = (
        users["age"]
        .value_counts()
        .sort_index()
        .rename_axis("âge")
        .reset_index(name="n_utilisateurs")
    )

    catalogue = (
        films_clean[GENRES]
        .sum()
        .astype(int)
        .rename_axis("genre")
        .reset_index(name="n_films")
        .sort_values("n_films", ascending=True)
    )

    pref_gender = _preference_share(df, "gender")
    gender_gap = pd.DataFrame(
        {
            "genre": GENRES,
            "Femmes_%": 100 * pref_gender.loc[GENRES, "F"].to_numpy(),
            "Hommes_%": 100 * pref_gender.loc[GENRES, "M"].to_numpy(),
        }
    )
    gender_gap["écart_pts"] = gender_gap["Femmes_%"] - gender_gap["Hommes_%"]
    gender_gap = gender_gap.sort_values("écart_pts").reset_index(drop=True)

    pref_age = _preference_share(df, "age_bin")
    age_curves = (
        (100 * pref_age.loc[AGE_FOCUS_GENRES])
        .T.reset_index(names="tranche")
        .melt(id_vars="tranche", var_name="genre", value_name="part_%")
    )
    age_curves["tranche"] = pd.Categorical(
        age_curves["tranche"], categories=AGE_LABELS, ordered=True
    )
    age_curves = age_curves.sort_values(["genre", "tranche"]).reset_index(drop=True)

    genre_flags = films_clean[GENRES]
    corr = genre_flags.corr(method="pearson").reindex(index=GENRES, columns=GENRES)
    genre_corr = (
        corr.reset_index(names="genre_1")
        .melt(id_vars="genre_1", var_name="genre_2", value_name="corr")
        .dropna()
        .reset_index(drop=True)
    )

    return {
        "n_films": n_films,
        "n_users": n_users,
        "n_notes": n_notes,
        "male_share": male_share,
        "notes_by_gender": notes_by_gender,
        "age_hist": age_hist,
        "catalogue": catalogue,
        "gender_gap": gender_gap,
        "age_curves": age_curves,
        "genre_corr": genre_corr,
    }
