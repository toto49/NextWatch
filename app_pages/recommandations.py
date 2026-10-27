import streamlit as st
import pandas as pd
from pathlib import Path
from nextwatch_recommend import list_genres, list_movies, recommend

# --- CHARGEMENT DU FICHIER POSTER.CSV ---
@st.cache_data
def load_poster_data():
    try:
        base_path = Path(__file__).resolve().parent.parent
        poster_file = base_path / "dataset" / "poster.csv"

        df = pd.read_csv(poster_file)
        df["movie_id"] = df["movie_id"].astype(int)
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture de dataset/poster.csv : {e}")
        return pd.DataFrame(columns=["movie_id", "imdb_url", "poster_url"])


df_posters = load_poster_data()

# --- EN-TÊTE HERO ---
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1e1b4b 0%, #0f172a 100%); border-radius: 16px; padding: 24px; margin-bottom: 20px; border: 1px solid #312e81;">
        <h1 style="color: #ffffff; margin:0; font-size: 2.2rem;">🎥 NextWatch AI</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 6px; margin-bottom: 0;">
            Moteur de recommandation basé sur le jeu de données MovieLens 100k (1997-1998).
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# BLOC PROFIL ET SÉLECTION DE FILMS
# ---------------------------------------------------------
_, col_form, _ = st.columns([1, 6, 1])

with col_form:
    with st.container(border=True):
        st.subheader("👤 Profil & préférences")

        search_mode = st.segmented_control(
            "Type de recherche",
            ["Par film", "Par genre"],
            default="Par film",
            help="Par genre : jusqu'à 3 genres, modèle Cluster uniquement. Par film : Cluster ou Voisins.",
            width="stretch",
            key="search_mode",
        )
        search_by_genre = search_mode == "Par genre"

        col_genre, col_age, col_compare = st.columns(3, gap="medium")

        with col_genre:
            genre_label = st.segmented_control(
                "Sexe",
                ["Homme", "Femme"],
                default="Homme",
                width="stretch",
            )
            gender = "M" if genre_label == "Homme" else "F"

        with col_age:
            age = st.number_input(
                label="Âge",
                min_value=7,
                max_value=73,
                value=25,
                width="stretch",
            )

        with col_compare:
            st.write("")
            st.write("")
            compare_mode = st.checkbox(
                "⚡ Mode comparatif (A/B Test)",
                value=False,
                disabled=search_by_genre,
                help="Compare Cluster et Voisins (disponible uniquement en recherche par film)",
            )
        if search_by_genre:
            compare_mode = False

        if search_by_genre:
            model = "cluster"
            st.caption(
                "Recherche par genre → modèle Cluster uniquement (Voisins a besoin de films)."
            )
        elif not compare_mode:
            model_label = st.segmented_control(
                "Modèle à utiliser",
                ["Cluster", "Voisins"],
                default="Cluster",
                help="Cluster = groupes de profils · Voisins = utilisateurs aux goûts similaires",
                width="stretch",
            )
            model = "cluster" if model_label == "Cluster" else "cf"
        else:
            model = "cluster"

        st.write("")

        choices: list[str] = []
        genre_choices: list[str] = []

        if search_by_genre:
            try:
                genres_list = list_genres(model="cluster")
            except Exception as exc:
                st.error(
                    "Les genres ne peuvent pas être chargés. "
                    f"Détail technique : {exc}"
                )
                genres_list = []
            genre_choices = st.multiselect(
                "Choisis jusqu'à 3 genres",
                options=genres_list,
                max_selections=3,
                placeholder="Ex. Comedy, Drama, Sci-Fi…",
            )
        else:
            try:
                movies_list = list_movies(model=model)
            except Exception as exc:
                st.error(
                    "Le catalogue ne peut pas être chargé. "
                    f"Détail technique : {exc}"
                )
                movies_list = []
            choices = st.multiselect(
                "Choisis jusqu'à 3 films",
                options=movies_list,
                max_selections=3,
                placeholder="Tape un titre (ex. Star Wars, Toy Story...)",
            )

        st.write("")

        submit_btn = st.button(
            "Obtenir mes recommandations 🍿",
            type="primary",
            width="stretch",
        )

st.divider()


def render_recommendations(selected_model, *, liked_movies=None, liked_genres=None):
    rec_kwargs = {
        "age": age,
        "gender": gender,
        "model": selected_model,
    }
    if liked_genres is not None:
        rec_kwargs["liked_genres"] = liked_genres
    else:
        rec_kwargs["liked_movies"] = liked_movies

    recs_ids = recommend(**rec_kwargs, return_ids=True)
    recs_titles = recommend(**rec_kwargs, return_ids=False)

    default_poster = (
        "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&auto=format&fit=crop"
    )
    card_items = []

    for i, (mid, title) in enumerate(zip(recs_ids, recs_titles), 1):
        row = df_posters[df_posters["movie_id"] == int(mid)]

        poster_url = default_poster
        imdb_url = "#"

        if not row.empty:
            p_url = row.iloc[0]["poster_url"]
            i_url = row.iloc[0]["imdb_url"]

            if pd.notna(p_url) and str(p_url).startswith("http"):
                poster_url = str(p_url)
            if pd.notna(i_url) and str(i_url).startswith("http"):
                imdb_url = str(i_url)

        card_html = (
            f'<div class="tcg-card">'
            f'<div class="tcg-badge">#{i}</div>'
            f'<img src="{poster_url}" class="tcg-poster" alt="{title}">'
            f'<div class="tcg-title">{title}</div>'
            f'<a href="{imdb_url}" target="_blank" class="tcg-footer-btn">Fiche IMDb ↗</a>'
            f"</div>"
        )
        card_items.append(card_html)

    full_html = f'<div class="tcg-container">{"".join(card_items)}</div>'
    st.markdown(full_html, unsafe_allow_html=True)


# ---------------------------------------------------------
# BLOC RECOMMANDATIONS (AFFICHAGE SIMPLE OU COMPARATIF)
# ---------------------------------------------------------
if submit_btn:
    if search_by_genre and not genre_choices:
        st.warning("⚠️ Sélectionne au moins un genre (maximum 3).")
    elif not search_by_genre and not choices:
        st.warning("⚠️ Sélectionne au moins un film dans la liste ci-dessus.")
    else:
        with st.spinner("Analyse des préférences en cours..."):
            try:
                if search_by_genre:
                    st.subheader("🎬 Tes recommandations de films")
                    render_recommendations("cluster", liked_genres=genre_choices)
                elif compare_mode:
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.subheader("🎯 Modèle Cluster")
                        render_recommendations("cluster", liked_movies=choices)
                    with col_right:
                        st.subheader("🤝 Modèle Voisins")
                        render_recommendations("cf", liked_movies=choices)
                else:
                    st.subheader("🎬 Tes recommandations de films")
                    render_recommendations(model, liked_movies=choices)
            except Exception as e:
                st.error(f"Une erreur est survenue lors de la recommandation : {e}")
else:
    st.info(
        "👆 Remplissez votre profil ci-dessus et cliquez sur le bouton "
        "pour faire apparaître vos cartes de recommandation."
    )
