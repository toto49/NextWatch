import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from nextwatch_recommend import list_movies, recommend

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="NextWatch 🎥",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS GLASSMORPHISM & NEON ---
st.markdown(
    """
    <style>
    /* Fond sombre principal */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* Grille pour les cartes TCG */
    .stApp .tcg-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        align-items: stretch;
        margin-top: 15px;
    }

    /* Style d'une carte Glassmorphism */
    .stApp .tcg-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        width: 200px;
        padding: 14px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
    }

    .stApp .tcg-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: #38bdf8;
        box-shadow: 0 12px 28px rgba(56, 189, 248, 0.25), 0 0 20px rgba(225, 29, 72, 0.3);
    }

    .stApp .tcg-badge {
        position: absolute;
        top: -10px;
        right: -10px;
        background: linear-gradient(45deg, #e11d48, #f43f5e);
        color: white;
        font-weight: 800;
        font-size: 0.8rem;
        padding: 3px 9px;
        border-radius: 20px;
        border: 2px solid #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }

    .stApp .tcg-poster {
        width: 100%;
        height: 250px;
        object-fit: cover;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }

    .stApp .tcg-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.6em;
        margin-bottom: 8px;
    }

    /* Bouton IMDb stylisé */
    .stApp .tcg-footer-btn {
        margin-top: auto;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #000000;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 6px 12px;
        border-radius: 8px;
        text-decoration: none;
        transition: opacity 0.2s, transform 0.2s;
        display: inline-block;
        width: 100%;
    }

    .stApp .tcg-footer-btn:hover {
        opacity: 0.9;
        transform: scale(1.03);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- CHARGEMENT DU FICHIER POSTER.CSV ---
@st.cache_data
def load_poster_data():
    try:
        base_path = Path(__file__).resolve().parent
        poster_file = base_path / "dataset" / "poster.csv"
        
        df = pd.read_csv(poster_file)
        df["movie_id"] = df["movie_id"].astype(int)
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture de dataset/poster.csv : {e}")
        return pd.DataFrame(columns=["movie_id", "imdb_url", "poster_url"])

df_posters = load_poster_data()

@st.cache_data
def load_evaluation_summary():
    results_file = Path(__file__).resolve().parent / "results" / "final_evaluation.json"
    try:
        return json.loads(results_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

evaluation_summary = load_evaluation_summary()

# --- EN-TÊTE HERO ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #1e1b4b 0%, #0f172a 100%); border-radius: 16px; padding: 24px; margin-bottom: 20px; border: 1px solid #312e81;">
        <h1 style="color: #ffffff; margin:0; font-size: 2.2rem;">🎥 NextWatch AI</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 6px; margin-bottom: 0;">
            Moteur de recommandation basé sur le jeu de données MovieLens 100k (1997-1998).
        </p>
    </div>
""", unsafe_allow_html=True)

# Indicateurs de données en haut de page
m1, m2, m3 = st.columns(3)
m1.metric(label="Films au catalogue", value="1 682")
m2.metric(label="Utilisateurs analysés", value="943")
m3.metric(label="Évaluations enregistrées", value="100 000")

st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.title('NextWatch')
st.sidebar.info("💡 **NextWatch** compare la recommandation par Clustering et la recommandation par Filtrage Collaboratif (User-CF).")

# ---------------------------------------------------------
# BLOC PROFIL ET SÉLECTION DE FILMS
# ---------------------------------------------------------
_, col_form, _ = st.columns([1, 6, 1])

with col_form:
    with st.container(border=True):
        st.subheader("👤 Profil & Sélection de films")

        col_genre, col_age, col_compare = st.columns(3, gap="medium")

        with col_genre:
            genre_label = st.segmented_control(
                "Genre",
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
            compare_mode = st.checkbox("⚡ Mode comparatif (A/B Test)", value=False, help="Affiche les résultats des 2 modèles côte à côte")

        if not compare_mode:
            model_label = st.segmented_control(
                "Modèle à utiliser",
                ["Cluster", "CF"],
                default="Cluster",
                help="Cluster = MODEL_CLUSTER · CF = User–CF",
                width="stretch",
            )
            model = "cluster" if model_label == "Cluster" else "cf"
        else:
            model = "cluster"

        st.write("")

        # Catalogue selon le modèle choisi
        try:
            movies_list = list_movies(model=model)
        except Exception:
            movies_list = []

        # Sélection des films
        choices = st.multiselect(
            "Choisis jusqu'à 3 films",
            options=movies_list,
            max_selections=3,
            placeholder="Tape un titre (ex. Star Wars, Toy Story...)"
        )

        st.write("")

        # Bouton de génération
        submit_btn = st.button(
            "Obtenir mes recommandations 🍿",
            type="primary",
            width="stretch",
        )

st.divider()

# --- FONCTION UTILITAIRE D'AFFICHAGE DES CARTES ---
def render_recommendations(selected_model):
    method_name = {
        "cluster": "user_cluster_groups",
        "cf": "user_user_cf",
    }[selected_model]
    model_scores = evaluation_summary.get("scores", {}).get(method_name, {})
    hit_rate_5 = model_scores.get("HitRate@5")
    if hit_rate_5 is not None:
        st.metric(
            "Hit Rate@5 — au moins 1 film aimé",
            f"{float(hit_rate_5):.1%}",
            border=True,
        )
        st.caption(
            f"Mesuré sur {evaluation_summary.get('n_cases', 0)} cas du test final commun."
        )

    recs_ids = recommend(
        age=age,
        gender=gender,
        liked_movies=choices,
        model=selected_model,
        return_ids=True,
    )
    recs_titles = recommend(
        age=age,
        gender=gender,
        liked_movies=choices,
        model=selected_model,
        return_ids=False,
    )

    default_poster = "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=500&auto=format&fit=crop"
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

        # Enchaînement compact sur une seule ligne pour éviter que Streamlit n'interprète le HTML comme du texte brut
        card_html = (
            f'<div class="tcg-card">'
            f'<div class="tcg-badge">#{i}</div>'
            f'<img src="{poster_url}" class="tcg-poster" alt="{title}">'
            f'<div class="tcg-title">{title}</div>'
            f'<a href="{imdb_url}" target="_blank" class="tcg-footer-btn">Fiche IMDb ↗</a>'
            f'</div>'
        )
        card_items.append(card_html)

    full_html = f'<div class="tcg-container">{"".join(card_items)}</div>'
    st.markdown(full_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# BLOC RECOMMANDATIONS (AFFICHAGE SIMPLE OU COMPARATIF)
# ---------------------------------------------------------
if submit_btn:
    if not choices:
        st.warning("⚠️ Veuillez sélectionner au moins un film dans la liste ci-dessus.")
    else:
        with st.spinner("Analyse des préférences en cours..."):
            try:
                if compare_mode:
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.subheader("🎯 Modèle Cluster")
                        render_recommendations("cluster")
                    with col_right:
                        st.subheader("🤝 Modèle User-CF")
                        render_recommendations("cf")
                else:
                    st.subheader("🎬 Tes recommandations de films")
                    render_recommendations(model)
            except Exception as e:
                st.error(f"Une erreur est survenue lors de la recommandation : {e}")
else:
    st.info("👆 Remplissez votre profil ci-dessus et cliquez sur le bouton pour faire apparaître vos cartes de recommandation.")