import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from nextwatch_recommend import list_movies, recommend

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="NextWatch 🎥",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS (a voir pour fichier separer) ---
st.markdown(
    """
    <style>
    /* Fond sombre principal */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* Grille pour les cartes TCG */
    .tcg-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        align-items: stretch;
        margin-top: 20px;
    }

    /* Style d'une carte TCG */
    .tcg-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #38bdf8;
        border-radius: 16px;
        width: 200px;
        padding: 12px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5), 0 0 15px rgba(56, 189, 248, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
    }

    .tcg-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 20px 30px rgba(0,0,0,0.7), 0 0 25px rgba(225, 29, 72, 0.6);
        border-color: #e11d48;
    }

    .tcg-badge {
        position: absolute;
        top: -12px;
        right: -12px;
        background: linear-gradient(45deg, #e11d48, #f43f5e);
        color: white;
        font-weight: 800;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 20px;
        border: 2px solid #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }

    .tcg-poster {
        width: 100%;
        height: 260px;
        object-fit: cover;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }

    .tcg-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f1f5f9;
        text-decoration: none;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 2.6em;
    }

    .tcg-title:hover {
        color: #38bdf8;
    }

    .tcg-footer {
        margin-top: 8px;
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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

# --- EN-TÊTE PRINCIPAL ---
st.title('NextWatch 🎥')
st.markdown('Si une personne a aimé un film en 1997, quels autres films aurait-elle pu aimer ?')
st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.title('NextWatch')
st.sidebar.info("💡 **NextWatch** est un système de recommandation basé sur le jeu de données MovieLens 100k (1997-1998).")

# --- CHARGEMENT DU CATALOGUE DE FILMS ---
try:
    movies_list = list_movies()
except Exception:
    movies_list = []

# ---------------------------------------------------------
# BLOC PROFIL ET SÉLECTION DE FILMS
# ---------------------------------------------------------
_, col_form, _ = st.columns([1, 2, 1])

with col_form:
    with st.container(border=True):
        st.subheader("👤 Profil & Sélection de films")

        # Genre
        genre_label = st.segmented_control("Genre", ['Homme', 'Femme'], default='Homme')
        gender = "M" if genre_label == "Homme" else "F"

        # Âge 
        age = st.number_input(label='Âge', min_value=7, max_value=73, value=25)

        st.write("")

        # Sélection des films
        choices = st.multiselect(
            "Choisis jusqu'à 3 films",
            options=movies_list,
            max_selections=3,
            placeholder="Tape un titre (ex. Star Wars, Toy Story...)"
        )

        st.write("")

        # Bouton de génération
        submit_btn = st.button("Obtenir mes recommandations 🍿", type="primary", use_container_width=True)

st.divider()

# ---------------------------------------------------------
# BLOC RECOMMANDATIONS CARTES TCG 
# ---------------------------------------------------------
st.subheader("🎬 Tes recommandations de films")

if submit_btn:
    if not choices:
        st.warning("⚠️ Veuillez sélectionner au moins un film dans la liste ci-dessus.")
    else:
        with st.spinner("Analyse des préférences en cours..."):
            try:
                recs_ids = recommend(age=age, gender=gender, liked_movies=choices, return_ids=True)
                recs_titles = recommend(age=age, gender=gender, liked_movies=choices, return_ids=False)

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

                    card_html = f'<div class="tcg-card"><div class="tcg-badge">#{i}</div><a href="{imdb_url}" target="_blank"><img src="{poster_url}" class="tcg-poster" alt="{title}"></a><a href="{imdb_url}" target="_blank" class="tcg-title">{title}</a><div class="tcg-footer">Fiche IMDb ↗</div></div>'
                    card_items.append(card_html)

                full_html = f'<div class="tcg-container">{"".join(card_items)}</div>'
                st.markdown(full_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la recommandation : {e}")
else:
    st.info("👆 Remplissez votre profil ci-dessus et cliquez sur le bouton pour faire apparaître vos cartes de recommandation.")