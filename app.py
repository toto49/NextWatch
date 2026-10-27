import streamlit as st

st.set_page_config(
    page_title="NextWatch 🎥",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Fond sombre principal */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* Pas de barre latérale (nav en haut) */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
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

page = st.navigation(
    [
        st.Page(
            "app_pages/recommandations.py",
            title="Recommandations",
            icon=":material/movie:",
            default=True,
        ),
        st.Page(
            "app_pages/donnees.py",
            title="Les données",
            icon=":material/bar_chart:",
        ),
    ],
    position="top",
)
page.run()
