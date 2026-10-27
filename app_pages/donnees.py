import importlib

import altair as alt
import pandas as pd
import streamlit as st

import nextwatch_eda
from nextwatch_eda import GENRES, load_eda_bundle

COLOR_H = "#4C72B0"
COLOR_F = "#DD8452"
COLOR_TEXT = "#e2e8f0"
COLOR_MUTED = "#94a3b8"


def _load_bundle() -> dict:
    """Recharge le module EDA si le cache / process sert encore l'ancien schéma."""
    bundle = load_eda_bundle()
    if "genre_corr" in bundle:
        return bundle
    importlib.reload(nextwatch_eda)
    nextwatch_eda.load_eda_bundle.clear()
    return nextwatch_eda.load_eda_bundle()


def _dark_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(
            labelColor=COLOR_MUTED,
            titleColor=COLOR_TEXT,
            gridColor="#1e293b",
            domainColor="#334155",
        )
        .configure_legend(
            labelColor=COLOR_TEXT,
            titleColor=COLOR_TEXT,
            fillColor="transparent",
            strokeColor="transparent",
        )
        .configure_title(color=COLOR_TEXT, fontSize=14, anchor="start")
    )


bundle = _load_bundle()

st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1e1b4b 0%, #0f172a 100%); border-radius: 16px; padding: 24px; margin-bottom: 20px; border: 1px solid #312e81;">
        <h1 style="color: #ffffff; margin:0; font-size: 2rem;">Les données MovieLens</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 6px; margin-bottom: 0;">
            Constats descriptifs de l'analyse exploratoire (1997–1998) avant la recommandation.
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 1. Le panel
# ------------------------------------------------------------------
st.subheader("Le panel")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Films au catalogue", f"{bundle['n_films']:,}".replace(",", " "), border=True)
m2.metric("Utilisateurs", f"{bundle['n_users']:,}".replace(",", " "), border=True)
m3.metric("Notes analysées", f"{bundle['n_notes']:,}".replace(",", " "), border=True)
m4.metric(
    "Notes faites par des hommes",
    f"{100 * bundle['male_share']:.0f} %",
    border=True,
)

c1, c2 = st.columns(2)
with c1:
    chart_gender = (
        alt.Chart(bundle["notes_by_gender"])
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("sexe:N", title=None, sort=["Hommes", "Femmes"]),
            y=alt.Y("n_notes:Q", title="Nombre de notes"),
            color=alt.Color(
                "sexe:N",
                scale=alt.Scale(
                    domain=["Hommes", "Femmes"],
                    range=[COLOR_H, COLOR_F],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("sexe:N", title="Sexe"),
                alt.Tooltip("n_notes:Q", title="Notes", format=","),
            ],
        )
        .properties(height=280, title="Volume de notes Hommes / Femmes")
    )
    st.altair_chart(
        _dark_chart(chart_gender),
        width="stretch",
        alt="Les hommes produisent environ trois quarts des notes du panel",
    )

with c2:
    chart_age = (
        alt.Chart(bundle["age_hist"])
        .mark_bar(color="#f97316", cornerRadiusEnd=2)
        .encode(
            x=alt.X("âge:Q", title="Âge"),
            y=alt.Y("n_utilisateurs:Q", title="Utilisateurs"),
            tooltip=[
                alt.Tooltip("âge:Q", title="Âge"),
                alt.Tooltip("n_utilisateurs:Q", title="Utilisateurs"),
            ],
        )
        .properties(height=280, title="Répartition des âges")
    )
    st.altair_chart(
        _dark_chart(chart_age),
        width="stretch",
        alt="Le public MovieLens est surtout jeune adulte",
    )

st.caption(
    "Panel déséquilibré : plus d'hommes que de femmes, et un public surtout jeune adulte. "
    "Les résultats hors ligne reflètent cette composition."
)

st.divider()

# ------------------------------------------------------------------
# 2. Le catalogue
# ------------------------------------------------------------------
st.subheader("Le catalogue")

chart_cat = (
    alt.Chart(bundle["catalogue"])
    .mark_bar(color="#818cf8", cornerRadiusEnd=4)
    .encode(
        x=alt.X("n_films:Q", title="Nombre de films"),
        y=alt.Y("genre:N", sort="-x", title=None),
        tooltip=[
            alt.Tooltip("genre:N", title="Genre"),
            alt.Tooltip("n_films:Q", title="Films"),
        ],
    )
    .properties(height=420, title="Nombre de films par genre")
)
st.altair_chart(
    _dark_chart(chart_cat),
    width="stretch",
    alt="Drama et Comedy dominent le catalogue MovieLens",
)
st.caption(
    "Drama et Comedy dominent l'offre. Un film peut porter plusieurs genres : "
    "les totaux dépassent le nombre de films."
)

st.divider()

# ------------------------------------------------------------------
# 3. Hommes et femmes
# ------------------------------------------------------------------
st.subheader("Hommes et femmes")

gap = bundle["gender_gap"].copy()
gap["couleur"] = gap["écart_pts"].apply(lambda v: "Femmes" if v >= 0 else "Hommes")

chart_gap = (
    alt.Chart(gap)
    .mark_bar(cornerRadiusEnd=3)
    .encode(
        x=alt.X(
            "écart_pts:Q",
            title="Écart de part de notes (points de %) — Femmes − Hommes",
        ),
        y=alt.Y("genre:N", sort=gap["genre"].tolist(), title=None),
        color=alt.Color(
            "couleur:N",
            scale=alt.Scale(domain=["Femmes", "Hommes"], range=[COLOR_F, COLOR_H]),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            alt.Tooltip("genre:N", title="Genre"),
            alt.Tooltip("écart_pts:Q", title="Écart (pts)", format="+.1f"),
            alt.Tooltip("Femmes_%:Q", title="Part femmes %", format=".1f"),
            alt.Tooltip("Hommes_%:Q", title="Part hommes %", format=".1f"),
        ],
    )
    .properties(height=480, title="Parts de notes par genre : écart Femmes − Hommes")
)
rule = (
    alt.Chart(pd.DataFrame({"x": [0]}))
    .mark_rule(color=COLOR_TEXT, strokeWidth=1)
    .encode(x="x:Q")
)
st.altair_chart(
    _dark_chart(chart_gap + rule),
    width="stretch",
    alt="Romance et Drama plus notés chez les femmes, Action et Sci-Fi chez les hommes",
)
st.caption(
    "À droite : genres relativement plus notés par les femmes (Romance, Drama). "
    "À gauche : Action et Sci-Fi plus présents chez les hommes. "
    "Drama et Comedy restent élevés pour les deux — l'écart relativise mieux que la part brute."
)

st.divider()

# ------------------------------------------------------------------
# 4. Avec l'âge
# ------------------------------------------------------------------
st.subheader("Avec l'âge")

chart_age_lines = (
    alt.Chart(bundle["age_curves"])
    .mark_line(point=True, strokeWidth=2.5)
    .encode(
        x=alt.X("tranche:N", title="Tranche d'âge", sort=None),
        y=alt.Y("part_%:Q", title="Part des notes du groupe (%)"),
        color=alt.Color("genre:N", title="Genre"),
        tooltip=[
            alt.Tooltip("tranche:N", title="Tranche"),
            alt.Tooltip("genre:N", title="Genre"),
            alt.Tooltip("part_%:Q", title="Part %", format=".1f"),
        ],
    )
    .properties(height=360, title="Évolution des parts de notes selon l'âge")
)
st.altair_chart(
    _dark_chart(chart_age_lines),
    width="stretch",
    alt="Avec l'âge, Drama monte tandis qu'Action, Comedy et Children's baissent",
)
st.caption(
    "Avec l'âge, la part de Drama (et souvent War) augmente ; "
    "Action, Comedy et Children's baissent plutôt chez les plus âgés. "
    "Indicateur de volume d'activité, pas de satisfaction."
)

st.divider()

# ------------------------------------------------------------------
# 5. Familles de genres
# ------------------------------------------------------------------
st.subheader("Familles de genres")

_genre_rank = {g: i for i, g in enumerate(GENRES)}
heat = bundle["genre_corr"].copy()
heat["i"] = heat["genre_1"].map(_genre_rank)
heat["j"] = heat["genre_2"].map(_genre_rank)
# Triangle inférieur uniquement (sans diagonale) : plus lisible
heat = heat.loc[heat["i"] > heat["j"]].copy()
heat["label"] = heat["corr"].map(
    lambda v: f"{v:.2f}" if abs(v) >= 0.10 else ""
)

heat_base = alt.Chart(heat).encode(
    x=alt.X(
        "genre_2:N",
        title=None,
        sort=GENRES,
        axis=alt.Axis(labelAngle=-40, labelLimit=140, tickSize=0, domain=False),
    ),
    y=alt.Y(
        "genre_1:N",
        title=None,
        sort=GENRES,
        axis=alt.Axis(tickSize=0, domain=False),
    ),
)

chart_tiles = heat_base.mark_rect(cornerRadius=3, stroke="#0f172a", strokeWidth=1.5).encode(
    color=alt.Color(
        "corr:Q",
        title="Co-occurrence",
        scale=alt.Scale(
            domain=[-0.34, -0.17, 0, 0.17, 0.44],
            range=["#e11d48", "#fb7185", "#64748b", "#38bdf8", "#0284c8"],
            clamp=True,
        ),
        legend=alt.Legend(
            orient="right",
            gradientLength=200,
            gradientThickness=14,
            titleFontSize=12,
            labelFontSize=11,
        ),
    ),
    tooltip=[
        alt.Tooltip("genre_1:N", title="Genre"),
        alt.Tooltip("genre_2:N", title="Avec"),
        alt.Tooltip("corr:Q", title="Corrélation", format=".2f"),
    ],
)

chart_labels = heat_base.mark_text(fontSize=10, fontWeight=600).encode(
    text="label:N",
    color=alt.condition(
        alt.datum.corr > 0,
        alt.value("#0f172a"),
        alt.value("#fff7ed"),
    ),
)

chart_heat = (
    (chart_tiles + chart_labels)
    .properties(
        height=580,
        title="Quels genres voyagent ensemble dans le catalogue ?",
    )
)
st.altair_chart(
    _dark_chart(chart_heat),
    width="stretch",
    alt="Animation et Children's sont très liés ; Comedy et Drama s'opposent",
)
st.caption(
    "Bleu = tags souvent présents sur le même film (Animation ↔ Children's, Action ↔ Adventure / Sci-Fi). "
    "Rose = genres qui s'excluent plutôt (Comedy ↔ Drama). "
    "Quand un groupe note plus d'Action, Adventure et Sci-Fi peuvent être entraînés par "
    "ces co-occurrences."
)

st.divider()
st.info(
    "Ces écarts décrivent MovieLens 100k (1997–1998). "
    "Ils sont descriptifs, non causaux, et ne prescrivent pas une pondération "
    "directe des recommandations."
)
