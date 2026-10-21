import streamlit as st
import pandas as pd
import numpy as np
from nextwatch_recommend import list_movies, recommend

def afficherReccomandation(age,genre,choices):
    st.write(recommend(age,genre,choices))

st.title('Faire une prédiction 🎯')
st.markdown('Si une personne a aimé un film en 1997, quels autres films aurait-elle pu aimer ?')
st.divider()

st.sidebar.title('NextWatch 🎥')

cAgeGenre = st.container()
col1, col2, col3 = cAgeGenre.columns(3)

genre = col1.segmented_control("Genre", ['Homme','Femme'], default='Homme')
age = col2.number_input(label='Âge', min_value=7, max_value=73)
choices = st.multiselect("Choisi jusqu'a 3 films", list_movies(),max_selections=3)
st.button(label='Envoyer', on_click=afficherReccomandation, args=(age,genre,choices))
