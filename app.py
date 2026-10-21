import streamlit as st
import pandas as pd
import numpy as np

st.title('Faire une prédiction 🎯')
st.markdown('Si une personne a aimé un film en 1997, quels autres films aurait-elle pu aimer ?')
st.divider()

st.sidebar.title('NextWatch 🎥')

st.segmented_control("Genre", ['Homme','Femme'])
st.number_input(label='Âge', min_value=7, max_value=73)
choices = st.multiselect("Choisi jusqu'a 3 films", [])
