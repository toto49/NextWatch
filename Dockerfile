# Image Python 3.14 figée sur le digest validé par le projet
FROM python:3.14-slim@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système nécessaires (ex: compilation si besoin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances
COPY requirements.txt .

# Installation des paquets Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste des fichiers du projet
COPY . .

# Définition des ports et variables d'environnement pour Streamlit
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Commande de lancement de l'application Streamlit
CMD ["streamlit", "run", "app.py"]