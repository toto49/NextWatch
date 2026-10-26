import os
import re
import time
import pandas as pd
import requests

API_KEY = os.getenv("TMDB_API_KEY")
INPUT_CSV = "dataset/u.item_updated.csv"
OUTPUT_CSV = "dataset/u.item_with_posters.csv"

if not API_KEY:
  raise RuntimeError(
      "TMDB_API_KEY est absente. Copiez .env.example vers .env puis exportez "
      "la variable avant de lancer ce script."
  )

def clean_title_and_year(raw_title):
  year_match = re.search(r'\((\d{4})\)', str(raw_title))
  year = year_match.group(1) if year_match else None
  title = re.sub(r'\s*\(\d{4}\)', '', str(raw_title)).strip()

  article_match = re.match(
      r'^(.*),\s*(The|A|An|Il|La|Le|Les)\b', title, re.IGNORECASE
  )
  if article_match:
    title = f'{article_match.group(2)} {article_match.group(1)}'

  return title, year


# Reprise depuis le fichier de sortie s'il existe déjà, sinon lecture du fichier source
if os.path.exists(OUTPUT_CSV):
  df = pd.read_csv(OUTPUT_CSV)
  print(f"Reprise à partir du fichier existant : {OUTPUT_CSV}")
else:
  df = pd.read_csv(INPUT_CSV)

if 'poster_url' not in df.columns:
  df['poster_url'] = None

total = len(df)
print(f'Traitement de {total} films au total...\n')

for index, row in df.iterrows():
  # Skip si l'URL existe déjà
  if pd.notna(row['poster_url']) and str(row['poster_url']).strip() != '':
    continue

  title, year = clean_title_and_year(row['title'])

  params = {"api_key": API_KEY, "query": title, "language": "fr-FR"}
  if year:
    params["year"] = year

  poster_url = None
  try:
    res = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params=params,
        timeout=5,
    ).json()
    results = res.get('results', [])

    if results and results[0].get('poster_path'):
      poster_url = f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
  except Exception:
    pass

  # Mise à jour du dataframe
  df.at[index, 'poster_url'] = poster_url

  # Affichage de l'avancement dans le terminal
  status = 'OK' if poster_url else 'Non trouvé'
  print(f'[{index + 1}/{total}] {title} ({year or "N/A"}) -> {status}')

  # Sauvegarde intermédiaire tous les 50 films pour sécuriser la progression
  if (index + 1) % 50 == 0:
    df.to_csv(OUTPUT_CSV, index=False)

  time.sleep(0.02)

# Sauvegarde finale
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nTerminé ! Fichier à jour sauvegardé dans '{OUTPUT_CSV}'.")