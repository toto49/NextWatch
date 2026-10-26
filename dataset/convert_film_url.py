import os
from pathlib import Path
import re
import time
import pandas as pd
import requests

API_KEY = os.getenv("OMDB_API_KEY")
BASE_DIR = Path(__file__).resolve().parent

if not API_KEY:
  raise RuntimeError(
      "OMDB_API_KEY est absente. Copiez .env.example vers .env puis exportez "
      "la variable avant de lancer ce script."
  )
input_file = BASE_DIR / "u.item_updated.csv"
if not input_file.exists():
  input_file = BASE_DIR / "u.item.csv"

df = pd.read_csv(input_file, encoding="latin-1")

if "imdb_url" not in df.columns:
  df["imdb_url"] = ""

limit_reached = False


def clean_title(title):
  # Transforme "Postino, Il" en "Il Postino" si nécessaire
  for article in [", The", ", A", ", An", ", Il", ", Le", ", La", ", L'"]:
    if title.endswith(article):
      clean_art = article.replace(", ", "").strip()
      return f"{clean_art} {title[:-len(article)]}"
  return title


def get_imdb_url(row):
  global limit_reached
  if limit_reached:
    return row.get("imdb_url", "")

  # Ne pas recalculer si déjà trouvé
  current_url = str(row.get("imdb_url", ""))
  if current_url.startswith("https://www.imdb.com/title/tt"):
    return current_url

  title_raw = str(row["title"])
  match = re.search(r"^(.*?)\s*\((\d{4})\)", title_raw)
  if match:
    title = clean_title(match.group(1).strip())
    year = match.group(2).strip()
    params = {"t": title, "y": year, "apikey": API_KEY}
  else:
    params = {"t": clean_title(title_raw.strip()), "apikey": API_KEY}

  try:
    res = requests.get("https://www.omdbapi.com/", params=params, timeout=5)
    data = res.json()

    if data.get("Response") == "True" and data.get("imdbID"):
      return f"https://www.imdb.com/title/{data['imdbID']}/"
    elif data.get("Error") == "Request limit reached!":
      print("\n[!] Limite de 1 000 requêtes journalières atteinte.")
      limit_reached = True
  except Exception as e:
    print(f"Erreur sur {title_raw}: {e}")

  return ""


print("Traitement en cours...")
for idx, row in df.iterrows():
  if limit_reached:
    break
  new_url = get_imdb_url(row)
  if new_url:
    df.at[idx, "imdb_url"] = new_url

  # Sauvegarde intermédiaire toutes les 50 requêtes
  if idx % 50 == 0 and idx > 0:
    df.to_csv(BASE_DIR / "u.item_updated.csv", index=False, encoding="utf-8")
    print(f"{idx} films traités...")

df.to_csv(BASE_DIR / "u.item_updated.csv", index=False, encoding="utf-8")
print("Sauvegarde effectuée dans u.item_updated.csv !")