# NextWatch

Système de recommandation de films (MovieLens 100k) : à partir de l’**âge**, du
**sexe** et de **1–3 films aimés**, l’app propose un **top 5**.

L’interface (`app.py`) permet de choisir le modèle (**Cluster** ou **Voisins**), le
profil, jusqu’à 3 films, puis d’afficher les recommandations avec posters IMDb.

Le détail (question, features, baseline, Precision@5) est dans
[`PROJECT_CHARTER.md`](PROJECT_CHARTER.md).

## Lancer l’app (Docker — recommandé)

Prérequis : [Docker](https://docs.docker.com/get-docker/).

L’image est définie par [`Dockerfile`](Dockerfile) (deps via
[`requirements.txt`](requirements.txt), lancement de `app.py` sur le port
**8501**). Les fichiers inutiles sont exclus par [`.dockerignore`](.dockerignore).

Les artefacts `artifacts/cluster.joblib` et `artifacts/usercf.joblib` sont
déjà dans le dépôt et inclus au build.

```bash
docker build -t nextwatch .
docker run --rm -p 8501:8501 nextwatch
```

Puis ouvrir [http://localhost:8501](http://localhost:8501).

Arrêt : `Ctrl+C`, ou `docker stop <CONTAINER_ID>` après `docker ps`.

| `model=` | Source | Artefact |
| --- | --- | --- |
| `"cluster"` (défaut) | `MODEL_CLUSTER.ipynb` | `artifacts/cluster.joblib` |
| `"cf"` | User–CF (`MODEL_CF`) | `artifacts/usercf.joblib` |

## Environnement Python (notebooks, tuning, exports)

Prérequis : **Python 3.14**. Les versions directes sont figées dans
[`requirements.txt`](requirements.txt), comme l’image Docker.

**Linux / macOS :**

```bash
./scripts/setup.sh
source .venv/bin/activate
```

**Windows (PowerShell) :**

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Le script crée `.venv`, installe [`requirements.txt`](requirements.txt) et
enregistre le kernel Jupyter **NextWatch**.

### Streamlit en local (bonus)

Utile pour développer l’UI sans rebuild Docker :

```bash
source .venv/bin/activate
streamlit run app.py
```

Puis [http://localhost:8501](http://localhost:8501). Arrêt : `Ctrl+C`.

### Régénérer les résultats et les artefacts

```bash
python scripts/tuner/tune_optuna.py --n-trials 300
python scripts/tuner/tune_cluster_optuna.py --n-trials 300
python scripts/Models/export_cluster.py
python scripts/Models/export_usercf.py
python scripts/Models/evaluate_final.py
```

Cet ordre est obligatoire : tuning sur la validation, reconstruction sur
apprentissage+validation, puis consommation unique du test final. La synthèse
officielle est écrite dans `results/final_evaluation.json`.

### Résultat final actuel

Sur 921 cas test communs :

- baseline overlap+moyenne : P@5 = **0,0343**, Hit Rate@5 = **15,5 %** ;
- Cluster utilisateurs : P@5 = **0,0534**, Hit Rate@5 = **22,7 %** ;
- User–User CF : P@5 = **0,0932**, Hit Rate@5 = **37,6 %**.

UserCF reste le meilleur des deux modèles servis sur ce protocole. Les deux
modèles ont désormais été réglés sur **300 essais Optuna** ; le meilleur essai
Cluster est le numéro 240, avec P@5 validation = **0,0467**.

Le **Hit Rate@5** est le pourcentage de cas ayant au moins un film réellement
aimé parmi les cinq recommandations.

### Test rapide sans UI

```bash
python nextwatch_recommend.py
```

Ou en Python :

```python
from nextwatch_recommend import recommend, list_movies

recommend(age=24, gender="M", liked_movies=[50, 174], model="cluster")
recommend(age=24, gender="M", liked_movies=[50, 174], model="cf")
list_movies()
list_movies(with_ids=True)
```

## Notebooks (Jupyter)

```bash
source .venv/bin/activate
jupyter notebook
```

Choisir le kernel **NextWatch**.

| Fichier | Contenu |
| --- | --- |
| [`notbook/EDA_NextWatch.ipynb`](notbook/EDA_NextWatch.ipynb) | Exploration |
| [`notbook/MODEL_BASELINE.ipynb`](notbook/MODEL_BASELINE.ipynb) | Définition et résultats officiels de la baseline |
| [`notbook/MODEL_CF.ipynb`](notbook/MODEL_CF.ipynb) | Paramètres, artefact, inférence et score UserCF |
| [`notbook/MODEL_CLUSTER.ipynb`](notbook/MODEL_CLUSTER.ipynb) | Paramètres, artefact, inférence et score Cluster |
| [`notbook/EVAL_VARIANCE_POPULARITY.ipynb`](notbook/EVAL_VARIANCE_POPULARITY.ipynb) | Dispersion P@5 et diagnostic de popularité communs |
| [`notbook/CONCLUSION.ipynb`](notbook/CONCLUSION.ipynb) | Synthèse des trois méthodes sur le test final |

Le métier (`occupation`) est documenté dans le dataset mais **hors scope** du
modèle (voir charter et EDA).

## Dataset

Dataset [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
([GroupLens](https://grouplens.org/), Université du Minnesota). Fichiers dans
[`dataset/`](dataset/), description dans [`dataset/data.md`](dataset/data.md) :

- notes utilisateurs / films ;
- infos films (titre, date, genres) + posters (`poster.csv`) ;
- profils (âge, sexe, métier).

## Protocole et fine-tune des hyperparamètres

Le protocole commun est défini dans `scripts/evaluation_protocol.py` :

- 65 % apprentissage, 15 % validation, 20 % test final ;
- aimé = note strictement supérieure à la moyenne utilisateur de référence ;
- 1 à 3 films tirés aléatoirement avec une seed fixe ;
- tuning uniquement sur validation ;
- mêmes cas de test pour la baseline, Cluster et UserCF.

### UserCF servi

```bash
source .venv/bin/activate
python scripts/tuner/tune_optuna.py --n-trials 300
```

Sortie :
[`results/best_hyperparams_cf_optuna.json`](results/best_hyperparams_cf_optuna.json).

### Cluster utilisateurs servi

```bash
source .venv/bin/activate
python scripts/tuner/tune_cluster_optuna.py --n-trials 300
python scripts/tuner/tune_cluster_optuna.py --n-trials 30   # essai court
```

Le tuner optimise le Cluster pur, sans fusion UserCF ni mélange de popularité.
Sortie :
[`results/best_hyperparams_cluster_optuna.json`](results/best_hyperparams_cluster_optuna.json).

## Clés d’enrichissement facultatives

L’application n’a besoin d’aucune clé. Seuls les scripts de régénération des
posters et liens IMDb utilisent `TMDB_API_KEY` et `OMDB_API_KEY`.

```bash
cp .env.example .env
set -a; source .env; set +a
python dataset/convert_film_url.py
python dataset/image.py
```

Ne commitez jamais `.env`. Les anciennes clés présentes dans l’historique
doivent être révoquées auprès de TMDB et OMDb.

## Structure utile

```
NextWatch/
├── app.py                      # UI Streamlit
├── nextwatch_recommend.py      # Inférence produit
├── artifacts/                  # cluster.joblib, usercf.joblib
├── dataset/                    # MovieLens + poster.csv
├── notbook/                    # EDA, modèles, conclusion
├── results/                    # eval_*.json, best_hyperparams_*.json
└── scripts/
    ├── setup.sh / setup.ps1
    ├── Models/
    │   ├── export_cluster.py
    │   ├── export_usercf.py
    │   ├── evaluate_final.py
    │   └── eval_diagnostics.py
    ├── evaluation_protocol.py
    └── tuner/
        ├── tune_optuna.py              # CF
        └── tune_cluster_optuna.py      # Cluster
```

## Personnes ayant accès au dépôt

| Identité                                  | Nom               | Prénom   |
| ----------------------------------------- | ----------------- | -------- |
| [toto49](https://github.com/toto49)       | Boudaud           | Tom      |
| [GRem0](https://github.com/GRem0)         | Goubin            | Rémi     |
| [Oxilian37](https://github.com/Oxilian37) | Jautrou           | Lilian   |
| [Myriamirl](https://github.com/Myriamirl)   | Taleb             | Morgiane |
