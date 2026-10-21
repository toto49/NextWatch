# NextWatch

## Environnement Jupyter

Prérequis : **Python 3.10+** installé sur la machine (Windows, macOS ou Linux).

### Installation

**Linux / macOS :**

```bash
./scripts/setup.sh
```

**Windows (PowerShell) :**

```powershell
.\scripts\setup.ps1
```

Le script crée un environnement virtuel `.venv`, installe les dépendances de
[`requirements.txt`](requirements.txt) et enregistre le kernel Jupyter
**NextWatch**.

### Lancement

**Linux / macOS :**

```bash
source .venv/bin/activate
jupyter notebook
```

**Windows (PowerShell) :**

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

Dans Jupyter, choisissez le kernel **NextWatch**.

## Dataset

Le projet utilise le dataset [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
publié par [GroupLens](https://grouplens.org/), de l'Université du Minnesota.
Les fichiers utilisés sont disponibles dans le dossier [`dataset/`](dataset/) et
sont décrits dans [`dataset/data.md`](dataset/data.md).

Le dataset contient :

- les notes données par les utilisateurs aux films ;
- les informations sur les films, notamment leur titre, leur date de sortie et
  leurs genres ;
- les profils utilisateurs, avec l'âge, le sexe et le métier.

## Objectif du projet

NextWatch recommande un **top 5** de films proches d’un film aimé (au moins un
genre en commun), avec un score genres **pondéré par l’âge et le sexe**.
Le détail (question, features, baseline, Precision@5) est dans
[`PROJECT_CHARTER.md`](PROJECT_CHARTER.md).

Le dataset documente aussi le **métier** (`occupation`) ; ce signal est **hors
scope** du modèle (voir charter et EDA).

### Inférence produit (âge / sexe / films → top 5)

Une fois (construit `artifacts/usercf.joblib`) :

```bash
source .venv/bin/activate
python scripts/export_usercf.py
```

Puis dans Python :

```python
from nextwatch_recommend import recommend, list_movies

recommend(age=24, gender="M", liked_movies=[50, 174])
# ou avec des titres :
recommend(age=53, gender="F", liked_movies=["Star Wars (1977)"], k=5)

list_movies()              # tous les titres
list_movies(with_ids=True) # [(id, titre), ...]
```

## Fine-tune des hyperparamètres

Recherche aléatoire (ou grille) sur les hypers du notebook, métrique
**Precision@5 en mode produit** (âge + sexe + film aimé) :

```bash
source .venv/bin/activate
python scripts/tune_hyperparams.py
python scripts/tune_hyperparams.py --n-iter 200
python scripts/tune_hyperparams.py --n-iter 500
python scripts/tune_hyperparams.py --max-cases 150 --n-iter 30   # smoke test
```

Résultats écrits dans [`results/tune_hyperparams.csv`](results/tune_hyperparams.csv)
et [`results/best_hyperparams.json`](results/best_hyperparams.json).
Copier ensuite les meilleurs params dans `notbook/MODEL_BASELINE.ipynb` si besoin.

### Tuning MODEL_CLUSTER (experts + fusion)

```bash
source .venv/bin/activate
python scripts/tune_cluster_optuna.py
```

`TRAIN_FRAC` est **fixé à 0.7** (comme `tune_optuna.py`). Sortie :
[`results/best_hyperparams_cluster_optuna.json`](results/best_hyperparams_cluster_optuna.json).

## Personnes ayant accès au dépôt

| Identité | Nom | Prénom
| --- | --- | --- 
| [toto49](https://github.com/toto49) | Boudaud | Tom 
| [GRem0](https://github.com/GRem0) | Goubin | Rémi
| [Oxilian37](https://github.com/Oxilian37) |Jautrou | Lilian
| [quervainmissamou](https://github.com/quervainmissamou) | MISSAMOU BANSIMBA | Quervain

> Cette liste regroupe les collaborateurs actifs et les invitations en attente
> retournés par GitHub au moment de la vérification. Les auteurs de commits ne
> sont pas nécessairement des personnes ayant actuellement accès au dépôt.
