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

NextWatch recommande un **top 5** de films à partir d’âge, sexe et 1–3 films
aimés. Le détail (question, features, baseline, Precision@5) est dans
[`PROJECT_CHARTER.md`](PROJECT_CHARTER.md).

### Notebooks

| Fichier | Contenu |
| --- | --- |
| [`notbook/EDA_NextWatch.ipynb`](notbook/EDA_NextWatch.ipynb) | Exploration |
| [`notbook/MODEL_BASELINE.ipynb`](notbook/MODEL_BASELINE.ipynb) | Dummy + baseline heuristique |
| [`notbook/MODEL_CF.ipynb`](notbook/MODEL_CF.ipynb) | UserCF, FilmCF, hybride |
| [`notbook/MODEL_CLUSTER.ipynb`](notbook/MODEL_CLUSTER.ipynb) | Experts sexe/âge/catégories |
| [`notbook/CONCLUSION.ipynb`](notbook/CONCLUSION.ipynb) | Synthèse + variance P@5 + popularité (tous modèles) |

Le dataset documente aussi le **métier** (`occupation`) ; ce signal est **hors
scope** du modèle (voir charter et EDA).

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
