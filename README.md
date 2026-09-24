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

NextWatch a pour objectif de développer un système de recommandation de films.
À partir de l'historique des notes et des caractéristiques des utilisateurs et
des films, le projet doit :

- analyser les préférences des utilisateurs ;
- prédire les films susceptibles de plaire à un utilisateur ;
- proposer des recommandations personnalisées ;
- comparer et évaluer différentes méthodes de recommandation.

## Personnes ayant accès au dépôt

| Identité | Type | Mission |
| --- | --- | --- |
| [toto49](https://github.com/toto49) | Propriétaire | |
| [GRem0](https://github.com/GRem0) | Collaborateur | |
| [Oxilian37](https://github.com/Oxilian37) | Invitation en attente | |
| [Myriamirl](https://github.com/Myriamirl) | Invitation en attente | |

> Cette liste regroupe les collaborateurs actifs et les invitations en attente
> retournés par GitHub au moment de la vérification. Les auteurs de commits ne
> sont pas nécessairement des personnes ayant actuellement accès au dépôt.

Les invitations de [Oxilian37](https://github.com/Oxilian37) et
[Myriamirl](https://github.com/Myriamirl) sont encore en attente d'acceptation.
Ils sont inclus dans la liste afin d'anticiper leur accès prochain au dépôt.
