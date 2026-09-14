# MovieLens 100k — les 3 fichiers CSV

Trois fichiers extraits du dataset [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) (GroupLens, Université du Minnesota), collecté sur MovieLens du 19 septembre 1997 au 22 avril 1998.


| Fichier      | Contenu                                                  |
| ------------ | -------------------------------------------------------- |
| `u.data.csv` | Notes des utilisateurs sur les films                     |
| `u.item.csv` | Description des films (titre, date, genres)              |
| `u.user.csv` | Profil des utilisateurs (âge, sexe, métier, code postal) |


Encodage UTF-8, séparateur virgule, première ligne = noms de colonnes.

Chaque utilisateur a noté au moins 20 films. Les utilisateurs avec moins de 20 notes ou sans informations démographiques complètes ont été retirés par GroupLens.


---

## Jointures

```
u.user.csv.user_id  →  u.data.csv.user_id
u.item.csv.movie_id →  u.data.csv.item_id
```

`u.data.csv` est la table centrale. `item_id` et `movie_id` désignent le même identifiant film.

Pour charger `zip_code` sans perdre les zéros initiaux :

```python
import pandas as pd

notes = pd.read_csv("u.data.csv")
films = pd.read_csv("u.item.csv")
users = pd.read_csv("u.user.csv", dtype={"zip_code": "string"})
```

---

## `u.data.csv` — notes

Une ligne = une note d’un utilisateur sur un film. Ordre des lignes aléatoire.


| Colonne     | Type   | Description                                                            |
| ----------- | ------ | ---------------------------------------------------------------------- |
| `user_id`   | entier | Identifiant utilisateur. Même valeur que `user_id` dans `u.user.csv`.  |
| `item_id`   | entier | Identifiant film. Même valeur que `movie_id` dans `u.item.csv`.        |
| `rating`    | entier | Note, de 1 à 5.                                                        |
| `timestamp` | entier | Instant de la note, en secondes Unix (depuis le 1er janvier 1970 UTC). |


---

## `u.item.csv` — films

Une ligne = un film. Un film peut appartenir à plusieurs genres.


| Colonne               | Type   | Description                                                                                      |
| --------------------- | ------ | ------------------------------------------------------------------------------------------------ |
| `movie_id`            | entier | Identifiant film. Jointure avec `item_id`.                                                       |
| `title`               | texte  | Titre, en général suivi de l’année entre parenthèses. Peut contenir des virgules et des accents. |
| `release_date`        | texte  | Date de sortie, format `JJ-Mmm-AAAA` (ex. `01-Jan-1995`). Peut être vide.                        |
| `video_release_date`  | texte  | Date de sortie vidéo. Colonne présente mais vide.                                                |
| `imdb_url`            | texte  | Lien IMDb. Peut être vide.                                                                       |
| `unknown` … `Western` | 0 ou 1 | 19 colonnes de genres. `1` = le film est de ce genre, `0` sinon.                                 |


Genres, dans l’ordre des colonnes :

`unknown`, `Action`, `Adventure`, `Animation`, `Children's`, `Comedy`, `Crime`, `Documentary`, `Drama`, `Fantasy`, `Film-Noir`, `Horror`, `Musical`, `Mystery`, `Romance`, `Sci-Fi`, `Thriller`, `War`, `Western`

---

## `u.user.csv` — utilisateurs

Une ligne = un utilisateur.


| Colonne      | Type   | Description                                                                                                                                                                                                                                                                                 |
| ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `user_id`    | entier | Identifiant utilisateur. Jointure avec `u.data.csv`.                                                                                                                                                                                                                                        |
| `age`        | entier | Âge en années.                                                                                                                                                                                                                                                                              |
| `gender`     | texte  | `M` (homme) ou `F` (femme).                                                                                                                                                                                                                                                                 |
| `occupation` | texte  | Métier. Valeurs possibles : `administrator`, `artist`, `doctor`, `educator`, `engineer`, `entertainment`, `executive`, `healthcare`, `homemaker`, `lawyer`, `librarian`, `marketing`, `none`, `other`, `programmer`, `retired`, `salesman`, `scientist`, `student`, `technician`, `writer`. |
| `zip_code`   | texte  | Code postal américain, 5 caractères. À lire en texte (ex. `05201`).                                                                                                                                                                                                                         |


---

## Remarques

- `zip_code` est du texte, pas un nombre.
- `item_id` et `movie_id` sont le même identifiant, sous deux noms.
- `video_release_date` est vide.
- Les titres peuvent contenir des virgules et des accents ; le CSV est quoté en UTF-8.
- Un film peut avoir plusieurs genres à 1.

---

## Licence et citation

Usage recherche uniquement, selon les conditions GroupLens

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
>
> DOI : [10.1145/2827872](http://dx.doi.org/10.1145/2827872)

