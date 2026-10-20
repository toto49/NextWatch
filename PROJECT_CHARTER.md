# Project Charter — NextWatch

## Question et décision

**Question :** Si une personne a aimé un film en 1997, quels autres films aurait-elle pu aimer ?

**Décision :** proposer un **top 5** de films à partir d’un **film aimé** (ses genres) et du profil (**âge**, **sexe**). On compare une **baseline heuristique** à **trois modèles sklearn** (clustering enrichi de films, filtrage collaboratif item–item, régression logistique).

À l’usage : l’utilisateur saisit **âge**, **sexe** et un **film aimé**. Ce n’est pas un utilisateur du dataset en entrée.

## Unité, population, horizon

| Élément | Définition |
| --- | --- |
| **Unité** | Triplet `(âge, sexe, film_aimé) → classement de films candidats` ; en apprentissage : couples dérivés des notes MovieLens |
| **Population** | MovieLens 100k (19 sept. 1997 – 22 avr. 1998) : ~943 users, ~1 682 films, ~100k notes brutes ; **périmètre** ≈ **99 990** notes après exclusion du genre `unknown` |
| **Horizon** | Recommander d’autres films *a posteriori* sur ce catalogue historique (pas de temps réel) |

## Cible \(y\) et variables \(X\)

Chaque exemple est un couple *(profil + film aimé A, film candidat B)*.

| | Définition |
| --- | --- |
| **\(y\)** (cible) | \(y = 1\) si la note user→B est **≥ 4**, sinon \(y = 0\). Ranking : maximiser les \(y=1\) dans le top-k (Precision@5 / @10). Poids plus fort si note **5**. |
| **\(X\)** | **Profil :** `age`, `gender`. **A / B :** vecteurs de `genres` (0/1). **Dérivés selon le modèle :** même cluster KMeans ; similarité cosine genres ; overlap de genres ; note moyenne train de B (sans fuite). |
| **Sortie** | Score \(\hat{s}(X)\) par candidat B → top 5 (produit) / top 10 (eval). Pas de prédiction de la note 1–5 brute. |

### Features et colonnes interdites

| | Choix |
| --- | --- |
| **Dans \(X\)** | `age` ; `gender` ; genres de A et de B ; dérivés cluster / similarité / overlap / mean_rating train de B |
| **Hors \(X\) / interdits** | `occupation` ; note de B à prédire ; agrégats calculés avec cette note ; genre `unknown` ; popularité (volume de notes) comme facteur modèle |

## Type de problème, baseline, modèles, métrique

| | Choix |
| --- | --- |
| **Type** | Ranking / recommandation basée contenu (genres) + profil (âge, sexe) |
| **Dummy** | Top-k films aléatoires (seed fixe) |
| **Baseline** | Heuristique **sans ML** : overlap ≥ 1 genre avec A + **note moyenne train** (min. de notes). **Ce n’est pas le clustering.** |
| **Modèles** | (1) **KMeans** films × genres ; (2) **Item–item CF** (`NearestNeighbors` cosine sur notes train) ; (3) **LogisticRegression** \(P(y=1 \mid X)\) — détail ci-dessous |
| **Métrique** | **Precision@5** (principale) et **Precision@10** : part des films du top-k réellement aimés (note ≥ 4) sur le hold-out |

**Protocole d’évaluation :** film aimé A + profil → top-k ; on vérifie la présence de films B aimés cachés. On reporte P@5 / P@10 pour dummy, baseline et chaque modèle. En inférence produit : seul le film saisi est exclu.

## Construction de la baseline (heuristique)

| | Rôle |
| --- | --- |
| **1. Overlap de genres** | Candidats = films partageant **≥ 1 genre** avec A |
| **2. Note moyenne** | Score = moyenne des notes **train** (seuil min. de notes) |
| **3. Exclusion** | Retirer le film A |

La baseline doit être battue (ou au moins égalée de façon utile) par les modèles ML.

## Les méthodes (baseline + modèles)

| Nom | En une phrase | Comment ça choisit les films |
| --- | --- | --- |
| **Baseline** | Règle simple sans ML | Même genre(s) que le film aimé, triés par note moyenne (train). |
| **1. KMeans enrichi** | Regroupe les films qui se ressemblent | Ressemblance = **genres** + **profil du film** (note moyenne, volume de notes, public âge/sexe sur le train). On propose d'autres films du **même groupe**, un peu réordonnés selon l'âge/sexe de l'utilisateur (`α_age`, `α_sexe`). Ce n'est **pas** la baseline. |
| **2. Item–item CF** | « Souvent aimés avec… » | Compare les films via les **notes des utilisateurs**. On agrège les voisins de tous les films déjà aimés (pondérés par la note). |
| **3. LogisticRegression** | Estime la chance qu'un film plaise | Apprend une proba d'aimer selon âge, sexe, genres, overlap, note moyenne, même cluster ; on classe par cette proba. |

Toutes les méthodes **excluent** le film saisi et les films déjà aimés en train.

**Poids profil (KMeans) :** `α_age` et `α_sexe` règlent l'importance de l'âge et du sexe dans le re-ranking (âge un peu plus que le sexe). Trop forts → on s'éloigne du film aimé ; trop faibles → on retombe sur « meilleurs notes du cluster ».

## Pourquoi ces données permettent / ne permettent pas ce test

**Permettent :** apprendre des groupes de films / similarités de genres / un score supervisé profil+contenu ; comparer à une baseline overlap+moyenne ; mesurer Precision@5/@10 (aimé ≥ 4).

**Ne permettent pas :** visionnage sans note ; généraliser hors MovieLens 1997–98 ; prouver une causalité âge/sexe → goût ; reco hors genres (synopsis, cast, etc.).

**Biais à garder (EDA) :** hommes sur-représentés ; offre Drama / Comedy dominante ; « plus noté » ≠ « plus aimé » ; données historiques.
