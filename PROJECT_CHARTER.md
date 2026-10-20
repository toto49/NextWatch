# Project Charter — NextWatch

## Question et décision

**Question :** Si une personne a aimé un film en 1997, quels autres films aurait-elle pu aimer en 1997 ?

**Décision :** proposer un **top 5** de films à partir d’un **film aimé** et du profil (**âge**, **sexe**). On compare une **baseline heuristique** à **trois modèles** : User–User CF, Film–Film CF, et un **hybride Film×User** (tous **non supervisés**).

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
| **\(y\)** (cible) | \(y = 1\) si la note user→B est **≥ 4**, sinon \(y = 0\). Ranking : maximiser les \(y=1\) dans le top-k (Precision@5 / @10). |
| **\(X\)** | **Entrée produit :** `age`, `gender`, film aimé A. **Signaux modèles :** similarité démo (User–User), voisinages film–film (Film–Film / hybride) ; baseline : genres + mean_rating train de B. |
| **Sortie** | Score \(\hat{s}(X)\) par candidat B → top 5 (produit) / top 10 (eval). |

### Features et colonnes interdites

| | Choix |
| --- | --- |
| **Dans \(X\)** | `age` ; `gender` ; genres A/B (baseline) ; mean_rating train de B ; similarités CF user/item |
| **Hors \(X\) / interdits** | `occupation` ; note de B à prédire ; agrégats calculés avec cette note ; genre `unknown` |

## Type de problème, baseline, modèles, métrique

| | Choix |
| --- | --- |
| **Type** | Ranking / recommandation (collaboratif non supervisé) |
| **Dummy** | Top-k films aléatoires (seed fixe) |
| **Baseline** | Heuristique **sans ML** : overlap ≥ 1 genre avec A + **note moyenne train** (min. de notes) |
| **Modèles** | (1) **User–User CF** ; (2) **Film–Film CF** ; (3) **Hybride Film×User** |
| **Métrique** | **Precision@5** (principale) et **Precision@10** |

**Protocole d’évaluation :** film aimé A + profil → top-k ; films B aimés cachés. En produit : pas d’historique user MovieLens.

## Construction de la baseline (heuristique)

| | Rôle |
| --- | --- |
| **1. Overlap de genres** | Candidats = films partageant **≥ 1 genre** avec A |
| **2. Note moyenne** | Score = moyenne des notes **train** (seuil min. de notes) |
| **3. Exclusion** | Retirer le film A (et les déjà aimés en eval train) |

## Les méthodes (en termes simples)

| Nom | En une phrase | Comment ça choisit les films |
| --- | --- | --- |
| **Baseline** | Règle simple sans ML | Même genre(s) que le film aimé, triés par note moyenne (train). |
| **1. User–User CF** | « Des gens comme toi… » | Entrée app = âge + sexe + film A seulement. Voisins = users train qui ont aimé A, pondérés âge/sexe → leurs films aimés. |
| **2. Film–Film CF** | « Souvent aimés avec… » | Voisins films via les co-notes → agrégation depuis A (et historique train en eval). |
| **3. Hybride Film×User** | Film–Film enrichi par User–User | Fusion des listes / rangs : poids `α` sur Film–Film, `1−α` sur User–User (profil âge/sexe). |

Toutes les méthodes **excluent** le film saisi et, en évaluation, les films déjà aimés en train.

## Pourquoi ces données permettent / ne permettent pas ce test

**Permettent :** voisinages collaboratifs et un hybride ; comparer à une baseline ; mesurer Precision@5/@10 (aimé ≥ 4).

**Ne permettent pas :** visionnage sans note ; généraliser hors MovieLens 1997–98 ; prouver une causalité âge/sexe → goût ; reco hors genres / hors panel.

**Biais à garder (EDA) :** hommes sur-représentés ; offre Drama / Comedy dominante ; « plus noté » ≠ « plus aimé » ; données historiques.
