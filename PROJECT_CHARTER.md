# Project Charter — NextWatch

## Question et décision

**Question :** Si une personne a aimé un film en 1997, quels autres films aurait-elle pu aimer ?

**Décision :** proposer un **top 5** de films via un score basé sur les **genres du film aimé** (surtout l’**affinité de goûts** entre genres) et **pondéré par l’âge et le sexe** (poids appris sur MovieLens). Pas de filtre « ≥ 1 genre en commun » côté **modèle** : l’affinité suffit à relier les genres, y compris sans tag partagé.

À l’usage : l’utilisateur saisit **âge**, **sexe** et un **film aimé** (ses genres en découlent). Ce n’est pas un utilisateur du dataset en entrée.

## Unité, population, horizon

| Élément | Définition |
| --- | --- |
| **Unité** | Triplet `(âge, sexe, film_aimé) → classement de films candidats` ; en apprentissage : couples dérivés des notes MovieLens |
| **Population** | MovieLens 100k (19 sept. 1997 – 22 avr. 1998) : ~943 users, ~1 682 films, ~100k notes brutes ; **périmètre d’analyse / modèle** ≈ **99 990** notes après exclusion du genre `unknown` |
| **Horizon** | Recommander d’autres films *a posteriori* sur ce catalogue historique (pas de temps réel) |

## Target, features candidates, colonnes interdites

| | Choix |
| --- | --- |
| **Target** | Film **aimé** si note **≥ 3** ; poids d’exemple plus fort si la note est plus haute (4–5) |
| **Features candidates** | Âge ; sexe ; genres du film aimé ; genres du candidat ; **affinité de goûts** entre genres (corr. des notes moyennes par user) |
| **Colonnes / signaux interdits** | `occupation` (présent dans le CSV, hors scope pipeline et modèle) ; note du candidat à prédire ; agrégats calculés avec cette note ; genre `unknown` / films sans genre utile ; co-occurrence catalogue ; popularité comme facteur du **modèle** ; filtre hard overlap côté **modèle** (réservé à la baseline) |


## Type de problème, baseline, métrique

| | Choix |
| --- | --- |
| **Type** | Ranking / recommandation basée contenu (genres) + profil (âge, sexe) |
| **Baseline** | Top 5 / top 10 des films **populaires dans les genres du film aimé** (overlap naïf + volume de notes) — détail ci-dessous |
| **Métrique** | **Precision@5** (principale, alignée au top 5 produit) et **Precision@10** (secondaire, pour vérifier le ranking au-delà du 5ᵉ) : part des films du top-k réellement aimés (note ≥ 3) sur le hold-out |

**Protocole d’évaluation :** sur MovieLens, à partir d’un film aimé A et du profil (âge, sexe), vérifier si un film B aimé (caché) apparaît dans le top-k. On reporte **Precision@5** et **Precision@10** pour le **modèle** et pour la **baseline**. En inférence produit : seul le film saisi est exclu du top 5 (pas d’historique user).

## Construction de la baseline

**Pourquoi pas la popularité globale ?** Classer les films les plus notés du catalogue entier ignore le film aimé : ce n’est plus « quels *autres* films après celui-ci », c’est « quels films tout le monde note ». Peu cohérent avec la question produit.

**Baseline retenue :** même entrée (film aimé → ses genres), même sortie (top 5 / eval @5 et @10), mais **sans** affinité apprise, **sans** âge, **sans** sexe.

| | Rôle dans le score baseline |
| --- | --- |
| **1. Overlap de genres (naïf)** | On ne garde que les films qui partagent **≥ 1 genre** avec le film aimé. Lien contenu simple (tags), pas une corr. de goûts. |
| **2. Popularité** | Parmi ces candidats, score = volume de notes sur le jeu d’apprentissage (**sans fuite**). Plus le film est souvent noté, plus il monte. |
| **3. Exclusion** | Le film saisi est retiré (on ne se recommande pas lui-même). |

**Popularité** = souvent noté (nombre de notes), pas « bien noté » (moyenne).

**Lecture :** la baseline = « dans le même rayon de tags, propose les blockbusters ». Le **modèle** doit faire mieux en remplaçant l’overlap naïf par l’**affinité de goûts** et en ajoutant **âge** + **sexe**.

## Comment le modèle construit son score

À partir du film aimé, de l’âge et du sexe, chaque candidat du catalogue (hors film saisi) reçoit un **score** (top 5 produit ; top 10 en eval). Tableau trié du **plus important au moins important**.

| | Rôle dans le score |
| --- | --- |
| **1. Affinité de goûts** | Relie le film aimé aux candidats : plus les genres du candidat sont souvent notés haut **avec** ceux du film aimé (corr. des moyennes par user ; NaN ignorés ; multi-genres OK), plus le candidat monte. Signal genres principal, y compris sans tag en commun. |
| **2. Âge** | Correctif profil le plus fort côté démographie (EDA) : les parts de notes bougent nettement avec la tranche (ex. Drama ~33 % → ~47 % de ≤17 à 55+ ; Action ~29 % → ~19 %). Un profil plus âgé pousse Drama/War ; un profil plus jeune pousse Action/Comedy/Children's. |
| **3. Sexe** | Correctif profil plus modéré (écarts typiques de quelques points de % : Action ~−6 F−M, Romance ~+4). Affinage selon F/M (ex. +Romance/Drama femme ; +Action/Sci-Fi homme), sans l’ampleur des écarts liés à l’âge. |

**Contrôle `α` :** ce n’est pas un facteur du score au même titre que les lignes ci-dessus, mais un **hyperparamètre** qui règle l’amplitude des correctifs **âge** puis **sexe** (avec régularisation), choisi par validation vs baseline. Trop fort → le top se détache du film aimé ; trop faible → le ranking repose surtout sur l’affinité seule.

**Principe :** affinité + âge + sexe (l’âge pèse plus que le sexe d’après l’EDA). Pas de popularité ni d’overlap hard dans le modèle. Livrable : **top 5** ; métriques : Precision@5 et @10, comparées à la baseline.

## Pourquoi ces données permettent / ne permettent pas ce test

**Permettent :** apprendre des liens genres ↔ profil (âge/sexe) et une affinité de goûts entre genres ; comparer à une baseline « overlap naïf + popularité » ; mesurer si le top-k contient des films vraiment notés ≥ 3 (Precision@5 et @10).

**Ne permettent pas :** modéliser le visionnage sans note ; généraliser hors panel MovieLens 1997–98 ; prouver une causalité âge/sexe → goût ; une reco riche hors genres (pas de synopsis, cast, etc.).

**Biais à garder (EDA) :** hommes sur-représentés ; offre Drama / Comedy dominante ; « plus noté » ≠ « plus aimé » ; données historiques.
