# Project Charter — NextWatch

## Question et décision

**Question :** Si une personne a aimé un film en 1997, quels autres films aurait-elle pu aimer ?

**Décision :** proposer un **top 5** de films, parmi ceux qui partagent **au moins un genre** avec le film aimé, via un score basé sur les genres et **pondéré par l’âge et le sexe** (poids appris sur MovieLens).

À l’usage : l’utilisateur saisit **âge**, **sexe** et un **film aimé** (ses genres en découlent). Ce n’est pas un utilisateur du dataset en entrée.

## Unité, population, horizon

| Élément | Définition |
| --- | --- |
| **Unité** | Triplet `(âge, sexe, film_aimé) → classement de films candidats` ; en apprentissage : couples dérivés des notes MovieLens |
| **Population** | MovieLens 100k (sept. 1997 – avr. 1998) : ~943 users, ~1 682 films, ~100k notes |
| **Horizon** | Recommander d’autres films *a posteriori* sur ce catalogue historique (pas de temps réel) |

## Target, features candidates, colonnes interdites

| | Choix |
| --- | --- |
| **Target** | Film **aimé** si note **≥ 3** ; poids d’exemple plus fort si la note est plus haute (4–5) |
| **Features candidates** | Âge ; sexe ; genres du film aimé ; genres du candidat ; overlap de genres ; éventuellement popularité du candidat (sans fuite) |
| **Colonnes / signaux interdits** | `occupation` (hors scope) ; note du candidat à prédire ; agrégats calculés avec cette note ; genre `unknown` / films sans genre utile |


## Type de problème, baseline, métrique

| | Choix |
| --- | --- |
| **Type** | Ranking / recommandation basée contenu (genres) + profil (âge, sexe) |
| **Baseline** | Top 5 films les plus populaires partageant ≥ 1 genre avec le film aimé, **sans** âge/sexe |
| **Métrique** | Pertinence du top 5 : part de films du top 5 réellement aimés (≥ 3) + pondération par âge/sexe |

**Protocole d’évaluation. :** sur MovieLens, à partir d’un film aimé A et du profil (âge, sexe), vérifier si un film B aimé (caché) apparaît dans le top 5. En inférence produit : seul le film saisi est exclu du top 5 (pas d’historique user).

## Contrôle de la pondération (âge / sexe)

- Les **poids âge/sexe** sur les genres sont **appris par le modèle** sur MovieLens (pas fixés à la main genre par genre).
- Pour éviter une pondération trop forte : **régularisation** des coefficients et/ou un facteur global `α` (amplitude du correctif profil) choisi par **validation**, en comparaison à la baseline sans âge/sexe.
- Principe : le film aimé (genres) reste le signal principal ; l’âge et le sexe ne font qu’un **réglage fin**.

## Pourquoi ces données permettent / ne permettent pas ce test

**Permettent :** apprendre des liens genres ↔ profil (âge/sexe) et une similarité de genres ; filtrer les candidats par overlap ; comparer à une baseline popularité ; mesurer si le top 5 contient des films vraiment notés ≥ 3.

**Ne permettent pas :** modéliser le visionnage sans note ; généraliser hors panel MovieLens 1997–98 ; prouver une causalité âge/sexe → goût ; une reco riche hors genres (pas de synopsis, cast, etc.).
