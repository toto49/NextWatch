# Project Charter — NextWatch

## Question et décision produit

**Question :** à partir de l’âge, du sexe et de **1 à 3 films aimés**, quels
autres films du catalogue MovieLens 100k recommander ?

**Décision :** proposer un top 5 avec deux modèles sélectionnables :

1. **Cluster utilisateurs** : KMeans sur les profils de goûts par genre et les
   variables démographiques, puis classement des films appréciés dans le groupe.
2. **User–User CF** : recherche de profils ayant aimé les mêmes films, pondérée
   par la proximité d’âge et de sexe.

La baseline n’est pas servie dans l’application ; elle sert uniquement de
référence d’évaluation.

## Population, cible et exclusions

| Élément | Définition |
| --- | --- |
| Population | MovieLens 100k, limité à 99 990 interactions après retrait des deux films de genre `unknown` |
| Entrée produit | Âge, sexe et 1 à 3 films aimés |
| Cible « aimé » | Note strictement supérieure à la moyenne de l’utilisateur calculée sur les données de référence |
| Sortie | Classement de films ; top 5 dans l’application |
| Hors scope | Métier, code postal, temps réel, catalogue extérieur à MovieLens 1997–1998 |

Seuls les films fournis en entrée sont exclus du classement. Le volume de notes
n’est pas un facteur de score ; il sert au filtrage de support et, pour Cluster,
uniquement de repli lorsque le groupe ne fournit pas assez de candidats.

## Méthodes comparées

| Méthode | Principe |
| --- | --- |
| Baseline overlap+moyenne | Films partageant au moins un genre avec les seeds, ayant au moins 5 notes de référence, triés par note moyenne |
| Cluster utilisateurs | Profil moyen de notes par genre + âge + sexe, KMeans, puis score des films aimés dans le cluster et overlap de genres |
| User–User CF | Fans des seeds classés par proximité démographique, puis agrégation pondérée de leurs autres films aimés |

Les anciens prototypes FilmCF, hybride FilmCF/UserCF, fusion Cluster/UserCF et
régression logistique ne font pas partie du produit ni de la comparaison finale.
Les notebooks canoniques ne présentent plus leurs anciens scores.

## Protocole d’évaluation

Le protocole canonique est implémenté dans
`scripts/evaluation_protocol.py`.

- split déterministe par interaction : **65 % apprentissage, 15 % validation,
  20 % test final** ;
- seed aléatoire fixée à 42 ;
- 1 à 3 films tirés aléatoirement et de façon reproductible parmi les films
  aimés de référence ;
- même définition de la cible et mêmes cas pour tous les modèles ;
- tuning Optuna uniquement sur la validation ;
- sélection des hyperparamètres sans lecture du test ;
- reconstruction des artefacts sur apprentissage + validation ;
- consommation du test uniquement par `scripts/Models/evaluate_final.py` ;
- métrique principale : Precision@5 ;
- métriques complémentaires : Precision@10 et Hit Rate@5, soit la part des cas
  ayant au moins un film aimé dans le top 5.

Ce protocole simule l’usage produit pour des utilisateurs présents dans
MovieLens. Il ne constitue pas un test de véritable nouvel utilisateur absent
des données.

## Traçabilité et reproductibilité

Chaque artefact enregistre :

- le protocole et le split d’entraînement ;
- les hyperparamètres sélectionnés ;
- les versions Python, NumPy, pandas, scikit-learn et joblib ;
- le SHA-256 des trois fichiers MovieLens utilisés ;
- la confirmation que le test final n’a pas servi à l’apprentissage.

Les scores officiels sont ceux de `results/final_evaluation.json`. Les bundles
`eval_baseline.json`, `eval_cluster.json` et `eval_cf.json` doivent provenir de
la même exécution finale.

## Limites

Les données permettent une comparaison hors ligne sur un catalogue historique.
Elles ne permettent pas d’observer les visionnages sans note, de généraliser à
un catalogue moderne, ni de démontrer une causalité entre âge, sexe et goûts.
Les biais relevés dans l’EDA restent applicables : hommes sur-représentés,
catalogue dominé par Drama/Comedy et confusion possible entre popularité et
préférence.
