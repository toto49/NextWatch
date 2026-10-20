"""Enrich KMeans features + simple model docs in notebook + charter."""
import json
from pathlib import Path
from nbformat import from_dict
from nbformat.validator import normalize

path = Path(__file__).with_name("MODEL_BASELINE.ipynb")
nb = json.loads(path.read_text())


def set_src(idx, src, cell_type=None):
    c = nb["cells"][idx]
    if cell_type:
        c["cell_type"] = cell_type
    c["source"] = [l + "\n" for l in src.strip("\n").split("\n")]
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None


set_src(
    0,
    """
# MODEL_BASELINE — NextWatch

On compare une **baseline simple** a **3 modeles**. Tous proposent un top de films a partir d'un film aime (et du profil pour certains). Les films deja aimes en train sont exclus des suggestions.

## Lexique rapide des methodes

| Nom | En une phrase | Idee simple |
| --- | --- | --- |
| **Dummy** | Tire des films au hasard | Sert de plancher : si on ne fait pas mieux, le modele est inutile. |
| **Baseline** | Meme genre(s) que le film aime, classes par note moyenne | Regle du pouce sans ML : "dans le meme rayon, les mieux notes". |
| **KMeans (enrichi)** | Regroupe les films qui se ressemblent, puis propose dans le meme groupe | Ressemblance = **genres** + **profil du film** (note moyenne, popularite, public age/sexe sur le train). Puis on ajuste un peu selon l'age/sexe de l'utilisateur. |
| **Item–item CF** | "Ceux qui ont aime ce film ont aussi aime…" | Compare les films via les **notes des utilisateurs** (pas seulement les genres). |
| **LogisticRegression** | Estime la chance qu'un film plaise (note >= 4) | Apprend un score a partir de l'age, du sexe, des genres, etc., puis classe les candidats. |

**Metriques :** Precision@5 et Precision@10 (part de films vraiment aimes, note >= 4). **Seed :** 42.
""",
)

set_src(
    9,
    """
## 5. Entrainement des modeles (train uniquement)

### KMeans enrichi — regrouper les films
On ne clusterise plus seulement avec les genres. Chaque film devient un vecteur :
- ses **genres** (0/1) ;
- sa **note moyenne** et son **nombre de notes** (train) ;
- le **profil de son public** : part femmes / hommes et parts par tranches d'age parmi ceux qui l'ont note (train).

Ces chiffres sont **standardises** (meme echelle) puis on lance KMeans ; le meilleur nombre de groupes `K` est choisi avec la silhouette.

### Item–item CF — films souvent aimes ensemble
On place chaque film dans l'espace des notes utilisateurs, puis on trouve ses voisins (cosine).

### LogisticRegression — proba d'aimer
On apprend sur des exemples train : le film B est-il aime (note >= 4) selon le profil et le contenu.
""",
)

# Replace KMeans section at start of cell 10; keep CF and LogReg after
cell10 = "".join(nb["cells"][10]["source"])
marker_cf = "# --- Item-item CF (notes) ---"
if marker_cf not in cell10:
    raise SystemExit("CF marker missing")
rest = cell10[cell10.index(marker_cf) :]

kmeans_new = '''
# --- KMeans enrichi : genres + profil notes du film (train only) ---
# Profil public par film (sans fuite)
tr = train.copy()
tr["age_bin"] = pd.cut(
    tr["age"], bins=[0, 17, 24, 34, 44, 54, 100],
    labels=["<=17", "18-24", "25-34", "35-44", "45-54", "55+"],
)
AGE_BINS = ["<=17", "18-24", "25-34", "35-44", "45-54", "55+"]

film_n = train.groupby("item_id").size().rename("n_ratings")
film_mean = train.groupby("item_id")["rating"].mean().rename("mean_rating_raw")
film_female = train.assign(is_F=train["gender"].eq("F")).groupby("item_id")["is_F"].mean().rename("share_F")
age_share = (
    tr.groupby(["item_id", "age_bin"], observed=True).size()
    .unstack(fill_value=0)
    .reindex(columns=AGE_BINS, fill_value=0)
)
age_share = age_share.div(age_share.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
age_share.columns = [f"share_age_{c}" for c in age_share.columns]

# Matrice complete alignee sur le catalogue
feat = film_genre_X.copy()
feat["mean_rating_raw"] = film_mean.reindex(feat.index)
feat["log_n_ratings"] = np.log1p(film_n.reindex(feat.index).fillna(0))
feat["share_F"] = film_female.reindex(feat.index)
for c in age_share.columns:
    feat[c] = age_share[c].reindex(feat.index)

# Films sans assez de notes : moyenne globale / parts neutres (0.5 F, age uniforme)
global_mean = float(train["rating"].mean())
feat["mean_rating_raw"] = feat["mean_rating_raw"].fillna(global_mean)
feat["share_F"] = feat["share_F"].fillna(0.5)
for c in age_share.columns:
    feat[c] = feat[c].fillna(1.0 / len(AGE_BINS))

cluster_feature_cols = list(feat.columns)
film_cluster_X = feat[cluster_feature_cols].astype(float)

# Fit scaler + KMeans sur films vus en train
train_film_ids = train["item_id"].unique()
X_fit_raw = film_cluster_X.loc[film_cluster_X.index.isin(train_film_ids)]
cluster_scaler = StandardScaler()
X_fit = cluster_scaler.fit_transform(X_fit_raw.values)
X_all = cluster_scaler.transform(film_cluster_X.values)

sil_scores = {}
for k in KMEANS_K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_fit)
    if len(set(labels)) < 2:
        continue
    sil_scores[k] = silhouette_score(X_fit, labels)
best_k = max(sil_scores, key=sil_scores.get)
print("Silhouette (clustering enrichi):", {k: round(v, 3) for k, v in sil_scores.items()})
print("K choisi:", best_k)
print(f"Features clustering ({len(cluster_feature_cols)}) : genres + note + log(n) + share_F + parts age")

kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(X_fit)
film_cluster = pd.Series(kmeans.predict(X_all), index=film_cluster_X.index, name="cluster")
print("Taille des clusters (extrait) :")
print(film_cluster.value_counts().sort_index().head())

'''

set_src(10, kmeans_new + "\n" + rest)

set_src(
    11,
    """
## 6. Comment on recommande (en pratique)

Pour chaque methode : partir du film aime A (et du profil si besoin), construire un score, garder le top-k.
**Regle commune :** on n'propose ni A, ni les films deja aimes par l'utilisateur en train.
""",
)

# Charter update - simple language for models
charter = Path(__file__).resolve().parents[1] / "PROJECT_CHARTER.md"
ct = charter.read_text()
old = """## Les trois modèles (sklearn)

| Modèle | Idée | Inférence |
| --- | --- | --- |
| **1. KMeans** | **Modèle de clustering** des films sur les genres 0/1 (`K` via silhouette, seed fixe). Ce n’est **pas** la baseline. | A → son cluster → autres films du cluster ; re-rank avec poids **distincts** âge / sexe ; exclut les films déjà aimés en train |
| **2. Item–item CF** | `NearestNeighbors` cosine sur vecteurs **item × user** (notes train) | Agrège les voisins de **tous** les films aimés train (pondérés par la note) ; exclut déjà aimés |
| **3. LogisticRegression** | Supervisé : proba qu’un candidat soit aimé | Features : âge, sexe, genres A/B, overlap, mean_rating train de B, same_cluster ; rank par `predict_proba` ; exclut déjà aimés |

**Contrôle des poids profil (KMeans) :** `α_age` et `α_sexe` sont séparés (EDA : l’âge pèse plus que le sexe ; ex. `α_age` un peu au-dessus de `α_sexe`). Score ≈ note moyenne + `α_age` · boost_âge + `α_sexe` · boost_sexe. Trop forts → le top se détache du film aimé ; trop faibles → proche d’un ranking mean-in-cluster.
"""

# try flexible replace of the models section
import re
new_models = """## Les méthodes (baseline + modèles)

| Nom | En une phrase | Comment ça choisit les films |
| --- | --- | --- |
| **Baseline** | Règle simple sans ML | Même genre(s) que le film aimé, triés par note moyenne (train). |
| **1. KMeans enrichi** | Regroupe les films qui se ressemblent | Ressemblance = **genres** + **profil du film** (note moyenne, volume de notes, public âge/sexe sur le train). On propose d’autres films du **même groupe**, un peu réordonnés selon l’âge/sexe de l’utilisateur (`α_age`, `α_sexe`). Ce n’est **pas** la baseline. |
| **2. Item–item CF** | « Souvent aimés avec… » | Compare les films via les **notes des utilisateurs**. On agrège les voisins de tous les films déjà aimés (pondérés par la note). |
| **3. LogisticRegression** | Estime la chance qu’un film plaise | Apprend \(P(\\text{aimé} \\mid X)\) avec âge, sexe, genres, overlap, note moyenne, même cluster ; on classe par cette proba. |

Toutes les méthodes **excluent** le film saisi et les films déjà aimés en train.

**Poids profil (KMeans) :** `α_age` et `α_sexe` règlent l’importance de l’âge et du sexe dans le re-ranking (âge un peu plus que le sexe). Trop forts → on s’éloigne du film aimé ; trop faibles → on retombe sur « meilleurs notes du cluster ».
"""

if "## Les trois modèles (sklearn)" in ct:
    ct = re.sub(
        r"## Les trois modèles \(sklearn\).*?(?=\n## Pourquoi)",
        new_models + "\n",
        ct,
        flags=re.S,
    )
    charter.write_text(ct)
    print("charter updated")
else:
    print("charter section not found, writing append skip")

# Also update decision line if it still says voisins de genres
ct = charter.read_text()
ct = ct.replace(
    "trois modèles sklearn** (clustering de films, voisins de genres, régression logistique)",
    "trois modèles sklearn** (clustering enrichi de films, filtrage collaboratif item–item, régression logistique)",
)
charter.write_text(ct)

nb_obj = from_dict(nb)
normalize(nb_obj)
path.write_text(json.dumps(nb_obj, indent=1, ensure_ascii=False) + "\n")
print("notebook patched")
