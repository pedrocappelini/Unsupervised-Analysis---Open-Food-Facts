# UAOFF - Open Food Facts Clustering

Unsupervised clustering of a 300k-row sample of the
[Open Food Facts](https://world.openfoodfacts.org/) dataset. The goal is to
group food products by their nutritional profile (per 100g) and to study how
well those nutritionally-driven groups agree with the real food categories
(`pnns_groups_1`).

Three clustering algorithms are applied and compared:

- **K-Means** (K chosen from a 2-12 sweep of Silhouette / Davies-Bouldin /
  Calinski-Harabasz, not just the elbow).
- **DBSCAN** (eps chosen from a k-distance plot, with a `eps` x `min_samples`
  grid search on `RobustScaler`-scaled features).
- **Agglomerative / Hierarchical clustering** (Ward linkage).

Results are compared with the **Silhouette Score** (cohesion) and the
**Adjusted Rand Index** (agreement with the real `pnns_groups_1` category).

---

## Key findings

An **ARI of roughly 0.24-0.27** is the headline result. It shows that nutrient
values alone only *loosely* track the official food categories - they cannot,
for example, cleanly separate "Composite foods" from "Fish Meat Eggs".
Nutritional composition is an imperfect signature of gastronomic category.

- **Hierarchical** achieves the best ARI.
- **K-Means** is close behind and is easier to interpret.
- **DBSCAN** finds no usable structure (most points are a single blob, the
  rest are noise), which is consistent with the density of this data.

---

## Project structure

```
uaoff/
├── src/uaoff/__init__.py       # The whole pipeline, in one documented file
├── notebooks/
│   └── Trabalho2.ipynb          # Rebuilt notebook that imports from uaoff
├── plots/                       # Generated figures (git-ignored)
├── data/                        # Downloaded dataset (git-ignored)
├── pyproject.toml               # Dependencies + ``uaoff`` entry point
└── README.md
```

All the logic lives in a single module, `src/uaoff/__init__.py`, split into
clearly-commented sections:

1. Load (`load_data`)
2. Preprocess (`preprocess`)
3. Normalise + PCA for visualisation (`reduce_dims`)
4. Clustering (`kmeans`, `dbscan`, `hierarchical`)
5. Cluster profiles (`cluster_profiles`)
6. Comparison (`compare`) and a written `conclusion`

---

## Setup

```bash
uv sync
```

This creates a virtual environment and installs every dependency declared in
`pyproject.toml`:

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `gdown`,
`ipykernel`.

### Requirements

- Python `>= 3.14` (see `.python-version`).
- `uv` (the project manager).

---

## Running the analysis

### As a script

```bash
uv run uaoff
```

or

```bash
uv run python -m uaoff
```

This downloads the dataset if it is missing, runs the full pipeline, and saves
every figure to `plots/`. It also prints per-cluster top categories, a final
comparison table, per-cluster mean-nutrient profiles, and a written conclusion.

### As a notebook

```bash
uv run jupyter notebook notebooks/Trabalho2.ipynb
```

The notebook imports the same functions from the `uaoff` package, so its logic
is identical to the script version.

---

## Data source and sampling

The dataset is the Open Food Facts product database, a ~300k-row sample.
**Right now it is loaded from a personal Google Drive link** via `gdown` (the
same file used in the original Colab notebook). We do **not** commit the file to
this repository:

- It is large, and Git is not a good place for large CSV artefacts.
- The download is automated, so anyone cloning the repo can reproduce the
  analysis with a single command.

On the first run the file is downloaded to `data/openfoodfacts_300k.csv` (this
directory is git-ignored). Later runs reuse the cached file.

To point at a different source, edit the top of `src/uaoff/__init__.py`:

- `GDRIVE_FILE_ID` - the Google Drive file ID.
- `RAW_CSV` - where the file is stored.
- `load_data()` - replace `gdown.download(...)` with `pd.read_csv(...)` for a
  local file.

---

## Reproducibility

A global random seed is used throughout:

- `RANDOM_STATE = 42` (K-Means, PCA, the DBSCAN/Hierarchical sampling).

This makes the clustering results reproducible across runs on the same data.
Note that the dataset itself is a fixed sample; changing the source file will
change the results.

---

## Methodology

### Preprocessing

1. Keep `product_name`, `brands`, `pnns_groups_1` plus all `*_100g` nutrient
   columns.
2. Drop rows with no name/category, or category `unknown`.
3. Drop nutrient columns filled in fewer than 15% of rows.
4. Fill remaining nutrient `NaN`s with `0` (see limitations).
5. Drop rows with an empty nutritional profile, and spam-like short
   names/brands.
6. Drop duplicate `(name, brand)` pairs and redundant energy columns.
7. Remove rows where theoretical calories (computed from macronutrients)
   contradict the reported energy.
8. Drop rows with any nutrient above 100g per 100g, or energy above 900 kcal.
9. Require at least 3 non-zero nutrients per row.
10. Remove multivariate outliers using a robust z-score (median/IQR) threshold.

### Dimensionality reduction

Features are standardised with `StandardScaler`, then projected to 2 dimensions
with PCA. **PCA is used only for the 2D scatter plots**; the clustering itself
runs on all scaled nutrient features (11 in the final dataset). The code prints
`explained_variance_ratio_` so the reader can see how much variance the 2D
projection discards.

### Clustering

- **K-Means**: K is selected from a sweep of 2 to 12 using Silhouette,
  Davies-Bouldin, and Calinski-Harabasz in addition to the elbow.
- **DBSCAN**: features are scaled with `RobustScaler` (better for zero-heavy,
  skewed nutrient data); `eps` is informed by the k-distance plot and a
  `eps` x `min_samples` grid search.
- **Hierarchical**: Ward linkage, K matched to K-Means for a fair comparison.

### Evaluation

- **Silhouette Score** measures cluster cohesion. For DBSCAN, it is computed
  **excluding** the noise label (`-1`), and the noise percentage is reported
  separately.
- **ARI** measures agreement with the real categories; for DBSCAN it is also
  computed excluding noise.

---

## Outputs

The script saves the following figures to `plots/`:

- `kmeans_elbow.png` - inertia vs K.
- `kmeans_metrics_vs_k.png` - Silhouette / Davies-Bouldin / Calinski-Harabasz
  vs K.
- `kmeans_clusters.png`, `hierarchical_clusters.png`, `dbscan_clusters.png` -
  the 2D PCA projection coloured by cluster.
- `dbscan_kdistance.png` - k-distance curve with the detected knee.
- `hierarchical_dendrogram.png` - the dendrogram.
- `kmeans_cluster_profiles.png`, `hierarchical_cluster_profiles.png` -
  heatmaps of the mean nutrient profile per cluster.

The `plots/` directory is git-ignored, so nothing large is committed.

---

## Limitations

- **`NaN = 0 g`**: missing nutrient values are assumed to mean "0 g", which can
  understate products with incomplete fields.
- **PCA lossiness**: the clustering runs on all scaled features, but the 2D
  PCA plots discard variance and are for visualisation only.
- **Sample scope**: the ~300k-row sample may not represent the whole Open Food
  Facts database.
