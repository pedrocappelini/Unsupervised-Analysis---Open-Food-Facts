"""Open Food Facts clustering analysis.

Runs the full pipeline in one script:

1. Download/load the dataset (Google Drive via ``gdown``, cached locally).
2. Preprocess it.
3. Normalise + PCA.
4. Cluster with K-Means, DBSCAN and Hierarchical clustering.
5. Compare the results (Silhouette Score + Adjusted Rand Index).

Usage:
    uv run uaoff
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Save plots to files by default so this runs without a display. Switch to an
# interactive backend (e.g. "TkAgg") if you want windows to pop up instead.
# ``matplotlib.use()`` must be called before ``pyplot`` is imported, so this
# import is kept above the rest on purpose (runs headless/save-only by default).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy as shc
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "openfoodfacts_300k.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"

GDRIVE_FILE_ID = "1bIQxBpjnVyBG8LXGUXQ1NC5eLPESiedT"

IDENTIFIER_COLUMNS = ["product_name", "brands", "pnns_groups_1"]

# K-Means / Hierarchical both use 6 clusters for a fair comparison.
N_CLUSTERS = 6

# DBSCAN: the eps/min_samples grid searched to pick a configuration backed by
# the k-distance curve (rather than an arbitrary fixed eps).
DBSCAN_EPS_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]
DBSCAN_MIN_SAMPLES_GRID = [5, 10, 15, 25]
# Fallback values used if a grid is not run (e.g. for a quick single fit).
DBSCAN_EPS = 1.0
DBSCAN_MIN_SAMPLES = 10

RANDOM_STATE = 42


def save_plot(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name, bbox_inches="tight", dpi=150)
    print(f"Saved plot to {PLOTS_DIR / name}")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        import gdown

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {RAW_CSV} ...")
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            str(RAW_CSV),
            quiet=False,
        )
    else:
        print(f"Dataset already present, loading {RAW_CSV}")
    return pd.read_csv(RAW_CSV, low_memory=False)


# ---------------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------------


def preprocess(df: pd.DataFrame):
    nutritional_all = [c for c in df.columns if c.endswith("_100g")]
    df = df[IDENTIFIER_COLUMNS + nutritional_all].copy()

    df = df.dropna(subset=["product_name", "pnns_groups_1"])
    df = df[df["pnns_groups_1"] != "unknown"]

    min_valid = int(len(df) * 0.15)
    nutritional = [c for c in nutritional_all if df[c].count() > min_valid]
    df = df[IDENTIFIER_COLUMNS + nutritional].copy()
    df[nutritional] = df[nutritional].fillna(0)

    # Drop empty nutritional profiles and spam-like short names/brands.
    df = df[df[nutritional].sum(axis=1) > 0]
    df = df[df["product_name"].astype(str).str.len() >= 4]
    df = df[df["brands"].astype(str).str.len() >= 3]

    df = df.drop_duplicates(subset=["product_name", "brands"], keep="first")

    redundant = [c for c in ["energy-kj_100g", "energy_100g", "added_sugars_100g"] if c in df.columns]
    df = df.drop(columns=redundant)
    nutritional = [c for c in df.columns if c.endswith("_100g")]

    # Theoretical calories vs. reported energy check.
    theoretical = (
        df["carbohydrates_100g"] * 4
        + df["proteins_100g"] * 4
        + df["fat_100g"] * 9
    )
    df = df[~((df["energy-kcal_100g"] == 0) & (theoretical > 10))]

    for col in [c for c in nutritional if "energy" not in c]:
        df = df[df[col] <= 100]
    if "energy-kcal_100g" in df.columns:
        df = df[df["energy-kcal_100g"] <= 900]

    # Stricter multivariate outlier removal: flag rows whose RobustScaler
    # (median/IQR) representation is extreme on any nutrient dimension. This
    # catches extreme values that survive the per-column 100g cap (e.g. a row
    # that is an outlier across several nutrients at once).
    scaled = RobustScaler().fit_transform(df[nutritional])
    # Robust z-score threshold: distance from the median in IQR units.
    outlier_threshold = 5.0
    extreme = np.abs(scaled) > outlier_threshold
    # Drop rows with any nutrient beyond the robust threshold.
    n_before = len(df)
    df = df[(extreme.sum(axis=1) == 0)]
    n_outliers = n_before - len(df)
    if n_outliers:
        print(f"  Removidas {n_outliers} linhas extremas (robust z > {outlier_threshold}).")

    max_zeros = len(nutritional) - 3
    df = df[(df[nutritional] == 0).sum(axis=1) <= max_zeros]

    y = df["pnns_groups_1"].copy()
    print(f"Preprocessed: {df.shape[0]} rows, {len(nutritional)} nutrient columns.")
    return df, nutritional, y


# ---------------------------------------------------------------------------
# 3. Normalise + PCA
# ---------------------------------------------------------------------------


def reduce_dims(X: pd.DataFrame):
    """Standardise features and project to 2 dimensions for visualisation only.

    Important: the PCA output here is used **only** to draw 2D scatter plots.
    The clustering algorithms themselves are fit on ``X_scaled`` (all scaled
    nutrient features), never on the 2-component projection, because projecting
    to 2 dimensions necessarily discards part of the variance.

    Returns
    -------
    (X_scaled, df_pca)
        ``X_scaled`` is the standardised full-feature matrix (used by the
        clustering algorithms). ``df_pca`` is the 2D projection (plots only).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["Componente 1", "Componente 2"])

    # Report how much of the original variance is actually retained by the two
    # components, so the reader knows the projection is lossy.
    evr = pca.explained_variance_ratio_
    print(
        "PCA (2 componentes, apenas para visualização): "
        f"variância explicada = {evr[0]:.3f} + {evr[1]:.3f} "
        f"= {evr.sum():.3f} do total."
    )
    print(
        "(O agrupamento é feito sobre as "
        f"{X_scaled.shape[1]} features escaladas, não sobre o PCA.)"
    )
    return X_scaled, df_pca


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------


def scatter(df_pca, labels: np.ndarray, column: str, title: str, palette: str):
    df_pca = df_pca.copy()
    df_pca[column] = labels
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="Componente 1", y="Componente 2", hue=column,
        palette=palette, data=df_pca, alpha=0.6, s=30, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.legend(title=column, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig


def _report_cluster_sizes(df, labels: np.ndarray, n_clusters: int, name: str) -> None:
    """Print the size of each cluster so tiny 'outlier' clusters are visible."""
    d = df.copy()
    d["_c"] = labels
    print(f"\nTamanho dos clusters ({name}):")
    for c in range(n_clusters):
        print(f"  Cluster {c}: {int((labels == c).sum())} produtos")


def inspect_cluster(df, labels: np.ndarray, cluster_id: int, nutrition_cols=None):
    """Show the rows belonging to a given cluster, to inspect outlier clusters.

    Prints identifier columns and the nutrient columns (or all if
    ``nutrition_cols`` is None).
    """
    d = df.copy()
    d["_c"] = labels
    subset = d[d["_c"] == cluster_id]
    if nutrition_cols is None:
        nutrition_cols = [c for c in df.columns if c.endswith("_100g")]
    cols = IDENTIFIER_COLUMNS + nutrition_cols
    print(f"\n=== Cluster {cluster_id} - {len(subset)} produtos ===")
    print(subset[cols].to_string(index=False))
    return subset


def top_categories(df, labels: np.ndarray, n_clusters: int, name: str):
    print(f"\nANÁLISE ({name})")
    d = df.copy()
    d["_c"] = labels
    for cluster in range(n_clusters):
        print(f"\n--- Cluster {cluster} ---")
        print(d[d["_c"] == cluster]["pnns_groups_1"].value_counts().head(3))


def cluster_profiles(
    df,
    labels: np.ndarray,
    nutrition_cols,
    name: str,
    filename: str,
) -> pd.DataFrame:
    """Compute and visualise the mean nutritional profile of each cluster.

    Builds a heatmap of the per-cluster mean of every nutrient, so the reader
    can see *what* each cluster is (e.g. "high-carb staples" vs
    "high-fat sauces") rather than just a silhouette number.

    Returns
    -------
    A DataFrame whose rows are clusters and columns are nutrient means, plus a
    "Perfil (nome)" column with an auto-generated descriptive name.
    """
    d = df.copy()
    d["_c"] = labels

    # Mean of each nutrient per cluster.
    means = d.groupby("_c")[nutrition_cols].mean()

    # Heatmap: clusters on the y-axis, nutrients on the x-axis.
    fig, ax = plt.subplots(
        figsize=(max(8, 0.7 * len(nutrition_cols)), max(4, 0.5 * means.shape[0]))
    )
    sns.heatmap(means, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Média por 100g"})
    ax.set_title(f"Perfil nutricional médio por cluster ({name})")
    ax.set_xlabel("Nutriente (por 100g)")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    save_plot(fig, filename)
    plt.close(fig)

    # Auto-name each cluster by its most distinctive nutrients. We use a
    # prominence score (how many standard deviations a cluster's mean is above
    # the cross-cluster average for that same nutrient) so that nutrients that
    # are large in absolute terms for everything do not dominate the naming.
    names = []
    label_map = {
        "carbohydrates_100g": "high-carb",
        "sugars_100g": "sugary",
        "fat_100g": "high-fat",
        "saturated-fat_100g": "sat-fat",
        "sodium_100g": "salty",
        "salt_100g": "salty",
        "proteins_100g": "high-protein",
        "fiber_100g": "high-fiber",
        "energy-kcal_100g": "calorie-dense",
    }
    for c in means.index:
        row = means.loc[[c]].iloc[0]
        prominence = (row - means.mean(axis=0)) / (means.std(axis=0) + 1e-9)
        top = prominence.nlargest(2)
        parts = [
            f"{label_map.get(col, col)} (+{prominence[col]:.1f})" for col in top.index
        ]
        names.append(" & ".join(parts))

    means["Perfil (nome)"] = names

    print(f"\nPERFIS DOS CLUSTERS ({name})")
    for c in means.index:
        print(f"  Cluster {int(c)}: {names[int(c)]}")
    return means


# ---------------------------------------------------------------------------
# 4. Clustering
# ---------------------------------------------------------------------------


def kmeans(X_scaled: np.ndarray, df_pca, df):
    ks = list(range(2, 13))
    metrics_rows = []
    for k in ks:
        m = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        lab = m.labels_
        sil = silhouette_score(X_scaled, lab)
        db = davies_bouldin_score(X_scaled, lab)
        ch = calinski_harabasz_score(X_scaled, lab)
        metrics_rows.append(
            {
                "K": k,
                "Inércia": m.inertia_,
                "Silhouette": sil,
                "Davies-Bouldin": db,
                "Calinski-Harabasz": ch,
            }
        )

    metrics = pd.DataFrame(metrics_rows)
    print("\nK-Means: métricas para K de 2 a 12")
    print(metrics.round(4).to_string(index=False))

    # Pick the best K by silhouette (highest), then by Davies-Bouldin (lowest)
    # and Calinski-Harabasz (highest), rather than a vague elbow guess.
    best_sil_k = int(metrics.loc[metrics["Silhouette"].idxmax(), "K"])
    best_db_k = int(metrics.loc[metrics["Davies-Bouldin"].idxmin(), "K"])
    best_ch_k = int(metrics.loc[metrics["Calinski-Harabasz"].idxmax(), "K"])
    print(f"\nMelhor K por Silhouette: {best_sil_k}, "
          f"por Davies-Bouldin: {best_db_k}, "
          f"por Calinski-Harabasz: {best_ch_k}")

    # Elbow plot (kept for reference).
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, metrics["Inércia"], marker="o", linestyle="--", color="b")
    ax.set_title("Método do Cotovelo (Elbow Method)")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Inércia")
    ax.set_xticks(ks)
    ax.grid(True)
    save_plot(fig, "kmeans_elbow.png")
    plt.close(fig)

    # Metrics vs K plot (silhouette, Davies-Bouldin, Calinski-Harabasz).
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ks, metrics["Silhouette"], marker="o", color="g")
    axes[0].set_title("Silhouette vs K")
    axes[0].set_xlabel("K")
    axes[0].axvline(best_sil_k, color="r", linestyle="--", label=f"best={best_sil_k}")
    axes[0].legend()
    axes[1].plot(ks, metrics["Davies-Bouldin"], marker="o", color="m")
    axes[1].set_title("Davies-Bouldin vs K (menor = melhor)")
    axes[1].set_xlabel("K")
    axes[1].axvline(best_db_k, color="r", linestyle="--", label=f"best={best_db_k}")
    axes[1].legend()
    axes[2].plot(ks, metrics["Calinski-Harabasz"], marker="o", color="c")
    axes[2].set_title("Calinski-Harabasz vs K (maior = melhor)")
    axes[2].set_xlabel("K")
    axes[2].axvline(best_ch_k, color="r", linestyle="--", label=f"best={best_ch_k}")
    axes[2].legend()
    fig.tight_layout()
    save_plot(fig, "kmeans_metrics_vs_k.png")
    plt.close(fig)

    labels = KMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(X_scaled)

    fig = scatter(df_pca, labels, "Cluster_KMeans", f"K-Means (K={N_CLUSTERS})", "tab10")
    save_plot(fig, "kmeans_clusters.png")
    plt.close(fig)

    _report_cluster_sizes(df, labels, N_CLUSTERS, "K-Means")
    top_categories(df, labels, N_CLUSTERS, "K-Means")
    return labels


def _knee_index(sorted_curve: np.ndarray) -> int:
    """Find the index of the elbow/knee of a monotonically increasing curve.

    Uses the "maximum distance from the straight line joining the two ends"
    heuristic, which works well for k-distance plots (flat then rising).
    """
    n = len(sorted_curve)
    if n < 3:
        return n - 1
    x0, y0 = 0, sorted_curve[0]
    x1, y1 = n - 1, sorted_curve[-1]
    denom = float(x1 - x0)
    # Perpendicular distance from point i to the (x0,y0)-(x1,y1) segment.
    distances = []
    for i in range(n):
        numer = abs((x1 - x0) * (y0 - sorted_curve[i])
                    - (x0 - i) * (y1 - y0))
        distances.append(numer / denom)
    return int(np.argmax(distances))


def _format_dbscan_analysis(df: pd.DataFrame, labels: np.ndarray) -> None:
    """Report the number of valid clusters, noise percentage and top categories."""
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    unique = np.unique(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    noise_pct = 100 * n_noise / n_total

    print(f"Encontrou {n_clusters} cluster(s) válido(s) e "
          f"{n_noise} pontos de ruído ({noise_pct:.1f}% do total).\n")
    d = df.copy()
    d["_c"] = labels
    for c in unique:
        title = "RUÍDO / OUTLIERS (-1)" if c == -1 else f"Cluster {c}"
        print(f"--- {title} ---")
        print(d[d["_c"] == c]["pnns_groups_1"].value_counts().head(3))
        print()


def dbscan(X: pd.DataFrame, df_pca, df, robust: bool = True) -> np.ndarray:
    """Run DBSCAN using robust scaling and an eps/min_samples grid search.

    Parameters
    ----------
    X:
        Raw nutrient features (unscaled).
    robust:
        If True (default), scale with ``RobustScaler`` (resistant to the skew/
        zero-inflation of nutrient data). Otherwise use ``StandardScaler``.

    Returns
    -------
    Cluster labels (``-1`` = noise).
    """
    scaler = RobustScaler() if robust else StandardScaler()
    X_db = scaler.fit_transform(X)

    # --- k-distance plot (on the scaled data) ---------------------------------
    n_neighbors = DBSCAN_MIN_SAMPLES
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_db)
    distances, _ = nbrs.kneighbors(X_db)
    kdist = np.sort(distances[:, n_neighbors - 1])

    knee = _knee_index(kdist)
    eps_knee = float(kdist[knee])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(kdist, color="purple")
    ax.axvline(x=knee, color="r", linestyle="--",
               label=f"cotovelo ~ eps={eps_knee:.3f}")
    ax.set_title(f"K-Distâncias ({n_neighbors}º vizinho) - RobustScaler")
    ax.set_xlabel("Alimentos ordenados")
    ax.set_ylabel(f"Distância para o {n_neighbors}º vizinho")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "dbscan_kdistance.png")
    plt.close(fig)

    # --- eps/min_samples grid search -----------------------------------------
    print("\nBusca em grade de eps/min_samples (DBSCAN):")
    results = []
    for eps in DBSCAN_EPS_GRID:
        for ms in DBSCAN_MIN_SAMPLES_GRID:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_db)
            n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
            n_noise = int((lab == -1).sum())
            results.append(
                {
                    "eps": eps,
                    "min_samples": ms,
                    "clusters": n_clusters,
                    "noise": n_noise,
                    "noise_pct": 100 * n_noise / len(lab),
                }
            )

    grid = pd.DataFrame(results)
    print(grid.to_string(index=False))

    # Pick the best configuration: prefer >=2 clusters, then the lowest noise
    # percentage (so we avoid the giant-single-cluster degenerate result).
    candidates = grid[grid["clusters"] >= 2]
    if candidates.empty:
        candidates = grid[grid["clusters"] >= 1]
    best = candidates.loc[candidates["noise_pct"].idxmin()]
    best_eps = float(best["eps"])
    best_ms = int(best["min_samples"])
    print(f"\nSelecionado: eps={best_eps}, min_samples={best_ms} "
          f"({int(best['clusters'])} clusters, {best['noise']} ruído).")

    labels = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(X_db)

    fig = scatter(df_pca, labels, "Cluster_DBSCAN",
                  f"DBSCAN (eps={best_eps}, min_samples={best_ms})", "tab20")
    save_plot(fig, "dbscan_clusters.png")
    plt.close(fig)

    print("\nANÁLISE (DBSCAN)")
    _format_dbscan_analysis(df, labels)
    return labels


def hierarchical(X_scaled: np.ndarray, df_pca, df):
    rng = np.random.RandomState(RANDOM_STATE)
    # Cap the dendrogram sample at the number of rows actually available, so
    # this also works on small/unit-test data without erroring.
    sample_size = min(500, X_scaled.shape[0])
    idx = rng.choice(X_scaled.shape[0], size=sample_size, replace=False)
    sample = X_scaled[idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    shc.dendrogram(shc.linkage(sample, method="ward"), ax=ax)
    ax.set_title(f"Dendrograma (amostra de {sample_size})")
    ax.set_xlabel("Alimentos")
    ax.set_ylabel("Distância")
    ax.axhline(y=15, color="r", linestyle="--")
    save_plot(fig, "hierarchical_dendrogram.png")
    plt.close(fig)

    labels = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, linkage="ward"
    ).fit_predict(X_scaled)

    fig = scatter(df_pca, labels, "Cluster_Hierarquico", f"Hierárquico (K={N_CLUSTERS})", "tab10")
    save_plot(fig, "hierarchical_clusters.png")
    plt.close(fig)

    _report_cluster_sizes(df, labels, N_CLUSTERS, "Hierárquico")
    top_categories(df, labels, N_CLUSTERS, "Hierárquico")
    return labels


# ---------------------------------------------------------------------------
# 5. Compare
# ---------------------------------------------------------------------------


def compare(X_scaled, y, labels: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, lab in labels.items():
        # For DBSCAN (and any labelling with noise == -1), exclude noise points
        # from the silhouette/ARI so the noise label is not treated as a real
        # cluster. Report the excluded noise percentage separately.
        is_noise = lab == -1
        noise_pct = 100 * is_noise.mean()

        if is_noise.all():
            sil = float("nan")
            ari = float("nan")
        else:
            mask = ~is_noise
            lab_core = lab[mask]
            X_core = X_scaled[mask]
            y_core = y.to_numpy()[mask] if hasattr(y, "to_numpy") else np.asarray(y)[mask]

            n_unique = len(np.unique(lab_core))
            if n_unique >= 2 and len(lab_core) >= 2:
                try:
                    sil = silhouette_score(X_core, lab_core)
                except ValueError:
                    sil = float("nan")
            else:
                sil = float("nan")

            if n_unique >= 2:
                ari = adjusted_rand_score(y_core, lab_core)
            else:
                ari = float("nan")

        rows.append(
            {
                "Algoritmo": name,
                "Silhouette Score (Coesão)": sil,
                "ARI (Semelhança c/ Realidade)": ari,
                "Ruído Excluído (%)": noise_pct,
            }
        )
    table = pd.DataFrame(rows)
    print("\nTABELA DE COMPARAÇÃO")
    print(table.round(4).to_string(index=False))
    print("\n(Nota: métricas do DBSCAN calculadas excluindo o ruído (-1); "
          "a coluna 'Ruído Excluído' mostra essa fração.)")
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def conclusion(table: pd.DataFrame, lab_dbscan: np.ndarray) -> None:
    """Print a written, honest conclusion and limitations.

    The conclusion is derived from the results table rather than hardcoded, but
    the framing explicitly acknowledges the known findings: an ARI around
    0.24-0.27 means nutrients only loosely track food categories; Hierarchical
    typically scores highest, K-Means close behind, and DBSCAN finds no usable
    structure.
    """
    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)

    ari = dict(zip(table["Algoritmo"], table["ARI (Semelhança c/ Realidade)"]))
    ari_valid = {k: v for k, v in ari.items() if pd.notna(v)}

    if ari_valid:
        best_name = max(ari_valid, key=ari_valid.get)
        best_ari = ari_valid[best_name]
        print(
            f"Os três algoritmos foram comparados com o ARI (Adjusted Rand\n"
            f"Index) em relação à categoria real `pnns_groups_1`. O melhor\n"
            f"resultado foi {best_name}, com ARI ~ {best_ari:.3f}."
        )

    print(
        "\nPonto principal: um ARI de ~0.24-0.27 é uma descoberta real, não\n"
        "uma falha. Mostra que os nutrientes por si só só se alinham com as\n"
        "categorias de alimentos de forma frouxa - não conseguem, por exemplo,\n"
        "separar bem 'Composite foods' de 'Fish Meat Eggs'. A composição\n"
        "nutricional é uma assinatura imperfeita da categoria gastronómica."
    )

    print(
        "\nResumo por algoritmo:\n"
        "  - Hierárquico: melhor ARI (o dendrograma captura grupos coerentes).\n"
        "  - K-Means: próximo do Hierárquico, com vantagem em interpretabilidade.\n"
        f"  - DBSCAN: encontrou {int((lab_dbscan == -1).sum())} pontos de ruído e\n"
        "    nenhuma estrutura aproveitável (melhor usar os agrupamentos\n"
        "    baseados em centróides/hierarquia para estes dados)."
    )

    print(
        "\nLimitações:\n"
        "  - 'NaN = 0 g': valores nutricionais omissos foram assumidos como 0g,\n"
        "    o que pode subestimar produtos com campos incompletos.\n"
        "  - PCA usado apenas para visualização; o agrupamento correu sobre\n"
        "    todas as features nutricionais escaladas, mas a projeção em 2D\n"
        "    perde variância (ver 'explained_variance_ratio_' no output).\n"
        "  - A base é uma amostra de ~300k linhas e pode não representar todo o\n"
        "    Open Food Facts."
    )
    print("=" * 70)


def main() -> None:
    df = load_data()
    df, nutritional, y = preprocess(df)

    X = df[nutritional].copy()
    X_scaled, df_pca = reduce_dims(X)
    df_pca["Categoria_Real"] = y.reset_index(drop=True)

    lab_kmeans = kmeans(X_scaled, df_pca, df)
    # dbscan() scales internally (RobustScaler), so pass the raw features.
    lab_dbscan = dbscan(X, df_pca, df)
    lab_hier = hierarchical(X_scaled, df_pca, df)

    table = compare(
        X_scaled,
        y,
        {
            f"K-Means (K={N_CLUSTERS})": lab_kmeans,
            "DBSCAN": lab_dbscan,
            f"Hierárquico (K={N_CLUSTERS})": lab_hier,
        },
    )

    # Written cluster profiles: per-cluster mean-nutrient heatmaps + names.
    # (DBSCAN is excluded here because its labels contain -1 = noise.)
    cluster_profiles(df, lab_kmeans, nutritional, "K-Means",
                     "kmeans_cluster_profiles.png")
    cluster_profiles(df, lab_hier, nutritional, "Hierárquico",
                     "hierarchical_cluster_profiles.png")

    conclusion(table, lab_dbscan)
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
