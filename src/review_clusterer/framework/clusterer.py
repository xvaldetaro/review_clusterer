from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from rich.console import Console
import hdbscan
from umap import UMAP

console = Console()


def determine_optimal_clusters(
    embeddings: List[List[float]], max_clusters: int = 50
) -> Tuple[int, Dict[str, Any]]:
    if len(embeddings) <= 1:
        return 1, {"inertias": [], "silhouette_scores": [], "k_values": []}

    X = np.array(embeddings)

    k_values = range(2, min(max_clusters + 1, len(embeddings)))
    inertias = []
    silhouette_scores_list = []

    for k in k_values:
        if len(embeddings) <= k:
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

        if len(embeddings) > k:
            try:
                silhouette_avg = silhouette_score(X, kmeans.labels_)
                silhouette_scores_list.append(silhouette_avg)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not compute silhouette score for k={k}: {e}[/yellow]"
                )
                silhouette_scores_list.append(0)

    if not inertias:
        return min(3, len(embeddings)), {
            "inertias": [],
            "silhouette_scores": [],
            "k_values": [],
        }

    inertia_diffs = np.diff(inertias)
    inertia_diffs_normalized = inertia_diffs / np.abs(inertias[:-1])
    elbow_index = (
        np.argmax(inertia_diffs_normalized) if len(inertia_diffs_normalized) > 0 else 0
    )

    silhouette_index = (
        np.argmax(silhouette_scores_list) if silhouette_scores_list else 0
    )

    if silhouette_scores_list and abs(elbow_index - silhouette_index) <= 1:
        optimal_k = list(k_values)[silhouette_index]
    elif silhouette_scores_list:
        optimal_k = list(k_values)[silhouette_index]
    else:
        optimal_k = list(k_values)[elbow_index]

    visualization_data = {
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "k_values": list(k_values),
    }

    return optimal_k, visualization_data


def plot_elbow_method(visualization_data: Dict[str, Any], optimal_k: int) -> None:
    k_values = visualization_data["k_values"]
    inertias = visualization_data["inertias"]
    silhouette_scores = visualization_data["silhouette_scores"]

    if not k_values or not inertias:
        console.print("[yellow]Not enough data to generate elbow method plot[/yellow]")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=inertias,
            mode="lines+markers",
            name="Inertia",
            line=dict(color="blue"),
        )
    )

    if silhouette_scores:
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=silhouette_scores,
                mode="lines+markers",
                name="Silhouette Score",
                line=dict(color="green"),
                yaxis="y2",
            )
        )

    if optimal_k in k_values:
        optimal_index = k_values.index(optimal_k)
        fig.add_vline(
            x=optimal_k,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal k={optimal_k}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Elbow Method for Optimal k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        template="plotly_white",
        hovermode="x unified",
    )

    if silhouette_scores:
        fig.update_layout(
            yaxis2=dict(
                title="Silhouette Score", anchor="x", overlaying="y", side="right"
            )
        )

    fig.show()


def plot_elbow(reviews_with_embeddings: List[Dict[str, Any]]) -> None:
    embeddings = [review["embedding"] for review in reviews_with_embeddings]
    optimal_k, visualization_data = determine_optimal_clusters(embeddings)
    plot_elbow_method(visualization_data, optimal_k)
    console.print(f"[green]Optimal number of clusters determined: {optimal_k}[/green]")


def cluster_reviews(
    reviews_with_embeddings: List[Dict[str, Any]],
    n_clusters: int,
) -> List[Dict[str, Any]]:
    if not reviews_with_embeddings:
        return []

    embeddings = [review["embedding"] for review in reviews_with_embeddings]
    for i, vec in enumerate(embeddings):
        if np.linalg.norm(vec) < 1e-10:
            print(f"Warning: Vector {i} has near-zero norm")

    embeddings = [vec / (np.linalg.norm(vec) + 1e-10) for vec in embeddings]

    embed_array = np.array(embeddings)
    print(f"Max value in embeddings: {np.max(embed_array)}")
    print(f"Min value in embeddings: {np.min(embed_array)}")
    print(f"Any NaN: {np.isnan(embed_array).any()}")
    print(f"Any Inf: {np.isinf(embed_array).any()}")

    X = np.array(embeddings)
    X = np.clip(X, -100, 100)
    assert not np.isnan(X).any(), "NaNs in embeddings"
    assert not np.isinf(X).any(), "Infinite values in embeddings"

    kmeans = KMeans(
        n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init="auto"
    )
    labels = kmeans.fit_predict(X)

    for i, review in enumerate(reviews_with_embeddings):
        review["cluster"] = int(labels[i])

    centers = kmeans.cluster_centers_

    clusters = {}
    for i, review in enumerate(reviews_with_embeddings):
        cluster_id = review["cluster"]
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "id": cluster_id,
                "reviews": [],
                "center": centers[cluster_id],
            }
        clusters[cluster_id]["reviews"].append(review)

    cluster_results = []
    for cluster_id, cluster in clusters.items():
        reviews = cluster["reviews"]
        center = cluster["center"]

        distances = []
        ratings = []
        for review in reviews:
            embedding = np.array(review["embedding"])
            EPSILON = 1e-8
            norm_center = np.linalg.norm(center) + EPSILON
            norm_embedding = np.linalg.norm(embedding) + EPSILON
            distance = 1 - (np.dot(embedding, center) / (norm_embedding * norm_center))
            review["distance_from_center"] = float(distance)
            distances.append(distance)

            try:
                rating = float(review.get("review_rating", 0))
                ratings.append(rating)
            except (ValueError, TypeError):
                pass

        sorted_reviews = sorted(reviews, key=lambda x: x["distance_from_center"])

        mean_distance = float(np.mean(distances)) if distances else 0
        avg_rating = float(np.mean(ratings)) if ratings else 0

        cluster_results.append(
            {
                "id": cluster_id,
                "review_count": len(reviews),
                "mean_distance": mean_distance,
                "avg_rating": avg_rating,
                "reviews": sorted_reviews,
                "center": center.tolist(),
            }
        )

    cluster_results = sorted(cluster_results, key=lambda x: x["avg_rating"])

    return cluster_results


def hdbscan_cluster_reviews(
    reviews_with_embeddings: List[Dict[str, Any]],
    min_cluster_size: int = 10,
    min_samples: int = 5,
    use_umap: bool = True,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not reviews_with_embeddings:
        return [], []

    embeddings = [review["embedding"] for review in reviews_with_embeddings]

    for i, vec in enumerate(embeddings):
        if np.linalg.norm(vec) < 1e-10:
            print(f"Warning: Vector {i} has near-zero norm")

    embeddings = [vec / (np.linalg.norm(vec) + 1e-10) for vec in embeddings]

    X = np.array(embeddings)
    X = np.clip(X, -100, 100)
    assert not np.isnan(X).any(), "NaNs in embeddings"
    assert not np.isinf(X).any(), "Infinite values in embeddings"

    if use_umap:
        console.print("[green]Reducing dimensionality with UMAP...[/green]")
        X = UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            random_state=42,
        ).fit_transform(X)
        console.print(
            f"[green]Reduced dimensionality to {umap_n_components} components[/green]"
        )

    console.print(
        f"[green]Applying HDBSCAN clustering with min_cluster_size={min_cluster_size}, min_samples={min_samples}...[/green]"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    labels = clusterer.fit_predict(X)

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_outliers = np.sum(labels == -1)

    console.print(
        f"[green]Found {n_clusters} clusters and {n_outliers} outliers[/green]"
    )

    for i, review in enumerate(reviews_with_embeddings):
        review["cluster"] = int(labels[i])
        if hasattr(clusterer, "outlier_scores_"):
            review["outlier_score"] = float(clusterer.outlier_scores_[i])

    clustered_reviews = [r for r in reviews_with_embeddings if r["cluster"] != -1]
    unclustered_reviews = [r for r in reviews_with_embeddings if r["cluster"] == -1]

    if not clustered_reviews:
        return [], unclustered_reviews

    clusters = {}
    for review in clustered_reviews:
        cluster_id = review["cluster"]
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "id": cluster_id,
                "reviews": [],
            }
        clusters[cluster_id]["reviews"].append(review)

    cluster_results = []
    for cluster_id, cluster in clusters.items():
        reviews = cluster["reviews"]

        cluster_embeddings = np.array([r["embedding"] for r in reviews])
        center = np.mean(cluster_embeddings, axis=0)

        distances = []
        ratings = []
        for review in reviews:
            embedding = np.array(review["embedding"])
            EPSILON = 1e-8
            norm_center = np.linalg.norm(center) + EPSILON
            norm_embedding = np.linalg.norm(embedding) + EPSILON
            distance = 1 - (np.dot(embedding, center) / (norm_embedding * norm_center))
            review["distance_from_center"] = float(distance)
            distances.append(distance)

            try:
                rating = float(review.get("review_rating", 0))
                ratings.append(rating)
            except (ValueError, TypeError):
                pass

        sorted_reviews = sorted(reviews, key=lambda x: x["distance_from_center"])

        mean_distance = float(np.mean(distances)) if distances else 0
        avg_rating = float(np.mean(ratings)) if ratings else 0

        cluster_results.append(
            {
                "id": cluster_id,
                "review_count": len(reviews),
                "mean_distance": mean_distance,
                "avg_rating": avg_rating,
                "reviews": sorted_reviews,
                "center": center.tolist(),
            }
        )

    cluster_results = sorted(
        cluster_results, key=lambda x: x["avg_rating"], reverse=True
    )

    if unclustered_reviews and "outlier_score" in unclustered_reviews[0]:
        unclustered_reviews = sorted(
            unclustered_reviews, key=lambda x: x.get("outlier_score", 0), reverse=False
        )

    return cluster_results, unclustered_reviews
