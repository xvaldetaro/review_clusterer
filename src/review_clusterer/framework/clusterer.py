from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from rich.console import Console

console = Console()


def determine_optimal_clusters(
    embeddings: List[List[float]], max_clusters: int = 50
) -> Tuple[int, Dict[str, Any]]:
    """
    Determine the optimal number of clusters using the elbow method and silhouette scores.

    Args:
        embeddings: List of embedding vectors
        max_clusters: Maximum number of clusters to consider

    Returns:
        Tuple of (optimal_k, visualization_data)
    """
    if len(embeddings) <= 1:
        return 1, {"inertias": [], "silhouette_scores": [], "k_values": []}

    # Convert embeddings to numpy array
    X = np.array(embeddings)

    # Prepare for elbow method
    k_values = range(2, min(max_clusters + 1, len(embeddings)))
    inertias = []
    silhouette_scores_list = []

    # Calculate inertia and silhouette score for different k values
    for k in k_values:
        # Skip silhouette score calculation if only one sample per cluster would exist
        if len(embeddings) <= k:
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

        # Calculate silhouette score if we have enough samples
        if len(embeddings) > k:
            try:
                silhouette_avg = silhouette_score(X, kmeans.labels_)
                silhouette_scores_list.append(silhouette_avg)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not compute silhouette score for k={k}: {e}[/yellow]"
                )
                silhouette_scores_list.append(0)

    # If we couldn't calculate any values, default to 3 clusters
    if not inertias:
        return min(3, len(embeddings)), {
            "inertias": [],
            "silhouette_scores": [],
            "k_values": [],
        }

    # Find the elbow point (where the rate of decrease sharply changes)
    inertia_diffs = np.diff(inertias)
    inertia_diffs_normalized = inertia_diffs / np.abs(inertias[:-1])
    elbow_index = (
        np.argmax(inertia_diffs_normalized) if len(inertia_diffs_normalized) > 0 else 0
    )

    # Consider silhouette scores too (higher is better)
    silhouette_index = (
        np.argmax(silhouette_scores_list) if silhouette_scores_list else 0
    )

    # Balance between elbow method and silhouette score
    # If they agree, use that value, otherwise prefer silhouette
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
    """
    Generate and display an elbow method plot to visualize the optimal number of clusters.

    Args:
        visualization_data: Dictionary containing k_values, inertias, and silhouette_scores
        optimal_k: The determined optimal number of clusters
    """
    k_values = visualization_data["k_values"]
    inertias = visualization_data["inertias"]
    silhouette_scores = visualization_data["silhouette_scores"]

    if not k_values or not inertias:
        console.print("[yellow]Not enough data to generate elbow method plot[/yellow]")
        return

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add inertia trace
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=inertias,
            mode="lines+markers",
            name="Inertia",
            line=dict(color="blue"),
        )
    )

    # Add silhouette score trace if available
    if silhouette_scores:
        # Create secondary Y axis
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

    # Add vertical line at optimal_k
    if optimal_k in k_values:
        optimal_index = k_values.index(optimal_k)
        fig.add_vline(
            x=optimal_k,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal k={optimal_k}",
            annotation_position="top right",
        )

    # Update layout
    fig.update_layout(
        title="Elbow Method for Optimal k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        template="plotly_white",
        hovermode="x unified",
    )

    # Add secondary y-axis for silhouette score if needed
    if silhouette_scores:
        fig.update_layout(
            yaxis2=dict(
                title="Silhouette Score", anchor="x", overlaying="y", side="right"
            )
        )

    # Show the figure
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

    # Add this check before normalization
    embeddings = [review["embedding"] for review in reviews_with_embeddings]
    # Check for zero vectors before normalization
    for i, vec in enumerate(embeddings):
        if np.linalg.norm(vec) < 1e-10:  # Almost zero
            print(f"Warning: Vector {i} has near-zero norm")

    # Then normalize with protection against division by zero
    embeddings = [vec / (np.linalg.norm(vec) + 1e-10) for vec in embeddings]
    # Determine optimal number of clusters if not specified

    # After normalization, add these checks
    embed_array = np.array(embeddings)
    print(f"Max value in embeddings: {np.max(embed_array)}")
    print(f"Min value in embeddings: {np.min(embed_array)}")
    print(f"Any NaN: {np.isnan(embed_array).any()}")
    print(f"Any Inf: {np.isinf(embed_array).any()}")

    # Perform clustering
    X = np.array(embeddings)
    X = np.clip(X, -100, 100)
    assert not np.isnan(X).any(), "NaNs in embeddings"
    assert not np.isinf(X).any(), "Infinite values in embeddings"

    kmeans = KMeans(
        n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init="auto"
    )
    labels = kmeans.fit_predict(X)

    # Assign cluster labels to reviews
    for i, review in enumerate(reviews_with_embeddings):
        review["cluster"] = int(labels[i])

    # Calculate cluster centers
    centers = kmeans.cluster_centers_

    # Organize reviews by cluster
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

    # Calculate cluster metrics
    cluster_results = []
    for cluster_id, cluster in clusters.items():
        reviews = cluster["reviews"]
        center = cluster["center"]

        # Calculate distances from center
        distances = []
        ratings = []
        for review in reviews:
            embedding = np.array(review["embedding"])
            # Cosine distance
            EPSILON = 1e-8  # You already have this, which is good
            norm_center = np.linalg.norm(center) + EPSILON
            norm_embedding = np.linalg.norm(embedding) + EPSILON
            distance = 1 - (np.dot(embedding, center) / (norm_embedding * norm_center))
            review["distance_from_center"] = float(distance)
            distances.append(distance)

            # Extract rating
            try:
                rating = float(review.get("review_rating", 0))
                ratings.append(rating)
            except (ValueError, TypeError):
                pass

        # Sort reviews by distance from center
        sorted_reviews = sorted(reviews, key=lambda x: x["distance_from_center"])

        # Calculate cluster metrics
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

    # Sort clusters by average rating (ascending, from worst to best)
    cluster_results = sorted(
        cluster_results, key=lambda x: x["avg_rating"]
    )

    return cluster_results
