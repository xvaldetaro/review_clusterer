from pathlib import Path
from datetime import datetime


def generate_cluster_report(clusters: list, csv_file_path: Path, output_path: Path = None) -> Path:
    """
    Generate a markdown report for the clusters.

    Args:
        clusters: List of cluster dictionaries sorted by average rating
        csv_file_path: Path to the original CSV file
        output_path: Path where to save the report (defaults to csv_file_path.stem + '_report.md')

    Returns:
        Path to the generated report file
    """
    if output_path is None:
        output_path = csv_file_path.parent / f"{csv_file_path.stem}_cluster_report.md"

    # Start with a header and timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        f"# Cluster Analysis Report for {csv_file_path.name}\n",
        f"*Generated on: {now}*\n",
        f"## Overview\n",
        f"- **Source file**: {csv_file_path}\n",
        f"- **Total clusters**: {len(clusters)}\n",
        f"- **Total reviews**: {sum(cluster['review_count'] for cluster in clusters)}\n\n"
    ]

    # Generate cluster summaries
    report.append("## Clusters (sorted by average rating, worst to best)\n")

    for i, cluster in enumerate(clusters):
        # Create a cluster section
        report.append(
            f"### Cluster {i + 1}/{len(clusters)} (ID: {cluster['id']})\n"
        )
        report.append(
            f"- **Reviews**: {cluster['review_count']}\n"
            f"- **Mean distance**: {cluster['mean_distance']:.4f}\n"
            f"- **Average rating**: {cluster['avg_rating']:.1f}/5\n\n"
        )

        # Add top reviews table
        report.append("#### Most Representative Reviews\n")
        report.append("| ID | Rating | Distance | Title | Content |\n")
        report.append("| --- | --- | --- | --- | --- |\n")

        # Get the 5 most central reviews
        central_reviews = cluster["reviews"][:5]

        # Add rows for central reviews
        for review in central_reviews:
            review_id = review["id"]
            try:
                rating = f"{float(review.get('review_rating', 0)):.1f}/5"
            except (ValueError, TypeError):
                rating = "N/A"

            distance = f"{review.get('distance_from_center', 0):.4f}"
            title = review.get("review_title", "").replace("|", "\\|").replace("\n", " ")
            content = review.get("review_details", "").replace("|", "\\|").replace("\n", " ")

            # Truncate content if too long
            if len(content) > 100:
                content = content[:97] + "..."

            report.append(f"| {review_id} | {rating} | {distance} | {title} | {content} |\n")
        
        report.append("\n")

    # Write the report to the output file
    with open(output_path, "w") as f:
        f.writelines(report)

    return output_path


def generate_report_with_unclustered(
    clusters: list, 
    unclustered_reviews: list, 
    csv_file_path: Path, 
    output_path: Path = None,
    limit: int = 20
) -> Path:
    """
    Generate a markdown report for the clusters including unclustered reviews.

    Args:
        clusters: List of cluster dictionaries sorted by average rating
        unclustered_reviews: List of unclustered review dictionaries
        csv_file_path: Path to the original CSV file
        output_path: Path where to save the report (defaults to csv_file_path.stem + '_report.md')
        limit: Maximum number of unclustered reviews to include

    Returns:
        Path to the generated report file
    """
    # Generate the main cluster report
    output_path = generate_cluster_report(clusters, csv_file_path, output_path)
    
    # If there are unclustered reviews, append them to the report
    if unclustered_reviews:
        with open(output_path, "a") as f:
            f.write(f"## Unclustered Reviews\n")
            f.write(f"*{len(unclustered_reviews)} reviews were not assigned to any cluster*\n\n")
            
            # Add unclustered reviews table
            f.write("| ID | Rating | Outlier Score | Title | Content |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            
            # Get the top N reviews (by outlier score if available)
            reviews_to_display = unclustered_reviews[:limit]
            
            # Add rows for each review
            for review in reviews_to_display:
                review_id = review["id"]
                try:
                    rating = f"{float(review.get('review_rating', 0)):.1f}/5"
                except (ValueError, TypeError):
                    rating = "N/A"
                
                outlier_score = f"{review.get('outlier_score', 0):.4f}" if "outlier_score" in review else "N/A"
                title = review.get("review_title", "").replace("|", "\\|").replace("\n", " ")
                content = review.get("review_details", "").replace("|", "\\|").replace("\n", " ")
                
                # Truncate content if too long
                if len(content) > 100:
                    content = content[:97] + "..."
                
                f.write(f"| {review_id} | {rating} | {outlier_score} | {title} | {content} |\n")
    
    return output_path