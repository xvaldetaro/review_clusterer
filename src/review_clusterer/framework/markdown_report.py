from pathlib import Path
from datetime import datetime


def generate_cluster_report(clusters: list, csv_file_path: Path, output_path: Path = None) -> Path:
    if output_path is None:
        output_path = csv_file_path.parent / f"{csv_file_path.stem}_cluster_report.md"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        f"# Cluster Analysis Report for {csv_file_path.name}\n",
        f"*Generated on: {now}*\n",
        f"## Overview\n",
        f"- **Source file**: {csv_file_path}\n",
        f"- **Total clusters**: {len(clusters)}\n",
        f"- **Total reviews**: {sum(cluster['review_count'] for cluster in clusters)}\n\n"
    ]

    report.append("## Clusters (sorted by average rating, worst to best)\n")

    for i, cluster in enumerate(clusters):
        report.append(
            f"### Cluster {i + 1}/{len(clusters)} (ID: {cluster['id']})\n"
        )
        report.append(
            f"- **Reviews**: {cluster['review_count']}\n"
            f"- **Mean distance**: {cluster['mean_distance']:.4f}\n"
            f"- **Average rating**: {cluster['avg_rating']:.1f}/5\n\n"
        )

        report.append("#### Most Representative Reviews\n")
        report.append("| ID | Rating | Distance | Title | Content |\n")
        report.append("| --- | --- | --- | --- | --- |\n")

        central_reviews = cluster["reviews"][:5]

        for review in central_reviews:
            review_id = review["id"]
            try:
                rating = f"{float(review.get('review_rating', 0)):.1f}/5"
            except (ValueError, TypeError):
                rating = "N/A"

            distance = f"{review.get('distance_from_center', 0):.4f}"
            title = review.get("review_title", "").replace("|", "\\|").replace("\n", " ")
            content = review.get("review_details", "").replace("|", "\\|").replace("\n", " ")

            if len(content) > 100:
                content = content[:97] + "..."

            report.append(f"| {review_id} | {rating} | {distance} | {title} | {content} |\n")
        
        report.append("\n")

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
    output_path = generate_cluster_report(clusters, csv_file_path, output_path)
    
    if unclustered_reviews:
        with open(output_path, "a") as f:
            f.write(f"## Unclustered Reviews\n")
            f.write(f"*{len(unclustered_reviews)} reviews were not assigned to any cluster*\n\n")
            
            f.write("| ID | Rating | Outlier Score | Title | Content |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            
            reviews_to_display = unclustered_reviews[:limit]
            
            for review in reviews_to_display:
                review_id = review["id"]
                try:
                    rating = f"{float(review.get('review_rating', 0)):.1f}/5"
                except (ValueError, TypeError):
                    rating = "N/A"
                
                outlier_score = f"{review.get('outlier_score', 0):.4f}" if "outlier_score" in review else "N/A"
                title = review.get("review_title", "").replace("|", "\\|").replace("\n", " ")
                content = review.get("review_details", "").replace("|", "\\|").replace("\n", " ")
                
                if len(content) > 100:
                    content = content[:97] + "..."
                
                f.write(f"| {review_id} | {rating} | {outlier_score} | {title} | {content} |\n")
    
    return output_path