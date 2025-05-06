Read CLAUDE.md and CLAUDE_CHECKLIST.md.

We have already finished all the initial implementations and everything is working. Now I want to create functionality to further analyze the review clusters. I want to use an assistant LLM like Claude, or ChatGPT to read (up to its context size) the clusters, come up with summaries for clusters, mark relevant and irrelevant clusters and then look at unclustered reviews for things that should have been clustered and add them to the clusters.

Now my first question is how should we proceed in terms of the project structure here. I'm afraid keeping the same CLAUDE.md and the context for this whole current project will create more noise than good, but this is my first project with an AI coding assistant so I don't know the best approach.

Should we create a kind an api, or just a SQL table that summarizes the end to end functionality of the `cluster` command. Something like:
Option1

```
interface Review {
  id,created_at,reviewer_name,date,review_title
}
interface Cluster {
  avg_score, mean_distance,
  reviews: List<Review>
}
interface ReviewReport {
  all_reviews: List<Review>,
  clusters: <List<Cluster>>,
  unclustered_reviews: List<Review>,
}
def generate_cluster_report(csv_file: str): ReviewReport
```

Option2: Definition of sql tables schemas for Reviews, Clusters(for the metadata and id) and ClusteredReviews (keys to review and cluster)

Or we should keep going with the same project as is and continue to just ask you to generate more functionality on top of what we have
