Let's start with step 6. Cluster Refinement of ${ai_docs/specs/overview.md}

We have clustering working. And also have a client that calls LLMs successfully for step 6 and 7.

Cluster refinement will start from N buckets of clusters and the N+1 Bucket of unclustered reviews. The desired output is a list of M buckets of very cohesive thoroughly checked and annotated reviews (and no unclustered reviews anymore).

I want you to first give me a sample output of the cluster refinement algorithm. You can get a sample input from my last cluster run in @~/temp/voyageai_report.md
I believe having this sample output (just bits of it for brevity) will help me think about the problem