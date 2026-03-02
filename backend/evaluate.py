"""
evaluate.py — Evaluates the recommender pipeline on the hardcoded train set.
Metric: Mean Recall@10
Uses slug-based URL normalization to handle URL format differences.
"""
import re
from recommender import recommend

# Hardcoded train data (from Gen_AI_Dataset.xlsx Train-Set sheet)
TRAIN_DATA = [
    {
        "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        "relevant_urls": [
            "https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new/",
            "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
            "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
            "https://www.shl.com/solutions/products/product-catalog/view/core-java-advanced-level-new/",
            "https://www.shl.com/products/product-catalog/view/interpersonal-communications/",
        ],
    },
    {
        "query": "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options",
        "relevant_urls": [
            "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-7-1/",
        ],
    },
]


def extract_slug(url: str) -> str:
    """
    Extract the unique assessment slug from a URL.
    e.g. https://www.shl.com/solutions/products/product-catalog/view/java-8-new/
         → 'java-8-new'
         https://www.shl.com/products/catalog/view/java-8-new/
         → 'java-8-new'
    """
    url = url.rstrip("/")
    slug = url.split("/")[-1]
    return slug.lower()


def recall_at_k(relevant: list[str], predicted: list[str], k: int = 10) -> float:
    """
    Compute Recall@K using slug normalization.
    Fraction of relevant items found in top-K predictions.
    """
    if not relevant:
        return 0.0

    relevant_slugs = {extract_slug(u) for u in relevant}
    predicted_slugs = [extract_slug(u) for u in predicted[:k]]

    hits = sum(1 for slug in predicted_slugs if slug in relevant_slugs)
    return hits / len(relevant)


def run_evaluation(verbose: bool = True) -> float:
    """Run evaluation on TRAIN_DATA and return mean Recall@10."""
    scores = []

    for item in TRAIN_DATA:
        results = recommend(item["query"], top_k=10)
        predicted_urls = [r["url"] for r in results]

        relevant_slugs = {extract_slug(u) for u in item["relevant_urls"]}
        predicted_slugs = [extract_slug(u) for u in predicted_urls]

        score = recall_at_k(item["relevant_urls"], predicted_urls, k=10)

        if verbose:
            print(f"\nQuery: {item['query'][:80]}...")
            print(f"Relevant slugs: {relevant_slugs}")
            print(f"\nPredicted URLs ({len(predicted_urls)}):")
            for url, slug in zip(predicted_urls, predicted_slugs):
                hit = "✅" if slug in relevant_slugs else "  "
                print(f"  {hit} {url}")
            print(f"Recall@10: {score:.4f}")

        scores.append(score)

    mean_score = sum(scores) / len(scores) if scores else 0.0

    if verbose:
        print(f"\n{'='*50}")
        print(f"Mean Recall@10: {mean_score:.4f}")
        print(f"{'='*50}")

    return mean_score


if __name__ == "__main__":
    run_evaluation()
