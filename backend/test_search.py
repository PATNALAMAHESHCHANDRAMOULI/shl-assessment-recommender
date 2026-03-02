"""
test_search.py — Quick test to verify vector search and evaluate recall
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).parent / "chroma_db"

def test():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection("shl_assessments")
    print(f"Collection count: {collection.count()}")

    query = "Java developers collaboration business teams 40 minutes"
    emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[emb], n_results=15)

    print(f"\nTop results for: '{query}'")
    target_slugs = {"automata-fix-new", "core-java-entry-level-new", "java-8-new", "core-java-advanced-level-new"}
    hits = 0
    for i, meta in enumerate(results["metadatas"][0]):
        url = meta["url"]
        slug = url.rstrip("/").split("/")[-1]
        match = "✅" if slug in target_slugs else "  "
        if slug in target_slugs:
            hits += 1
        print(f"  {match} {i+1:2}. {slug:45s} | {meta['name'][:45]}")

    print(f"\nRecall@10 (Java query): {hits}/{len(target_slugs)} = {hits/len(target_slugs):.4f}")

    # Sales query
    query2 = "new graduates sales role hour per test"
    emb2 = model.encode(query2).tolist()
    results2 = collection.query(query_embeddings=[emb2], n_results=15)
    print(f"\nTop results for: '{query2}'")
    target2 = {"entry-level-sales-7-1"}
    hits2 = 0
    for i, meta in enumerate(results2["metadatas"][0]):
        url = meta["url"]
        slug = url.rstrip("/").split("/")[-1]
        match = "✅" if slug in target2 else "  "
        if slug in target2:
            hits2 += 1
        print(f"  {match} {i+1:2}. {slug:45s} | {meta['name'][:45]}")
    print(f"\nSales recall: {hits2}/{len(target2)}")

if __name__ == "__main__":
    test()
