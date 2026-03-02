"""
embedder.py — Builds a ChromaDB vector index from the scraped SHL catalog.
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings.
Compatible with chromadb >= 1.0.0
"""
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"
CHROMA_PATH = Path(__file__).parent / "chroma_db"

def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_PATH))

def build_index():
    """Ingest SHL assessments into ChromaDB."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH}. Run scraper.py first.")

    assessments = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(assessments)} assessments from catalog.")

    model = get_model()
    client = get_client()

    # Always do a fresh collection
    try:
        client.delete_collection("shl_assessments")
        print("Dropped existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        "shl_assessments",
        metadata={"hnsw:space": "cosine"},
    )

    # Build texts and metadata
    texts = []
    metadatas = []
    ids = []

    for i, a in enumerate(assessments):
        test_types_str = ", ".join(a.get("test_type") or [])
        text = (
            f"{a['name']}. "
            f"{a.get('description', '')}. "
            f"Test types: {test_types_str}. "
            f"Duration: {a.get('duration', 0)} minutes. "
            f"Remote: {a.get('remote_support', 'No')}. "
            f"Adaptive: {a.get('adaptive_support', 'No')}."
        )
        texts.append(text)
        metadatas.append({
            "url": a["url"],
            "name": a["name"],
            "description": (a.get("description") or "")[:500],
            "test_type": test_types_str,
            "duration": int(a.get("duration") or 0),
            "remote_support": a.get("remote_support", "No"),
            "adaptive_support": a.get("adaptive_support", "No"),
        })
        ids.append(f"assessment_{i}")

    # Encode all at once (fastest)
    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings_list = embeddings.tolist()
    print("Encoding complete.")

    # Insert in batches of 100
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        collection.add(
            documents=texts[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Inserted {end}/{len(texts)}")

    count = collection.count()
    print(f"\n✅ Ingested {count} assessments into ChromaDB at {CHROMA_PATH}")
    return count


def search(query: str, n_results: int = 20) -> list[dict]:
    """Search ChromaDB for relevant assessments."""
    client = get_client()
    model = get_model()

    collection = client.get_collection("shl_assessments")
    query_emb = model.encode(query).tolist()

    n = min(n_results, collection.count())
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append({
            "url": meta["url"],
            "name": meta["name"],
            "description": meta["description"],
            "test_type": [t.strip() for t in meta["test_type"].split(",") if t.strip()] if meta["test_type"] else [],
            "duration": int(meta.get("duration") or 0),
            "remote_support": meta.get("remote_support", "No"),
            "adaptive_support": meta.get("adaptive_support", "No"),
        })
    return hits


if __name__ == "__main__":
    count = build_index()
    sys.exit(0 if count > 0 else 1)
