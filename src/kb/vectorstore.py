"""ChromaDB-based vector store for knowledge base entries."""

import json
import logging
import os

from src.kb.models import KBEntry

logger = logging.getLogger(__name__)


class KBVectorStore:
    """Wrapper around ChromaDB for storing and querying KB entries."""

    def __init__(self, persist_dir: str = "data/kb/chroma_db"):
        import chromadb

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="trading_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("KBVectorStore initialized at %s (%d entries)", persist_dir, self.collection.count())

    def add_entry(self, entry: KBEntry) -> None:
        """Add or update a single KB entry."""
        self.collection.upsert(
            ids=[entry.id],
            documents=[entry.combined_text],
            metadatas=[{
                "source_file": entry.source_file,
                "chapter": entry.chapter,
                "slide_number": entry.slide_number,
                "slide_title": entry.slide_title,
                "tags": json.dumps(entry.tags),
                "image_paths": json.dumps(entry.image_paths),
                "text_content": entry.text_content[:1000],
            }],
        )

    def add_entries_batch(self, entries: list[KBEntry]) -> None:
        """Add or update multiple KB entries in batch."""
        if not entries:
            return
        self.collection.upsert(
            ids=[e.id for e in entries],
            documents=[e.combined_text for e in entries],
            metadatas=[{
                "source_file": e.source_file,
                "chapter": e.chapter,
                "slide_number": e.slide_number,
                "slide_title": e.slide_title,
                "tags": json.dumps(e.tags),
                "image_paths": json.dumps(e.image_paths),
                "text_content": e.text_content[:1000],
            } for e in entries],
        )
        logger.info("Added %d entries to vector store", len(entries))

    def query(self, query_text: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        """Search for relevant KB entries by semantic similarity.

        Returns list of dicts with keys: id, document, metadata, distance.
        """
        kwargs = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.collection.count() or 1),
        }
        if where:
            kwargs["where"] = where
        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.error("Vector store query failed: %s", e)
            return []

        entries = []
        if results and results.get("ids") and results["ids"][0]:
            for i, entry_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                entries.append({
                    "id": entry_id,
                    "document": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": meta,
                    "distance": results["distances"][0][i] if results.get("distances") else 0,
                })
        return entries

    def get_entry(self, entry_id: str) -> dict | None:
        """Retrieve a specific entry by ID."""
        try:
            result = self.collection.get(ids=[entry_id])
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0] if result.get("documents") else "",
                    "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                }
        except Exception as e:
            logger.error("Failed to get entry %s: %s", entry_id, e)
        return None

    def stats(self) -> dict:
        """Return statistics about the vector store."""
        count = self.collection.count()
        if count == 0:
            return {"total_entries": 0, "chapters": []}

        # Get all metadata to summarize chapters
        all_data = self.collection.get(include=["metadatas"])
        chapters = set()
        for meta in (all_data.get("metadatas") or []):
            if meta and meta.get("chapter"):
                chapters.add(meta["chapter"])

        return {
            "total_entries": count,
            "chapters": sorted(chapters),
            "num_chapters": len(chapters),
        }

    def delete_all(self) -> None:
        """Delete all entries from the collection."""
        self.client.delete_collection("trading_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="trading_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Deleted all entries from vector store")
