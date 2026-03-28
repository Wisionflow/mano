"""ChromaDB vector store wrapper for document storage and retrieval."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb


# Default paths
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "db")
COLLECTION_NAME = "medical_documents"

# Chunking parameters
CHUNK_SIZE = 512  # tokens (approximate: 1 token ≈ 4 chars for Russian)
CHUNK_OVERLAP = 50
CHARS_PER_TOKEN = 4  # approximate for Russian text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by approximate token count."""
    char_chunk = chunk_size * CHARS_PER_TOKEN
    char_overlap = overlap * CHARS_PER_TOKEN

    if len(text) <= char_chunk:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + char_chunk

        # Try to break at a sentence or newline boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? ", "; ", ", "]:
                boundary = text.rfind(sep, start + char_chunk // 2, end)
                if boundary != -1:
                    end = boundary + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - char_overlap

    return chunks


class VectorStore:
    """ChromaDB-backed vector store for medical documents."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(self, text: str, metadata: Dict[str, Any]) -> int:
        """Add a document to the store, chunked and embedded."""
        chunks = chunk_text(text)
        if not chunks:
            return 0

        base_id = metadata.get("file_name", "doc")
        ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]

        # Check for existing chunks from this file and delete them
        existing = self.collection.get(where={"file_name": metadata.get("file_name", "")})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        metadatas = []
        for i, chunk in enumerate(chunks):
            m = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            # ChromaDB metadata values must be str, int, float, or bool
            for k, v in list(m.items()):
                if not isinstance(v, (str, int, float, bool)):
                    m[k] = str(v)
            metadatas.append(m)

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                entry = {
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
                output.append(entry)

        return output

    def get_all_files(self) -> List[str]:
        """List all unique file names in the store."""
        all_data = self.collection.get()
        files = set()
        if all_data and all_data["metadatas"]:
            for m in all_data["metadatas"]:
                if "file_name" in m:
                    files.add(m["file_name"])
        return sorted(files)

    def get_document_count(self) -> int:
        """Return total number of chunks in the store."""
        return self.collection.count()

    def delete_file(self, file_name: str) -> int:
        """Delete all chunks for a given file."""
        existing = self.collection.get(where={"file_name": file_name})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            return len(existing["ids"])
        return 0

    def clear(self):
        """Delete all documents from the store."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
