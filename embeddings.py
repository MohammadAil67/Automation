"""
embeddings.py — Embedding model and ChromaDB vector store helpers.

Responsibilities
----------------
- Instantiate the HuggingFace embedding model (local, no API key needed).
- Build a new ChromaDB collection from a list of Document chunks.
- Load an existing ChromaDB collection from disk.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME    = "airtop_knowledge_base"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"   


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def build_vector_store(
    chunks: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:

    """
    Embed a list of Document chunks and persist them in a new ChromaDB collection.

    Parameters
    ----------
    chunks : list[Document]
        Pre-chunked documents (typically from chunking.py).
    persist_directory : str
        Directory where ChromaDB stores its files.

    Returns
    -------
    Chroma
        The populated vector store instance.
    """
    embeddings = get_embeddings()
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    print(f"  Stored {len(chunks)} chunks in ChromaDB at '{persist_directory}'.")
    return vs


def load_vector_store(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME, # <--- Added this
) -> Chroma:
    
    return Chroma(
        collection_name=collection_name, # <--- Updated this
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )