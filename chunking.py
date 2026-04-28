"""
chunking.py — Hybrid Metadata + Semantic chunking for markdown user guide files.

Strategy
--------
1. Header split  — MarkdownHeaderTextSplitter breaks each file at heading boundaries,
   carrying the full heading hierarchy (h1/h2/h3/h4) as metadata on every chunk.

2. Semantic sub-split — any section larger than GUIDE_CHUNK_SIZE chars is further
   split by RecursiveCharacterTextSplitter so embeddings stay focused and retrieval
   precision stays high.

3. Metadata enrichment — every final chunk carries:
     source_file  : filename
     full_path    : absolute path for traceability
     doc_type     : "user_guide"
     page_title   : outermost H1
     section      : H2 heading
     subsection   : H3 heading
"""

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GUIDE_CHUNK_SIZE    = 500   # max chars per final chunk
GUIDE_CHUNK_OVERLAP = 150   # overlap between semantic sub-chunks

MARKDOWN_HEADERS = [
    ("#",    "h1_title"),
    ("##",   "h2_section"),
    ("###",  "h3_subsection"),
    ("####", "h4_detail"),
]

#There maybe some problems with looping 

# ---------------------------------------------------------------------------
# Single-file chunking
# ---------------------------------------------------------------------------

def chunk_markdown_hybrid(file_path: str) -> list[Document]:
    """
    Apply hybrid chunking to a single markdown user guide file.

    Parameters
    ----------
    file_path : str
        Path to a .md or .markdown file.

    Returns
    -------
    list[Document]
        Chunks with enriched metadata ready for embedding.
    """
    path     = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")

    # ── Extract page title from the first H1 ──────────────────────────────
    page_title = ""
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            page_title = stripped.lstrip("# ").strip()
            break

    # ── Step 1: header-aware split ────────────────────────────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,    # keep headings in chunk text for LLM context
    )
    header_chunks = header_splitter.split_text(raw_text)

    # ── Step 2: semantic sub-split of large sections ──────────────────────
    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=GUIDE_CHUNK_SIZE,
        chunk_overlap=GUIDE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks: list[Document] = []
    for hchunk in header_chunks:
        if len(hchunk.page_content) > GUIDE_CHUNK_SIZE:
            sub_docs = semantic_splitter.create_documents(
                [hchunk.page_content],
                metadatas=[hchunk.metadata],
            )
        else:
            sub_docs = [hchunk]

        # ── Step 3: enrich metadata on every chunk ────────────────────────
        for doc in sub_docs:
            doc.metadata.update({
                "source_file" : path.name,
                "full_path"   : str(path.resolve()),
                "doc_type"    : "user_guide",
                "page_title"  : page_title or doc.metadata.get("h1_title", ""),
                "section"     : doc.metadata.get("h2_section", ""),
                "subsection"  : doc.metadata.get("h3_subsection", ""),
            })
            final_chunks.append(doc)

    print(f"  [{path.name}] → {len(final_chunks)} hybrid chunks  "
          f"(page: '{page_title}')")
    return final_chunks


# ---------------------------------------------------------------------------
# Directory chunking
# ---------------------------------------------------------------------------

def chunk_markdown_directory(directory: str) -> list[Document]:
    """
    Recursively chunk all .md / .markdown files found under a directory.

    Parameters
    ----------
    directory : str
        Root directory to search.

    Returns
    -------
    list[Document]
        Combined chunks from all markdown files found.
    """
    all_chunks: list[Document] = []
    for fp in Path(directory).rglob("*"):
        if fp.suffix.lower() in {".md", ".markdown"}:
            try:
                all_chunks.extend(chunk_markdown_hybrid(str(fp)))
            except Exception as exc:
                print(f"  Warning — skipped {fp.name}: {exc}")
    return all_chunks
