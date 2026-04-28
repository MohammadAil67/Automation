"""
inspect_db.py — Quick inspector for the ChromaDB guide knowledge base.
 
Usage
-----
  python inspect_db.py              # summary + first 5 chunks
  python inspect_db.py --all        # every chunk
  python inspect_db.py --search "SAML provisioning"   # similarity search
  python inspect_db.py --stats      # metadata breakdown only
  python inspect_db.py --collections # list all collection names in the DB
"""
 
import sys
from embeddings import load_vector_store, CHROMA_PERSIST_DIR
 
 
def print_chunk(i: int, doc, metadata: dict) -> None:
    source   = metadata.get("source_file", "?")
    title    = metadata.get("page_title", "")
    section  = metadata.get("section", "")
    subsec   = metadata.get("subsection", "")
    breadcrumb = " › ".join(filter(None, [source, title, section, subsec]))
 
    print(f"\n{'─'*70}")
    print(f"  Chunk #{i:>4}  |  {breadcrumb}")
    print(f"{'─'*70}")
    preview = doc.page_content.replace("\n", " ")
    print(f"  {preview[:300]}{'…' if len(doc.page_content) > 300 else ''}")
 
 
def cmd_summary(vs, total: int) -> None:
    print(f"\n📦 ChromaDB at '{CHROMA_PERSIST_DIR}'")
    print(f"   Total chunks : {total}\n")
 
    # Collect metadata stats
    results = vs.get(include=["metadatas"])
    sources: dict[str, int] = {}
    for m in results["metadatas"]:
        src = m.get("source_file", "unknown")
        sources[src] = sources.get(src, 0) + 1
 
    print("   Chunks per source file:")
    for src, count in sorted(sources.items()):
        print(f"     {count:>5}  {src}")
 
    print(f"\n   Showing first 5 chunks (use --all to see everything):")
    docs = vs.get(include=["documents", "metadatas"], limit=5)
    for i, (doc, meta) in enumerate(zip(docs["documents"], docs["metadatas"]), 1):
        class _D:
            page_content = doc
        print_chunk(i, _D(), meta)
 
 
def cmd_all(vs) -> None:
    docs = vs.get(include=["documents", "metadatas"])
    print(f"\n📦 All {len(docs['documents'])} chunks:\n")
    for i, (doc, meta) in enumerate(zip(docs["documents"], docs["metadatas"]), 1):
        class _D:
            page_content = doc
        print_chunk(i, _D(), meta)
 
 
def cmd_search(vs, query: str) -> None:
    print(f"\n🔍 Similarity search: '{query}'\n")
    results = vs.similarity_search(query, k=5)
    for i, doc in enumerate(results, 1):
        print_chunk(i, doc, doc.metadata)
 
 
def cmd_stats(vs) -> None:
    docs = vs.get(include=["metadatas"])
    total = len(docs["metadatas"])
    print(f"\n📊 Metadata breakdown ({total} chunks total)\n")
 
    fields = ["source_file", "page_title", "section", "subsection"]
    for field in fields:
        counts: dict[str, int] = {}
        for m in docs["metadatas"]:
            val = m.get(field, "") or "(none)"
            counts[val] = counts.get(val, 0) + 1
        print(f"  {field}:")
        for val, count in sorted(counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {count:>5}  {val}")
        if len(counts) > 15:
            print(f"           … and {len(counts)-15} more")
        print()


def cmd_collections() -> None:
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collections = client.list_collections()
    
    print(f"\n📂 ChromaDB Collections at '{CHROMA_PERSIST_DIR}'\n")
    if not collections:
        print("  No collections found.")
        return
        
    for c in collections:
        # chromadb >= 0.4.0 returns Collection objects, older versions returned strings
        name = c.name if hasattr(c, "name") else str(c)
        count = c.count() if hasattr(c, "count") else "?"
        print(f"  • {name} (Chunks: {count})")
    print()
 
 
def main() -> None:
    from pathlib import Path
    if not Path(CHROMA_PERSIST_DIR).exists():
        print(f"No ChromaDB found at '{CHROMA_PERSIST_DIR}'. Ingest some guides first.")
        sys.exit(1)
 
    args = sys.argv[1:]

    # Handle the collections command before loading the default vector store
    if args and args[0] == "--collections":
        cmd_collections()
        sys.exit(0)
 
    vs    = load_vector_store()
    total = vs._collection.count()
 
    if total == 0:
        print("Knowledge base is empty.")
        sys.exit(0)
 
    if not args:
        cmd_summary(vs, total)
    elif args[0] == "--all":
        cmd_all(vs)
    elif args[0] == "--search" and len(args) > 1:
        cmd_search(vs, " ".join(args[1:]))
    elif args[0] == "--stats":
        cmd_stats(vs)
    else:
        print(__doc__)
 
 
if __name__ == "__main__":
    main()