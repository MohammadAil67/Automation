# SOP → Airtop Agent Prompt Generator

A **Retrieval-Augmented Generation (RAG)** system that converts Standard Operating Procedure (SOP) documents and user guides into structured, actionable prompts for [Airtop](https://www.airtop.ai/) browser agents. The system is powered by Google Gemini and local HuggingFace embeddings stored in ChromaDB.

---

## Features

- **RAG pipeline** – Ingest SOP documents (PDF, DOCX, TXT, MD) or Braze markdown guides, chunk them, embed them locally, and store in ChromaDB.
- **Prompt generation** – Given a free-text task or a SOP file, retrieve the most relevant context and synthesise a precise, step-by-step Airtop agent prompt.
- **Hybrid markdown chunking** – Header-aware splitting followed by semantic sub-splitting preserves document structure and maximises retrieval precision.
- **Braze docs downloader** – One-click download of the entire Braze public documentation repository for use as the knowledge base.
- **Streamlit UI** – Interactive web interface for managing the knowledge base and generating prompts without touching the CLI.
- **CLI** – Full command-line interface for scripting and automation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Ingestion                             │
│  SOP / Guide files  →  chunking.py  →  embeddings.py        │
│                              ↓                               │
│                         ChromaDB (local vector store)        │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Query / Generation                        │
│  Task description  →  Retriever (top-5 chunks)               │
│                              ↓                               │
│                   Gemini 1.5 Flash (LLM)                     │
│                              ↓                               │
│                   Airtop Agent Prompt (output)               │
└──────────────────────────────────────────────────────────────┘
```

| File | Responsibility |
|------|---------------|
| `rag_pipeline.py` | Core RAG logic — document loading, splitting, chain building, and `SOPAgentRAG` high-level API |
| `chunking.py` | Hybrid header + semantic chunking for markdown guides |
| `embeddings.py` | HuggingFace embedding model & ChromaDB vector store helpers |
| `braze_downloader.py` | Downloads and extracts Braze public docs as markdown files |
| `app.py` | Streamlit web interface |
| `main.py` | CLI entry point |
| `inspect_db.py` | Utility to inspect the ChromaDB knowledge base |

---

## Prerequisites

- Python 3.10+
- A **Google AI API key** (for Gemini 1.5 Flash) — get one at [aistudio.google.com](https://aistudio.google.com/app/apikey)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MohammadAil67/Automation.git
cd Automation

# 2. Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Usage

### Streamlit Web UI

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

From the sidebar you can:
1. Enter your Google API key.
2. Upload SOP files or download the Braze documentation.
3. Index guides into the knowledge base.
4. Generate Airtop agent prompts from a task description or an uploaded SOP file.

---

### CLI

```bash
# Index markdown user guide files into ChromaDB
python main.py index-guides ./braze_docs/
python main.py index-guides ./guides/saml_jitp.md

# Generate a prompt from a SOP file (PDF / DOCX / TXT / MD)
python main.py generate ./sops/onboard_user.pdf

# Generate a prompt from a free-text task description
python main.py generate-prompt "Log into the CRM and export all open deals as CSV"

# Show knowledge base stats
python main.py info

# Wipe the ChromaDB knowledge base
python main.py clear
```

---

### Download Braze Documentation

```bash
python braze_downloader.py
```

This downloads the Braze public docs repository and extracts all markdown files into a `braze_docs_md/` directory. You can then index that folder with `python main.py index-guides ./braze_docs_md/`.

---

### Inspect the Knowledge Base

```bash
# Summary + first 5 chunks
python inspect_db.py

# All chunks
python inspect_db.py --all

# Similarity search
python inspect_db.py --search "SAML provisioning"

# Metadata breakdown
python inspect_db.py --stats

# List all ChromaDB collections
python inspect_db.py --collections
```

---

## Configuration

Key constants can be adjusted at the top of each module:

| Setting | File | Default | Description |
|---------|------|---------|-------------|
| `EMBEDDING_MODEL` | `embeddings.py` | `all-MiniLM-L6-v2` | Local HuggingFace embedding model (~90 MB, no API key needed) |
| `COLLECTION_NAME` | `embeddings.py` | `airtop_knowledge_base` | ChromaDB collection name |
| `CHROMA_PERSIST_DIR` | `rag_pipeline.py` | `./chroma_db` | Directory for persistent ChromaDB storage |
| `CHUNK_SIZE` | `rag_pipeline.py` | `700` | Characters per chunk (SOP documents) |
| `CHUNK_OVERLAP` | `rag_pipeline.py` | `150` | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | `rag_pipeline.py` | `5` | Number of chunks retrieved per query |
| `GUIDE_CHUNK_SIZE` | `chunking.py` | `500` | Characters per chunk (markdown guides) |
| `GUIDE_CHUNK_OVERLAP` | `chunking.py` | `150` | Overlap between markdown guide chunks |

---

## Project Structure

```
Automation/
├── app.py               # Streamlit web UI
├── main.py              # CLI entry point
├── rag_pipeline.py      # Core RAG pipeline & SOPAgentRAG class
├── chunking.py          # Hybrid markdown chunking
├── embeddings.py        # Embedding model & ChromaDB helpers
├── braze_downloader.py  # Braze docs downloader
├── inspect_db.py        # ChromaDB inspector utility
├── requirements.txt     # Python dependencies
└── .env                 # (not committed) Google API key
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` + ecosystem | RAG chain, document loaders, text splitters |
| `langchain-google-genai` | Gemini LLM integration |
| `langchain-huggingface` | Local HuggingFace embeddings |
| `langchain-chroma` | ChromaDB vector store integration |
| `sentence-transformers` | Embedding model backend |
| `chromadb` | Local vector database |
| `pypdf` | PDF loading |
| `python-docx` | DOCX loading |
| `streamlit` | Web UI |
| `python-dotenv` | `.env` file support |
| `requests` | HTTP downloads (Braze docs) |