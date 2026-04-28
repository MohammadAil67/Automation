"""
RAG pipeline for SOP-based Airtop agent prompt generation.

Flow:
  1. Ingest SOP documents (PDF / DOCX / TXT) → chunk → embed → store in ChromaDB
  2. Given a task description, retrieve the most relevant SOP chunks
  3. Feed chunks into an LLM to synthesise a structured Airtop agent prompt
"""

import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chunking import chunk_markdown_directory, chunk_markdown_hybrid
from embeddings import build_vector_store, load_vector_store

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = "./chroma_db"
#COLLECTION_NAME = "sop_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, local, no API key needed (~90 MB)

CHUNK_SIZE = 700       # characters per chunk
CHUNK_OVERLAP = 150     # overlap keeps context across boundaries
TOP_K_RESULTS = 5       # chunks returned per query


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

AIRTOP_PROMPT_TEMPLATE = """
You are an expert at writing precise, actionable prompts for Airtop browser agents.

Below are the most relevant sections extracted from Standard Operating Procedure (SOP)
documents. Use them as the authoritative source of truth for the agent's behaviour.

--- RETRIEVED SOP CONTEXT ---
{context}
--- END OF CONTEXT ---

TASK DESCRIPTION:
{task}

Using ONLY the information found in the SOP context above, generate a complete Airtop
agent prompt. The prompt must:

1. Start with a one-sentence objective that clearly states what the agent should achieve.
2. List every step the agent must follow, in order, as numbered instructions.
3. Call out any rules, restrictions, or edge-cases mentioned in the SOPs.
4. Specify the expected output or success condition.
5. Be written in clear, imperative language (e.g. "Navigate to…", "Click…", "Extract…").

Do NOT invent steps that are not grounded in the retrieved SOP context.
If the context is insufficient, say so explicitly and list what information is missing.

AIRTOP AGENT PROMPT:
""".strip()


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> list[Document]:
    """Load a single document — supports PDF, DOCX, TXT, and MD."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    elif suffix in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: .pdf, .docx, .txt, .md"
        )

    docs = loader.load()

    # Tag each page/chunk with the source filename for traceability
    for doc in docs:
        doc.metadata["source_file"] = path.name

    return docs


def load_documents_from_directory(directory: str) -> list[Document]:
    """Recursively load all supported documents from a directory."""
    all_docs: list[Document] = []
    supported = {".pdf", ".docx", ".doc", ".txt", ".md"}

    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in supported:
            print(f"  Loading: {file_path.name}")
            try:
                all_docs.extend(load_document(str(file_path)))
            except Exception as exc:
                print(f"  Warning — skipped {file_path.name}: {exc}")

    return all_docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks from {len(documents)} pages/sections.")
    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

def build_rag_chain(vector_store: Chroma):
    """
    Build a LangChain RAG chain that:
      - retrieves the top-K most relevant SOP chunks
      - formats them into the Airtop prompt template
      - calls the LLM and returns the generated prompt
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    prompt = ChatPromptTemplate.from_template(AIRTOP_PROMPT_TEMPLATE)

    def format_docs(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            parts.append(f"[Chunk {i} — source: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    chain = (
        {"context": retriever | format_docs, "task": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, llm


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

class SOPAgentRAG:
    """
    High-level interface for the SOP → Airtop prompt RAG system.

    Usage
    -----
    rag = SOPAgentRAG()
    rag.ingest("./sop_docs")          # or rag.ingest("path/to/file.pdf")
    prompt = rag.generate_prompt("Log into the CRM and extract all open deals")
    print(prompt)
    """

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self._vector_store: Optional[Chroma] = None
        self._chain = None
        self._retriever = None
        self._llm = None

        if Path(persist_directory).exists():
            print(f"Found existing ChromaDB at '{persist_directory}' — loading.")
            self._vector_store = load_vector_store(persist_directory)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_chain(self) -> None:
        """Build the RAG chain if it hasn't been built yet (lazy init)."""
        if self._chain is None and self._vector_store is not None:
            self._chain, self._retriever, self._llm = build_rag_chain(self._vector_store)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, path: str) -> None:
        """
        Ingest SOP documents from a file or directory.
        Re-calling this ADDS documents to the existing collection.
        """
        p = Path(path)
        print(f"\n[Ingest] Loading documents from: {path}")

        if p.is_dir():
            docs = load_documents_from_directory(path)
        elif p.is_file():
            docs = load_document(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        if not docs:
            print("  No documents found — nothing ingested.")
            return

        print(f"  Loaded {len(docs)} document sections.")
        chunks = split_documents(docs)

        if self._vector_store is None:
            self._vector_store = build_vector_store(chunks, self.persist_directory)
        else:
            self._vector_store.add_documents(chunks)
            print(f"  Added {len(chunks)} new chunks to existing collection.")

        self._chain, self._retriever, self._llm = build_rag_chain(self._vector_store)
        print("[Ingest] Complete.\n")

    # ------------------------------------------------------------------
    # Ingestion for GUIDES
    # ------------------------------------------------------------------
    def ingest_guides(self, path: str) -> None:
        """
        Specialized ingestion for Markdown user guides (e.g., Braze docs)
        using the header-aware hybrid chunker from chunking.py.
        """
        p = Path(path)
        print(f"\n[Ingest] Loading markdown guides from: {path}")

        # 1. Use the specialized hybrid chunker instead of the standard loader
        if p.is_dir():
            chunks = chunk_markdown_directory(str(path))
        elif p.is_file():
            chunks = chunk_markdown_hybrid(str(path))
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        if not chunks:
            print("  No markdown chunks generated — nothing ingested.")
            return

        print(f"  Generated {len(chunks)} hybrid chunks.")

        # 2. Embed and store the chunks in ChromaDB
        if self._vector_store is None:
            self._vector_store = build_vector_store(chunks, self.persist_directory)
        else:
            self._vector_store.add_documents(chunks)
            print(f"  Added {len(chunks)} new chunks to existing collection.")

        # 3. Rebuild the retrieval chain with the updated vector store
        self._chain, self._retriever, self._llm = build_rag_chain(self._vector_store)
        print("[Ingest] Markdown guides complete.\n")     
        

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def generate_prompt(self, task_description: str) -> dict:
        """
        Generate an Airtop agent prompt from a free-text task description.

        Use this when the knowledge base contains reference documentation
        (e.g. Braze docs) and the user describes a task against it.

        Parameters
        ----------
        task_description : str
            What the Airtop agent should do, e.g.
            "Navigate to the order management portal and export all pending orders."

        Returns
        -------
        dict
            {
                "final_prompt":     str,        # generated Airtop agent prompt
                "retrieved_chunks": list[Document],  # chunks used for context
            }
        """
        if self._chain is None:
            self._ensure_chain()
        if self._chain is None:
            raise RuntimeError(
                "No documents have been ingested yet. Call .ingest() or "
                ".ingest_guides() first."
            )

        print(f"[generate_prompt] Task: {task_description}")
        print("[generate_prompt] Retrieving relevant chunks…")

        retrieved = self._retriever.invoke(task_description)
        print(f"[generate_prompt] Retrieved {len(retrieved)} chunks:")
        for doc in retrieved:
            src = doc.metadata.get("source_file", "unknown")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  • [{src}] {preview}…")

        print("[generate_prompt] Generating Airtop prompt…\n")
        final_prompt = self._chain.invoke(task_description)

        return {
            "final_prompt": final_prompt,
            "retrieved_chunks": retrieved,
        }

    def generate_from_sop(self, sop_file_path: str) -> dict:
        """
        Load a SOP file, extract each numbered step as a sub-task, generate
        an Airtop prompt grounded in the knowledge base, and return a dict
        with the final combined prompt and the individual per-step prompts.

        Parameters
        ----------
        sop_file_path : str
            Path to a SOP file (.pdf, .docx, .txt, or .md).

        Returns
        -------
        dict
            {
                "final_prompt": str,       # single consolidated Airtop prompt
                "sub_prompts":  list[str], # one prompt fragment per SOP step
            }
        """
        if self._chain is None:
            self._ensure_chain()
        if self._chain is None:
            raise RuntimeError(
                "No documents have been ingested yet. Call .ingest() or "
                ".ingest_guides() first."
            )

        # ── 1. Load and read the SOP file ─────────────────────────────────
        docs = load_document(sop_file_path)
        sop_text = "\n\n".join(doc.page_content for doc in docs).strip()

        if not sop_text:
            raise ValueError(f"No text could be extracted from '{sop_file_path}'.")

        print(f"[generate_from_sop] Loaded SOP ({len(sop_text)} chars): {sop_file_path}")

        # ── 2. Extract numbered steps from the SOP ─────────────────────────
        step_pattern = re.compile(r"^\s*\d+[\.\)]\s+.+", re.MULTILINE)
        steps = [m.group().strip() for m in step_pattern.finditer(sop_text)]

        if not steps:
            # No numbered steps found — treat the whole SOP as a single task
            steps = [sop_text]

        print(f"[generate_from_sop] Found {len(steps)} SOP step(s) to process.")

        # ── 3. Generate a prompt fragment for each step ────────────────────
        sub_prompts: list[str] = []
        for i, step in enumerate(steps, 1):
            print(f"[generate_from_sop] Processing step {i}/{len(steps)}: {step[:60]}…")
            fragment = self._chain.invoke(step)
            sub_prompts.append(fragment)
            # Pause between calls to avoid hitting Gemini rate limits
            if i < len(steps):
                time.sleep(3)

        # ── 4. Combine all fragments into one consolidated prompt ──────────
        # Call the LLM directly — no retrieval needed here since each step
        # already retrieved its own relevant context above.
        print("[generate_from_sop] Synthesising final consolidated prompt…")
        time.sleep(3)  # pause before the final synthesis call too

        combined = "\n\n".join(
            f"Step {i}:\n{sp}" for i, sp in enumerate(sub_prompts, 1)
        )
        synthesis_instruction = (
            "Below are individual Airtop agent prompt fragments, one per SOP step.\n"
            "Combine them into a single, coherent, end-to-end Airtop agent prompt.\n"
            "Preserve all steps and instructions. Remove any redundancy. "
            "Use clear imperative language throughout.\n\n"
            f"{combined}"
        )
        from langchain_core.messages import HumanMessage
        final_prompt = (self._llm | StrOutputParser()).invoke(
            [HumanMessage(content=synthesis_instruction)]
        )

        print("[generate_from_sop] Done.\n")
        return {
            "final_prompt": final_prompt,
            "sub_prompts": sub_prompts,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_database(self) -> None:
        """Delete all documents from the ChromaDB collection."""
        if self._vector_store:
            self._vector_store.delete_collection()
            self._vector_store = None
            self._chain = None
            self._retriever = None
            self._llm = None
            print("ChromaDB collection cleared.")
        else:
            print("No active collection to clear.")

    def collection_info(self) -> dict:
        """Return basic stats about the current collection."""
        if self._vector_store is None:
            return {"status": "empty", "documents": 0}
        count = self._vector_store._collection.count()
        return {"status": "ready", "documents": count}