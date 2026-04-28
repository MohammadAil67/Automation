"""
app.py — Streamlit UI for the SOP → Airtop Agent Prompt RAG system.

Run with:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

SAVED_SOPS_DIR = Path("./saved_sops")

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="SOP → Airtop Prompt Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — only things Streamlit can't do natively ──────────────────
st.markdown("""
<style>
.prompt-box {
    background: var(--secondary-background-color);
    border-left: 4px solid #4f6ef7;
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    font-family: 'Courier New', monospace;
    font-size: 0.87rem;
    line-height: 1.75;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-color);
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        "rag": None,
        "generated_prompt": "",
        "retrieved_chunks": [],
        "sop_sub_prompts": [],
        "ingested_files": [],
        "saved_sop_files": [],   # filenames of SOPs saved to disk for re-use
        "api_key_set": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Restore saved SOP file list from disk on session start
if not st.session_state["saved_sop_files"] and SAVED_SOPS_DIR.exists():
    st.session_state["saved_sop_files"] = sorted(
        f.name for f in SAVED_SOPS_DIR.iterdir() if f.is_file()
    )

def get_rag():
    return st.session_state.get("rag")

def set_rag(rag) -> None:
    st.session_state["rag"] = rag


# ── Lazy import ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag_class():
    from rag_pipeline import SOPAgentRAG
    return SOPAgentRAG


# ── Restore RAG instance from disk if session was reset ───────────────────
if st.session_state["rag"] is None and Path("./chroma_db").exists():
    try:
        set_rag(load_rag_class()())
    except Exception:
        pass  # silently skip if DB is corrupt or embeddings not yet available


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🤖 SOP → Airtop")
    st.caption("RAG-powered agent prompt generator")
    st.divider()

    # ── API key ──────────────────────────────────────────────────────────
    st.subheader("🔑 Google API Key")
    api_key_input = st.text_input(
        "Paste your key",
        type="password",
        placeholder="AIza...",
        help="Stored only in this browser session. Get a free key at aistudio.google.com/app/apikey",
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.session_state["api_key_set"] = True
        st.success("Key saved for this session.", icon="✅")
    elif os.environ.get("GOOGLE_API_KEY"):
        st.session_state["api_key_set"] = True
        st.info("Key loaded from environment.", icon="ℹ️")
    else:
        st.caption("🔗 [Get a free key](https://aistudio.google.com/app/apikey)")

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    llm_model = st.selectbox(
        "LLM model",
        ["gemini-2.0-flash", "gemini-2.5-flash"],
        index=0,
    )

    st.divider()

# ── Collection status ─────────────────────────────────────────────────
    st.subheader("📦 Collection")
    import shutil
    
    rag = get_rag()
    db_path = Path("./chroma_db")
    
    # 1. Show metrics if RAG is loaded in memory
    if rag:
        info = rag.collection_info()
        col_a, col_b = st.columns(2)
        col_a.metric("Chunks", info["documents"])
        col_b.metric("Files", len(st.session_state["ingested_files"]))
    else:
        # Check if the folder exists from a previous session
        if db_path.exists():
            st.info("Database exists on disk.", icon="💾")
        else:
            st.caption("Database is currently empty.")

    # 2. Always show the clear button if there is data in memory OR on disk
    if (rag or db_path.exists()):
        if st.button("🗑️ Clear database", use_container_width=True):
            
            # Clear via LangChain/Chroma class method if active
            if rag:
                try:
                    rag.clear_database()
                except Exception:
                    pass
            
            # Hard wipe the folder from disk to kill ghost files
            shutil.rmtree(db_path, ignore_errors=True)
            shutil.rmtree(SAVED_SOPS_DIR, ignore_errors=True)

            # Reset the Streamlit session state
            set_rag(None)
            st.session_state.update({
                "ingested_files": [],
                "saved_sop_files": [],
                "generated_prompt": "",
                "retrieved_chunks": [],
                "sop_sub_prompts": [],
            })
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════

tab_ingest, tab_generate, tab_about = st.tabs(
    ["📂 Ingest Documents", "✨ Generate Prompt", "ℹ️ About"]
)


# ─── TAB 1: INGEST ────────────────────────────────────────────────────────

with tab_ingest:
    st.header("Ingest Documents")

    if not st.session_state["api_key_set"]:
        st.warning("Add your Google API key in the sidebar before ingesting.", icon="⚠️")

    # ── Section 1: SOP Files ──────────────────────────────────────────────
    st.subheader("📋 SOP Files")
    st.write(
        "Upload Standard Operating Procedure documents. "
        "These will be saved and made available for prompt generation in the Generate tab. "
        "Supported formats: **PDF, DOCX, TXT, MD**"
    )

    sop_files = st.file_uploader(
        "Upload SOP files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=not st.session_state["api_key_set"],
        key="sop_uploader",
    )

    col_sop, col_demo = st.columns([2, 1], gap="medium")

    with col_sop:
        ingest_sop_btn = st.button(
            "⚡ Ingest SOP files",
            use_container_width=True,
            disabled=not sop_files or not st.session_state["api_key_set"],
            type="primary",
        )

    with col_demo:
        demo_btn = st.button(
            "🎲 Load demo SOP",
            use_container_width=True,
            disabled=not st.session_state["api_key_set"],
            help="Loads a built-in sample SOP so you can try without uploading anything.",
        )

    if ingest_sop_btn and sop_files:
        SOPAgentRAG = load_rag_class()
        SAVED_SOPS_DIR.mkdir(exist_ok=True)

        with st.status("Ingesting SOP files…", expanded=True) as status:
            rag = get_rag() or SOPAgentRAG()
            for f in sop_files:
                saved_path = SAVED_SOPS_DIR / f.name
                saved_path.write_bytes(f.getvalue())
                st.write(f"• Ingesting **{f.name}**…")
                rag.ingest(str(saved_path))
                if f.name not in st.session_state["ingested_files"]:
                    st.session_state["ingested_files"].append(f.name)
                if f.name not in st.session_state["saved_sop_files"]:
                    st.session_state["saved_sop_files"].append(f.name)
            set_rag(rag)
            status.update(label="SOP ingestion complete!", state="complete")

        st.success(
            f"✅ {len(sop_files)} SOP file(s) ingested. "
            f"Collection now has **{rag.collection_info()['documents']}** chunks.",
        )

    if demo_btn:
        from main import DEMO_SOP_TEXT
        SOPAgentRAG = load_rag_class()
        SAVED_SOPS_DIR.mkdir(exist_ok=True)
        demo_name = "demo_sop.txt"
        demo_saved_path = SAVED_SOPS_DIR / demo_name

        with st.status("Loading demo SOP…", expanded=True) as status:
            demo_saved_path.write_text(DEMO_SOP_TEXT, encoding="utf-8")
            rag = get_rag() or SOPAgentRAG()
            rag.ingest(str(demo_saved_path))
            if "demo_sop.txt (built-in)" not in st.session_state["ingested_files"]:
                st.session_state["ingested_files"].append("demo_sop.txt (built-in)")
            if demo_name not in st.session_state["saved_sop_files"]:
                st.session_state["saved_sop_files"].append(demo_name)
            set_rag(rag)
            status.update(label="Demo SOP loaded!", state="complete")

        st.success(
            f"✅ Demo SOP ingested. "
            f"Collection now has **{rag.collection_info()['documents']}** chunks.",
        )

    st.divider()

    # ── Section 2: User Guide / Markdown Files ────────────────────────────
    st.subheader("📚 User Guides")
    st.write(
        "Upload Markdown documentation files (e.g. product user guides). "
        "These use a header-aware hybrid chunker for better retrieval quality. "
        "Supported formats: **MD**"
    )

    guide_files = st.file_uploader(
        "Upload user guide files",
        type=["md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=not st.session_state["api_key_set"],
        key="guide_uploader",
    )

    ingest_guides_btn = st.button(
        "⚡ Ingest user guides",
        use_container_width=True,
        disabled=not guide_files or not st.session_state["api_key_set"],
        type="primary",
    )

    if ingest_guides_btn and guide_files:
        SOPAgentRAG = load_rag_class()
        with st.status("Ingesting user guides…", expanded=True) as status:
            with tempfile.TemporaryDirectory() as tmpdir:
                rag = get_rag() or SOPAgentRAG()
                for f in guide_files:
                    dest = Path(tmpdir) / f.name
                    dest.write_bytes(f.getvalue())
                    st.write(f"• Ingesting **{f.name}**…")
                    rag.ingest_guides(str(dest))
                    if f.name not in st.session_state["ingested_files"]:
                        st.session_state["ingested_files"].append(f.name)
                set_rag(rag)
                status.update(label="User guide ingestion complete!", state="complete")

        st.success(
            f"✅ {len(guide_files)} guide file(s) ingested. "
            f"Collection now has **{rag.collection_info()['documents']}** chunks.",
        )

    # ── Ingested file list ────────────────────────────────────────────────
    if st.session_state["ingested_files"]:
        st.divider()
        st.subheader("Ingested files")
        for name in st.session_state["ingested_files"]:
            st.markdown(f"- 📄 `{name}`")


# ─── TAB 2: GENERATE ──────────────────────────────────────────────────────

with tab_generate:
    st.header("Generate Airtop Agent Prompt")

    rag = get_rag()

    # ── Mode selector — always visible ────────────────────────────────────
    mode = st.radio(
        "Generation mode",
        options=["From task description", "From SOP file"],
        horizontal=True,
        help=(
            "**From task description** — describe a task in plain English; "
            "the RAG system retrieves the most relevant knowledge-base chunks "
            "and generates a prompt from them. Best when the knowledge base "
            "contains reference docs (e.g. Braze docs).\n\n"
            "**From SOP file** — select a previously ingested SOP; steps are "
            "extracted automatically and a consolidated prompt is generated. "
            "No manual task description needed."
        ),
    )

    st.divider()

    # ── Mode A: task description ───────────────────────────────────────────
    if mode == "From task description":
        if not rag or rag.collection_info()["documents"] == 0:
            st.info(
                "Ingest at least one document into the knowledge base first — "
                "go to the **Ingest Documents** tab.",
                icon="👈",
            )
        else:
            import rag_pipeline as _rp
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough

            _retriever = rag._vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": _rp.TOP_K_RESULTS},
            )
            _llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.2)
            _prompt = ChatPromptTemplate.from_template(_rp.AIRTOP_PROMPT_TEMPLATE)

            def _format_docs(docs):
                parts = []
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source_file", "unknown")
                    parts.append(f"[Chunk {i} — source: {src}]\n{doc.page_content}")
                return "\n\n".join(parts)

            rag._chain = (
                {"context": _retriever | _format_docs, "task": RunnablePassthrough()}
                | _prompt
                | _llm
                | StrOutputParser()
            )
            rag._retriever = _retriever
            rag._llm = _llm

            st.write(
                "Describe what you want the Airtop agent to do. "
                "The RAG system will retrieve the most relevant chunks from the "
                "knowledge base and generate a precise, step-by-step agent prompt."
            )

            task = st.text_area(
                "Task description",
                placeholder=(
                    "e.g. Log into the order management portal and export all "
                    "pending orders from the last 30 days as a CSV file."
                ),
                height=120,
            )

            generate_btn = st.button(
                "🚀 Generate prompt",
                disabled=not task.strip(),
                type="primary",
            )

            if generate_btn and task.strip():
                with st.spinner("Retrieving chunks and generating prompt…"):
                    result = rag.generate_prompt(task)
                    st.session_state["generated_prompt"] = result["final_prompt"]
                    st.session_state["retrieved_chunks"] = result["retrieved_chunks"]
                    st.session_state["sop_sub_prompts"] = []

    # ── Mode B: SOP file ───────────────────────────────────────────────────
    else:
        saved_sops = st.session_state["saved_sop_files"]

        if not saved_sops:
            st.info(
                "No SOP files saved yet. Upload and ingest a SOP in the "
                "**Ingest Documents** tab first.",
                icon="👈",
            )
        else:
            st.write(
                "Select a previously ingested SOP file. Steps will be extracted "
                "automatically and a consolidated Airtop agent prompt will be "
                "generated — no task description needed."
            )

            selected_sop = st.selectbox(
                "Select SOP file",
                options=saved_sops,
                format_func=lambda name: f"📄 {name}",
            )

            generate_sop_btn = st.button(
                "🚀 Generate from SOP",
                type="primary",
            )

            if generate_sop_btn:
                if not rag or rag.collection_info()["documents"] == 0:
                    st.warning(
                        "The knowledge base is empty. Ingest some documents first "
                        "so the RAG system has context to ground the prompt.",
                        icon="⚠️",
                    )
                else:
                    sop_path = SAVED_SOPS_DIR / selected_sop
                    if not sop_path.exists():
                        st.error(
                            f"File '{selected_sop}' not found on disk. "
                            "Please re-ingest it from the Ingest Documents tab."
                        )
                    else:
                        with st.spinner("Extracting SOP steps and generating prompt…"):
                            result = rag.generate_from_sop(str(sop_path))
                            st.session_state["generated_prompt"] = result["final_prompt"]
                            st.session_state["retrieved_chunks"] = []
                            st.session_state["sop_sub_prompts"] = result["sub_prompts"]

    # ── Shared output ──────────────────────────────────────────────────────
    if st.session_state["generated_prompt"]:
        st.subheader("Generated Airtop Agent Prompt")

        st.markdown(
            f'<div class="prompt-box">{st.session_state["generated_prompt"]}</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        st.download_button(
            label="⬇️ Download as .txt",
            data=st.session_state["generated_prompt"],
            file_name="airtop_agent_prompt.txt",
            mime="text/plain",
        )

        # Retrieved chunks (task description mode)
        if st.session_state["retrieved_chunks"]:
            with st.expander(
                f"🔍 View {len(st.session_state['retrieved_chunks'])} retrieved chunks",
                expanded=False,
            ):
                for i, doc in enumerate(st.session_state["retrieved_chunks"], 1):
                    src = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", "—")
                    with st.container(border=True):
                        st.caption(f"Chunk {i} · {src} · page {page}")
                        st.text(doc.page_content)

        # Per-step sub-prompts (SOP file mode)
        if st.session_state.get("sop_sub_prompts"):
            with st.expander(
                f"🔍 View {len(st.session_state['sop_sub_prompts'])} per-step sub-prompts",
                expanded=False,
            ):
                for i, sp in enumerate(st.session_state["sop_sub_prompts"], 1):
                    with st.container(border=True):
                        st.caption(f"Step {i}")
                        st.text(sp)


# ─── TAB 3: ABOUT ─────────────────────────────────────────────────────────

with tab_about:
    st.header("About this tool")
    st.markdown("""
This app uses **Retrieval-Augmented Generation (RAG)** to turn your internal
Standard Operating Procedure documents into ready-to-use
[Airtop](https://airtop.ai) browser agent prompts.

### Pipeline

```
SOP files (PDF / DOCX / TXT / MD)       User Guides (MD)
        │                                       │
        ▼                                       ▼
  LangChain document loaders        Header-aware hybrid chunker
        │                                       │
        └───────────────┬───────────────────────┘
                        ▼
          sentence-transformers all-MiniLM-L6-v2
                        │
                        ▼
               ChromaDB (persisted to disk)
                        │
                        ▼
          Similarity search (top-k chunks)
                        │
                        ▼
         Gemini 2.0 Flash + Airtop prompt template
                        │
                        ▼
       Structured, step-by-step Airtop agent prompt
```

### Key design principles

- **Grounded output** — the LLM is instructed only to use information found in
  your SOPs. If context is thin it says so explicitly.
- **Persistent index** — ChromaDB stores embeddings on disk in `./chroma_db/`
  so you only pay embedding costs once per document.
- **Saved SOPs** — uploaded SOP files are persisted to `./saved_sops/` so they
  remain selectable for prompt generation across sessions.
- **Configurable** — chunk size, overlap, k, and LLM model are all adjustable
  from the sidebar without touching any code.

### Tech stack

| Layer | Library |
|---|---|
| SOP loading | `langchain-community` (PDF, DOCX, TXT, MD) |
| Guide chunking | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, no API key) |
| Vector store | ChromaDB |
| LLM | Gemini 2.0 Flash / 1.5 Pro / 1.5 Flash |
| Orchestration | LangChain LCEL |
| UI | Streamlit |
""")