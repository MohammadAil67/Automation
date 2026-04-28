"""
braze_downloader.py — Downloads and extracts Markdown files from the
Braze public documentation GitHub repository.
"""

import io
import os
import shutil
from pathlib import Path
from typing import Callable, Optional
import zipfile

import requests

REPO_ZIP_URL = "https://github.com/braze-inc/braze-docs/archive/refs/heads/main.zip"
OUTPUT_DIR = "braze_docs_md"
TEMP_DIR = "temp_braze_docs"


def download_braze_docs(
    output_dir: str = OUTPUT_DIR,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Download the Braze docs repository and extract all Markdown files.

    Parameters
    ----------
    output_dir : str
        Directory where extracted .md files will be saved.
    progress_callback : callable, optional
        Called with a status string at each major step so callers (e.g. a
        Streamlit UI) can surface progress to the user.

    Returns
    -------
    dict
        {
            "success": bool,
            "count": int,           # number of .md files extracted
            "output_dir": str,      # absolute path to the output folder
            "error": str | None,    # set when success is False
        }
    """

    def _log(msg: str) -> None:
        print(msg)
        if progress_callback:
            progress_callback(msg)

    # ── 1. Prepare output directory ──────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # ── 2. Download ZIP ──────────────────────────────────────────────────
    _log("📥 Downloading Braze docs repository… (this may take a moment)")
    try:
        response = requests.get(REPO_ZIP_URL, timeout=120)
    except requests.RequestException as exc:
        return {"success": False, "count": 0, "output_dir": output_dir, "error": str(exc)}

    if response.status_code != 200:
        return {
            "success": False,
            "count": 0,
            "output_dir": output_dir,
            "error": f"HTTP {response.status_code} when downloading repository.",
        }

    # ── 3. Extract ZIP ───────────────────────────────────────────────────
    _log("📦 Download complete! Extracting files…")
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(TEMP_DIR)
    except Exception as exc:
        return {"success": False, "count": 0, "output_dir": output_dir, "error": str(exc)}

    # ── 4. Collect .md files ─────────────────────────────────────────────
    _log("🔍 Gathering Markdown files…")
    source_dir = Path(TEMP_DIR)
    md_files = list(source_dir.rglob("*.md"))

    count = 0
    errors = []
    for md_file in md_files:
        try:
            relative_path = md_file.relative_to(source_dir)
            new_filename = str(relative_path).replace(os.sep, "_")
            output_path = os.path.join(output_dir, new_filename)
            shutil.copy2(md_file, output_path)
            count += 1
        except Exception as exc:
            errors.append(f"{md_file.name}: {exc}")

    if errors:
        _log(f"⚠️  Skipped {len(errors)} file(s) due to errors.")

    # ── 5. Clean up temp folder ──────────────────────────────────────────
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    _log(f"✅ Extracted {count} Markdown files to '{output_dir}'. Temp files cleaned up.")

    return {
        "success": True,
        "count": count,
        "output_dir": os.path.abspath(output_dir),
        "error": None,
    }


def list_downloaded_files(output_dir: str = OUTPUT_DIR) -> list[str]:
    """Return a sorted list of .md filenames in the output directory."""
    p = Path(output_dir)
    if not p.exists():
        return []
    return sorted(f.name for f in p.glob("*.md"))


def clear_downloaded_files(output_dir: str = OUTPUT_DIR) -> int:
    """Delete the output directory. Returns the number of files removed."""
    p = Path(output_dir)
    if not p.exists():
        return 0
    count = len(list(p.glob("*.md")))
    shutil.rmtree(p, ignore_errors=True)
    return count


# ---------------------------------------------------------------------------
# Fix missing import (zipfile used above but not imported at module level)
# ---------------------------------------------------------------------------
  # noqa: E402  (kept here to keep the function readable above)


if __name__ == "__main__":
    result = download_braze_docs()
    if result["success"]:
        print(f"Done — {result['count']} files saved to {result['output_dir']}")
    else:
        print(f"Failed: {result['error']}")
