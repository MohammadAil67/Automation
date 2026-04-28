"""
main.py — CLI for the User Guide–grounded SOP → Airtop Agent Prompt system.

Commands
--------
  python main.py index-guides <path>       Index markdown user guide file(s) into ChromaDB
  python main.py generate <sop_file>       Generate a prompt by extracting steps from a SOP file
  python main.py generate-prompt "<task>"  Generate a prompt from a free-text task description
  python main.py info                      Show knowledge base stats
  python main.py clear                     Wipe the ChromaDB knowledge base

Examples
--------
  python main.py index-guides ./braze_docs/
  python main.py index-guides ./guides/saml_jitp.md
  python main.py generate ./sops/onboard_user.pdf
  python main.py generate-prompt "Log into the CRM and export all open deals as CSV"
  python main.py info
"""

import sys
from pathlib import Path

from rag_pipeline import SOPAgentRAG


# ---------------------------------------------------------------------------
# Demo content (used by app.py's "Load demo SOP" button)
# ---------------------------------------------------------------------------

DEMO_SOP_TEXT = """
SOP: Enable SAML Just-in-Time Provisioning for New Hires

Purpose:
Configure Braze so that new employees can self-provision on first login via the
company IdP, without requiring manual account creation by an admin.

Steps:
1. Log into Braze as an administrator.
2. Open Security Settings under Admin Settings.
3. Enable the Automatic user provisioning toggle in the SAML SSO section.
4. Choose the correct default workspace for new users.
5. Assign the standard new-hire permission set.
6. Save the configuration.
7. Add all new-hire email addresses to the SSO provider directory.
8. Communicate the IdP portal login URL to new hires.

Restrictions:
- This procedure only applies to users with an approved corporate email domain.
- Google SSO users are not supported — escalate to IT if needed.
- JITP cannot be disabled without contacting Braze Support.
""".strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return

    command = args[0].lower()

    rag = SOPAgentRAG()

    if command == "index-guides":
        if len(args) < 2:
            print("Usage: python main.py index-guides <path>")
            sys.exit(1)
        rag.ingest_guides(args[1])
        info = rag.collection_info()
        print(f"Knowledge base now contains {info['documents']} chunks.")

    elif command == "generate":
        if len(args) < 2:
            print('Usage: python main.py generate "<sop_file_path>"')
            sys.exit(1)
        result = rag.generate_from_sop(args[1])
        print("\n" + "=" * 70)
        print("FINAL AIRTOP AGENT PROMPT")
        print("=" * 70)
        print(result["final_prompt"])
        print("=" * 70)
        print(f"\n({len(result['sub_prompts'])} SOP steps processed)")

    elif command == "generate-prompt":
        if len(args) < 2:
            print('Usage: python main.py generate-prompt "<task description>"')
            sys.exit(1)
        task = " ".join(args[1:])  # allow multi-word task without quotes
        result = rag.generate_prompt(task)
        print("\n" + "=" * 70)
        print("FINAL AIRTOP AGENT PROMPT")
        print("=" * 70)
        print(result["final_prompt"])
        print("=" * 70)
        print(f"\n({len(result['retrieved_chunks'])} chunks retrieved)")

    elif command == "info":
        info = rag.collection_info()
        print(f"Status  : {info['status']}")
        print(f"Chunks  : {info['documents']}")

    elif command == "clear":
        rag.clear_database()

    else:
        print(f"Unknown command: '{command}'")
        print("Run `python main.py --help` for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()