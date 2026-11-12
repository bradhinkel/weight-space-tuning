"""Step 0.1: download models + Exp A datasets to the HF cache and record exact revisions.

Writes env/revisions.json (repo -> commit sha). Fails loudly if a gated repo is not accessible
(operating rule 8: report, never substitute).
"""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
REPO = Path(__file__).resolve().parents[1]

TARGETS = [
    ("Qwen/Qwen3.5-2B", "model"),
    ("Qwen/Qwen3.5-0.8B", "model"),
    ("Salesforce/xlam-function-calling-60k", "dataset"),
    ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "dataset"),
]


def main() -> int:
    api = HfApi()
    out, failures = {}, {}
    for repo_id, rtype in TARGETS:
        try:
            info = api.repo_info(repo_id, repo_type=rtype)
            path = snapshot_download(repo_id, repo_type=rtype, revision=info.sha)
            out[repo_id] = {"type": rtype, "sha": info.sha, "path": path}
            print(f"OK   {repo_id} @ {info.sha}")
        except Exception as e:  # noqa: BLE001 - we want the raw error in the report
            failures[repo_id] = f"{type(e).__name__}: {e}"
            print(f"FAIL {repo_id}: {failures[repo_id]}", file=sys.stderr)
    (REPO / "env").mkdir(exist_ok=True)
    (REPO / "env" / "revisions.json").write_text(json.dumps({"ok": out, "failed": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
