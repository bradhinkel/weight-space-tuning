"""Test-set guard hashing (dependency-free; shared by the harness and exp_*/build_practice.py)."""
import hashlib


def norm_text(s) -> str:
    return " ".join(str(s).split()).lower()


def text_hash(s) -> str:
    return hashlib.sha256(norm_text(s).encode()).hexdigest()


def user_hashes(item: dict) -> list[str]:
    """One hash per user message; a prompt is refused if ANY of its user messages is a test message."""
    return [text_hash(m.get("content", "")) for m in item["messages"] if m.get("role") == "user"]
