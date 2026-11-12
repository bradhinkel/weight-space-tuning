"""A.8: export a driver session as a replayable timeline JSON (hypothesis -> probe -> edit -> result).

    python3 scripts/session_timeline.py artifacts/sessions/<session_id> --out artifacts/a/figures/timeline_b25.json

Every event comes from the session transcript (stream-json). Thinking text is not in the transcript
(only signatures), so "hypothesis" events are the driver's visible text plus its code comments.
Kinds:
  say       driver's visible text between tool calls
  analysis  python call that does not modify W (weights-only analysis, data prep, scoring)
  edit      python call that modifies W (in-place op or assignment on W[...]), or a restore/revert
  probe     forward_pass (one GPU pass, charged against the budget)
  save      save_weights;  submit: final note
Each tool event carries its (truncated) input and output; t is seconds since session start.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

EDIT = re.compile(r"W\[[^\]]+\](\[[^\]]*\])?\s*(=[^=]|\*=|\+=|-=|/=)|W\[[^\]]+\]\.(mul_|add_|sub_|copy_|zero_|div_|fill_)"
                  r"|load_state|restore|revert")
CUT = 1500


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")


def comments(code):
    return [l.strip()[1:].strip() for l in code.splitlines() if l.strip().startswith("#") and len(l.strip()) > 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.session)
    summ = json.loads((S / "driver_summary.json").read_text())
    t0 = ts(summ["start_utc"].replace("+00:00", "Z"))
    events, pending = [], {}
    for line in open(S / "transcript.jsonl"):
        d = json.loads(line)
        if d.get("type") not in ("assistant", "user") or not isinstance(d["message"]["content"], list):
            continue
        t = round(ts(d["timestamp"]) - t0, 1) if d.get("timestamp") else None
        for b in d["message"]["content"]:
            if d["type"] == "assistant" and b["type"] == "text" and b["text"].strip():
                events.append({"t": t, "kind": "say", "text": b["text"].strip()})
            elif b["type"] == "tool_use":
                name = b["name"].split("__")[-1]
                inp = b["input"]
                if name == "python":
                    code = inp.get("code", "")
                    kind = "edit" if EDIT.search(code) else "analysis"
                    ev = {"kind": kind, "code": code[:CUT], "comments": comments(code)[:8]}
                elif name == "forward_pass":
                    ev = {"kind": "probe", "args": {k: (v if k != "prompts" else f"{len(v)} inline prompts")
                                                     for k, v in inp.items()}}
                elif name == "save_weights":
                    ev = {"kind": "save", "note": inp.get("note", "")}
                elif name == "submit":
                    ev = {"kind": "submit", "note": inp.get("final_note", "")}
                else:
                    ev = {"kind": name, "args": inp}
                ev.update({"t": t, "tool": name, "id": b["id"]})
                events.append(ev); pending[b["id"]] = ev
            elif b["type"] == "tool_result" and b.get("tool_use_id") in pending:
                ev = pending.pop(b["tool_use_id"])
                out = text_of(b.get("content", ""))
                ev["output"] = out[:CUT]
                ev["t_done"] = t
                m = re.search(r'"used":\s*(\d+)', out) or re.search(r"passes used[^0-9]*(\d+)", out)
                if ev["kind"] == "probe" and m:
                    ev["budget_used"] = int(m.group(1))
                if b.get("is_error"):
                    ev["error"] = True
    probes, last_probe = 0, None
    for ev in events:
        ev.pop("id", None)
        if ev["kind"] == "probe":
            if '"error": "' in ev.get("output", "")[:200]:
                ev["error"] = True  # refused by the harness, not charged
            else:
                probes += 1; ev["probe_n"] = probes; last_probe = ev
        # the driver scores a pass in its next python call: attach "label {'irr_call_rate': ...}" lines to that pass
        if ev["kind"] in ("analysis", "edit") and last_probe is not None:
            for m in re.finditer(r"([^\n{\"]*?)\s*(\{'irr_call_rate'[^}]*\}|\{'n': \d+, 'call_rate'[^}]*\})",
                                 ev.get("output", "").replace('\\n', '\n')):
                last_probe.setdefault("scores", []).append({"label": m.group(1).strip(), "score": m.group(2)})
            if "scores" in last_probe:
                last_probe = None
    final = json.loads((S / "summary.json").read_text()) if (S / "summary.json").exists() else {}
    out = {"session_id": summ["session_id"], "driver_model": summ["model_id"], "budget": summ["config"]["budget"],
           "wall_clock_s": summ.get("wall_clock_s"), "n_events": len(events),
           "counts": {k: sum(e["kind"] == k for e in events) for k in
                      ("say", "analysis", "edit", "probe", "save", "submit")},
           "note": "thinking is not recorded in the transcript; kinds are assigned by scripts/session_timeline.py",
           "events": events}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "events"}, indent=1))


if __name__ == "__main__":
    main()
