"""Drop practice items whose user message hits the R.4a test guard (on the pod: the guard is root-only).

    python -m exp_r.r4b.filter_guard <in.jsonl> <out.jsonl>
"""
import json
import sys

from exp_r.r4a.common import TestGuard

g = TestGuard()
rows = [json.loads(l) for l in open(sys.argv[1])]
keep = [r for r in rows if not g.hit_messages(r["messages"])]
with open(sys.argv[2], "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
print(f"kept {len(keep)} of {len(rows)}; dropped {[r['id'] for r in rows if r not in keep]}")
