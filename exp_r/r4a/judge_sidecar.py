"""R.4a judge sidecar: runs on the driver host next to the driver, and answers the server's judge requests.

The server writes <sessions>/<sid>/judge/req_<k>.json ({"pass", "items": [{"key", "question", "tools", "shown"}]},
blind keys, shuffled) and waits for res_<k>.json. The sidecar polls the pod over SSH, grades the items with the
A.10 judge (scripts/score_answers.py: Sonnet 5.5 via the local `claude` CLI on Max, the same rubric and batches of
12), and writes {"judgements": [...]} or {"error": "..."} back.

    python -m exp_r.r4a.judge_sidecar <sid> [--once]        # also started by exp_r.r4a.driver
"""
import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time

from common.harness.driver import ssh
from scripts.score_answers import judge as judge_batch

SESS = "/workspace/r4a/sessions"


def grade(items, model="claude-sonnet-5-5", batch=12, workers=4, tries=3):
    batches = [items[i:i + batch] for i in range(0, len(items), batch)]

    def work(b):
        err = None
        for _ in range(tries):
            try:
                return judge_batch(b, model)
            except Exception as e:          # an omitted item or a CLI error: retry the batch
                err = e
        raise RuntimeError(f"batch failed {tries} times: {err}")

    out = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(work, batches):
            out.extend(res)
    return out


def poll_once(sid, log=print):
    d = f"{SESS}/{sid}/judge"
    try:
        names = ssh(f"ls {d} 2>/dev/null || true").split()
    except Exception as e:
        log(f"judge sidecar: ls failed: {e}")
        return 0
    reqs = sorted(n for n in names if n.startswith("req_") and n.endswith(".json"))
    done = 0
    for r in reqs:
        k = r[len("req_"):-len(".json")]
        if f"res_{k}.json" in names:
            continue
        try:
            req = json.loads(ssh(f"cat {d}/{r}"))
            t0 = time.time()
            js = grade(req["items"])
            body = {"judgements": js, "s": round(time.time() - t0, 1)}
            log(f"judge sidecar: pass {k}: {len(js)} judgements in {body['s']} s")
        except Exception as e:
            body = {"error": repr(e)}
            log(f"judge sidecar: pass {k} failed: {e!r}")
        ssh(f"cat > {d}/.res_{k}.tmp && mv {d}/.res_{k}.tmp {d}/res_{k}.json", stdin=json.dumps(body))
        done += 1
    return done


def run(sid, stop: threading.Event, every=10, log=print):
    while not stop.is_set():
        poll_once(sid, log)
        stop.wait(every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.once:
        poll_once(a.sid)
        return 0
    run(a.sid, threading.Event())


if __name__ == "__main__":
    sys.exit(main())
