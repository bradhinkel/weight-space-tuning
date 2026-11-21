"""Experiment V data: Waterbirds splits, the practice folder, and the split guard.

    ~/vision_env/.venv/bin/python -m exp_v.data build     # writes practice/ and the manifests
    ~/vision_env/.venv/bin/python -m exp_v.data check     # the split guard; exits non-zero on any leak

Labels: y = 1 waterbird, 0 landbird; place = 1 water background, 0 land.
Groups g = 2*y + place: 0 landbird/land, 1 landbird/water, 2 waterbird/land, 3 waterbird/water.

The guard: the sandbox only ever mounts PRACTICE (validation images, copied).
Test images stay under RAW and are read only by the scoring side.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "datasets" / "waterbirds"
RAW = ROOT / "extracted"            # the WILDS tarball, as released
PRACTICE = ROOT / "practice"        # validation split only; the sandbox's /practice
MANIFESTS = ROOT / "manifests"      # train.csv, val.csv, test.csv (scoring side)
SPLITS = {0: "train", 1: "val", 2: "test"}
GROUPS = {0: "landbird/land", 1: "landbird/water", 2: "waterbird/land", 3: "waterbird/water"}


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(n):
            h.update(b)
    return h.hexdigest()


def metadata():
    m = pd.read_csv(RAW / "metadata.csv")
    m["split_name"] = m.split.map(SPLITS)
    m["g"] = 2 * m.y + m.place
    return m


def build():
    m = metadata()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    m["sha256"] = [sha(RAW / f) for f in m.img_filename]
    for name, part in m.groupby("split_name"):
        part[["img_id", "img_filename", "y", "place", "g", "sha256"]].to_csv(MANIFESTS / f"{name}.csv", index=False)
    if PRACTICE.exists():
        shutil.rmtree(PRACTICE)
    (PRACTICE / "images").mkdir(parents=True)
    val = m[m.split_name == "val"].reset_index(drop=True)
    rows = []
    for i, r in val.iterrows():
        pid = f"p{i:04d}"                                   # practice id: no CUB class or filename leaks through
        shutil.copyfile(RAW / r.img_filename, PRACTICE / "images" / f"{pid}.jpg")
        rows.append({"id": pid, "y": int(r.y), "place": int(r.place), "g": int(r.g)})
    pd.DataFrame(rows).to_csv(PRACTICE / "labels.csv", index=False)
    print(f"practice: {len(rows)} images -> {PRACTICE}")
    check()


def check():
    m = pd.concat([pd.read_csv(MANIFESTS / f"{s}.csv").assign(split=s) for s in SPLITS.values()])
    test_hashes = set(m[m.split == "test"].sha256)
    train_hashes = set(m[m.split == "train"].sha256)
    val_hashes = set(m[m.split == "val"].sha256)
    files = sorted((PRACTICE / "images").glob("*"))
    prac = {sha(f) for f in files}
    report = {
        "practice_files": len(files),
        "practice_equals_val": prac == val_hashes,
        "practice_test_overlap": len(prac & test_hashes),
        "practice_train_overlap": len(prac & train_hashes),
        "val_test_overlap": len(val_hashes & test_hashes),
        "train_test_overlap": len(train_hashes & test_hashes),
        "non_image_files": [p.name for p in PRACTICE.iterdir() if p.name not in ("images", "labels.csv")],
    }
    print(json.dumps(report, indent=1))
    ok = (report["practice_equals_val"] and report["practice_test_overlap"] == 0
          and report["practice_train_overlap"] == 0 and not report["non_image_files"])
    print("split guard:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        build()
    else:
        sys.exit(0 if check() else 1)
