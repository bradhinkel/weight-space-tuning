"""A.8: Experiment A results table + figures, generated from the pulled result files (never typed by hand).

    <venv with matplotlib>/bin/python scripts/a8_figures.py      # writes artifacts/a/figures/*.png + results.{json,md}

Figures:
  irrelevance_vs_call.png       one point per arm (the plan's scatter)
  recovery_vs_touched.png       irrelevance recovery vs weights touched (log x)
  budget_curve.png              Arm 1 recovery and retention at budget 10/25/50 (two panels, one axis each)
  touched_heatmap.png           fraction of weights changed per layer x module: b25 (best Arm 1), b50, A.6, A.7
  timeline_b25.png              practice scores after each GPU pass of the b25 session (from timeline_b25.json)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

R = Path(__file__).resolve().parents[1]
SC = R / "artifacts/pulled/artifacts/a/score"
FIG = R / "artifacts/a/figures"
N_PARAMS = 2213241664
# reference palette (dataviz skill), light mode; families: driver / script / gradient; refs in gray
C = {"driver": "#2a78d6", "script": "#eb6834", "gradient": "#1baf7a", "ref": "#8a8984"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200})

# --- per-run cost/time (method only; the shared BFCL test eval is excluded). Sources in reports/A.2,A.5,A.5a,A.6,A.7.
POD = 0.59
COST = {  # run: (wall-clock s, GPU-h, notional API $)
    "broken": (239 * 60, 239 / 60, 0.0),
    "a5_b10_fable_cpu": (8475, 8475 / 3600, 10.42),
    "a5_b10": (1391, 1391 / 3600, 8.99),
    "a5_b25": (1984, 1984 / 3600, 12.81),
    "a5_b50": (2445, 2445 / 3600, 20.93),
    "a6_steer": (948, 948 / 3600, 0.0),
    "a7_lora": (1199 + 1541, (1199 + 1541) / 3600, 0.0),
}
LABEL = {"base": "base", "broken": "broken", "a5_b10_fable_cpu": "Fable b10, unmetered CPU",
         "a5_b10": "Fable b10", "a5_b25": "Fable b25", "a5_b50": "Fable b50",
         "a6_steer": "steering ablation", "a7_lora": "corrective LoRA"}
FAMILY = {"base": "ref", "broken": "ref", "a5_b10_fable_cpu": "driver", "a5_b10": "driver", "a5_b25": "driver",
          "a5_b50": "driver", "a6_steer": "script", "a7_lora": "gradient"}
ARM = {"base": "—", "broken": "—", "a5_b10_fable_cpu": "1 (separate point)", "a5_b10": "1", "a5_b25": "1",
       "a5_b50": "1", "a6_steer": "2", "a7_lora": "3"}


def rows():
    out = {}
    for f in ("a5_all_scores.json", "a7_scores.json"):
        for r in json.load(open(SC / f))["rows"]:
            out[r["run"]] = r
    for k, v in out.items():
        v["touched_n"] = None if v["weights_touched_pct"] is None else round(v["weights_touched_pct"] / 100 * N_PARAMS)
    return out


def table(R_):
    order = ["base", "broken", "a5_b10", "a5_b25", "a5_b50", "a6_steer", "a7_lora", "a5_b10_fable_cpu"]
    lines = ["| run | arm | test irrelevance | mean call acc | recovery | retention (pts) | weights touched | "
             "method wall-clock | GPU $ | driver $ (notional, Max) |", "|---|---|---|---|---|---|---|---|---|---|"]
    js = []
    for k in order:
        r = R_[k]
        w, g, api = COST.get(k, (None, None, None))
        tn = r["touched_n"]
        touched = "—" if tn is None else (f"{tn:,} ({r['weights_touched_pct']:.4f}%)" if r["weights_touched_pct"] < 0.01
                                          else f"{r['weights_touched_pct']:.2f}%")
        if k == "broken":
            touched += " vs base"
        lines.append(f"| {LABEL[k]} | {ARM[k]} | {r['irrelevance']:.2f} | {r['mean_call_accuracy']:.2f} | "
                     f"{r['recovery']:.3f} | {r['retention']:+.2f} | {touched} | "
                     f"{'—' if w is None else f'{w / 60:.0f} min'} | {'—' if g is None else f'${g * POD:.2f}'} | "
                     f"{'—' if not api else f'${api:.2f}'} |")
        js.append({**r, "label": LABEL[k], "arm": ARM[k], "wall_clock_s": w,
                   "gpu_usd": None if g is None else round(g * POD, 2), "driver_notional_usd": api})
    return "\n".join(lines), js


def label_points(ax, pts, offsets):
    for k, (x, y) in pts.items():
        dx, dy, ha = offsets.get(k, (6, 4, "left"))
        ax.annotate(LABEL[k], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=9, color=INK)


def scatter(R_):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.axhspan(R_["broken"]["mean_call_accuracy"] - 2, 90, color="#eef4fb", zorder=0, lw=0)
    ax.text(2, R_["broken"]["mean_call_accuracy"] + 1.4, "retention target (≥ broken − 2 pts)", fontsize=8.5,
            color=INK2)
    pts = {}
    for k, r in R_.items():
        x, y = r["irrelevance"], r["mean_call_accuracy"]
        hollow = k == "a5_b10_fable_cpu"
        ax.scatter(x, y, s=80 if k != "a5_b25" else 130, zorder=3, edgecolor=SURF if not hollow else C[FAMILY[k]],
                   linewidth=2 if not hollow else 1.8, facecolor="none" if hollow else C[FAMILY[k]],
                   marker="D" if FAMILY[k] == "ref" else "o")
        pts[k] = (x, y)
    label_points(ax, pts, {"base": (-10, -4, "right"), "broken": (10, -3, "left"), "a5_b10": (-10, 0, "right"),
                           "a5_b25": (0, 11, "center"), "a5_b50": (10, -3, "left"),
                           "a6_steer": (10, -11, "left"), "a7_lora": (0, -17, "center"),
                           "a5_b10_fable_cpu": (0, -17, "center")})
    ax.set_xlim(-3, 95); ax.set_ylim(64, 89)
    ax.set_xlabel("test irrelevance accuracy (%)  →  abstains when no tool fits")
    ax.set_ylabel("mean call accuracy, 6 BFCL categories (%)")
    ax.set_title("Experiment A: repairing over-calling in Qwen3.5-2B", loc="left", fontsize=12, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=C[f], ms=8, label=l) for f, l in
               [("driver", "Arm 1: Fable driver (weight surgery)"), ("script", "Arm 2: scripted steering"),
                ("gradient", "Arm 3: corrective LoRA")]] + \
              [plt.Line2D([], [], marker="D", ls="", color=C["ref"], ms=7, label="reference models"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", color=C["driver"], ms=8,
                          label="unmetered run (harness hole)")]
    ax.legend(handles=handles, loc="lower center", fontsize=8.5, frameon=False, ncol=2,
              bbox_to_anchor=(0.5, -0.33))
    fig.tight_layout(); fig.savefig(FIG / "irrelevance_vs_call.png"); plt.close(fig)


def recovery_touched(R_):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pts = {}
    for k, r in R_.items():
        if k in ("base", "broken"):
            continue
        hollow = k == "a5_b10_fable_cpu"
        ax.scatter(r["touched_n"], r["recovery"], s=80 if k != "a5_b25" else 130, zorder=3,
                   facecolor="none" if hollow else C[FAMILY[k]], edgecolor=C[FAMILY[k]] if hollow else SURF,
                   linewidth=1.8)
        pts[k] = (r["touched_n"], r["recovery"])
    label_points(ax, pts, {"a5_b25": (10, -10, "left"), "a5_b10": (10, 2, "left"), "a5_b50": (10, -8, "left"),
                           "a7_lora": (-10, 4, "right"), "a6_steer": (-10, 6, "right"),
                           "a5_b10_fable_cpu": (8, -4, "left")})
    ax.set_xscale("log"); ax.set_xlim(5e2, 5e9); ax.set_ylim(-0.08, 1.15)
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(1.5e5, 0.965, "base-model level", fontsize=8.5, color=INK2)
    ax.set_xlabel("weights changed relative to the broken model (log scale; total 2.2B)")
    ax.set_ylabel("irrelevance recovery")
    ax.set_title("Recovery per weight changed", loc="left", fontsize=12, color=INK)
    for x, t in [(2.2e9, "100%"), (2.2e7, "1%"), (2.2e5, "0.01%")]:
        ax.axvline(x, color=GRID, lw=1, zorder=0)
        ax.text(x, -0.06, t, fontsize=8, color=INK2, ha="center")
    fig.tight_layout(); fig.savefig(FIG / "recovery_vs_touched.png"); plt.close(fig)


def budget_curve(R_):
    ks, b = ["a5_b10", "a5_b25", "a5_b50"], [10, 25, 50]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, key, ttl, ref in [(axs[0], "recovery", "irrelevance recovery", 1.0),
                              (axs[1], "retention", "call retention (pts vs broken)", -2.0)]:
        y = [R_[k][key] for k in ks]
        ax.plot(b, y, color=C["driver"], lw=2, marker="o", ms=8, mec=SURF, mew=2, zorder=3)
        for xi, yi in zip(b, y):
            below = key == "retention" and yi < -12
            ax.annotate(f"{yi:.2f}" if key == "recovery" else f"{yi:+.1f}", (xi, yi), xytext=(0, -16 if below else 9),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.axhline(ref, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.text(62, ref + (0.012 if key == "recovery" else 0.4), "base level" if key == "recovery" else "target ≥ −2",
                fontsize=8.5, color=INK2, va="bottom", ha="right")
        ax.set_xticks(b); ax.set_xlabel("GPU-pass budget"); ax.set_title(ttl, loc="left", fontsize=10.5, color=INK)
        ax.set_xlim(5, 62)
    axs[0].set_ylim(0.6, 1.15); axs[1].set_ylim(-18, 2)
    fig.suptitle("Arm 1 (Fable 5.1): one session per budget", x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "budget_curve.png"); plt.close(fig)


MODS = ["embed_tokens", "attn: q/k/v (full)", "attn: o_proj (full)", "linear_attn: in_proj", "linear_attn: out_proj",
        "mlp: gate/up", "mlp: down_proj", "norms & other"]


def cell(name):
    m = re.search(r"layers\.(\d+)\.", name)
    layer = "embed" if m is None else int(m.group(1))
    if "embed_tokens" in name:
        return "embed", "embed_tokens"
    if "self_attn" in name:
        return layer, "attn: o_proj (full)" if "o_proj" in name else ("norms & other" if "norm" in name
                                                                       else "attn: q/k/v (full)")
    if "linear_attn" in name:
        return layer, "linear_attn: out_proj" if "out_proj" in name else (
            "linear_attn: in_proj" if "in_proj" in name else "norms & other")
    if "mlp" in name:
        return layer, "mlp: down_proj" if "down_proj" in name else "mlp: gate/up"
    return layer, "norms & other"


def grid(tensors):
    rows_ = ["embed"] + list(range(24))
    ch = np.zeros((len(rows_), len(MODS))); num = np.zeros_like(ch)
    for t in tensors:
        l, m = cell(t["name"])
        if l not in rows_:
            continue
        i, j = rows_.index(l), MODS.index(m)
        ch[i, j] += t["n_changed"]; num[i, j] += t["numel"]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(ch > 0, ch / np.maximum(num, 1), np.nan)
    return rows_, frac, ch


def heatmap():
    runs = [("Fable b25 (best Arm 1)", json.load(open(SC / "a5_b25_fable_patched_final.json"))["tensors"]),
            ("Fable b50", json.load(open(SC / "a5_b50_fable_patched_final.json"))["tensors"]),
            ("A.6 steering ablation", json.load(open(SC / "touched_a6_vs_broken.json"))["tensors"]),
            ("A.7 corrective LoRA", json.load(open(SC / "touched_a7_vs_broken.json"))["tensors"])]
    fig, axs = plt.subplots(1, 4, figsize=(13, 6.4), sharey=True)
    cmap = matplotlib.colormaps["Blues"].copy(); cmap.set_bad("#f1f0ec")
    norm = LogNorm(1e-7, 1)
    for ax, (ttl, ts) in zip(axs, runs):
        rows_, frac, ch = grid(ts)
        im = ax.imshow(frac, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(MODS)), MODS, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows_)), [str(r) for r in rows_], fontsize=8)
        ax.set_title(f"{ttl}\n{int(ch.sum()):,} weights", loc="left", fontsize=10, color=INK)
        if ch.sum() < 1e8:  # sparse edits: label every touched cell, they are hard to see otherwise
            for i, j in zip(*np.nonzero(ch)):
                txt = f"{int(ch[i, j]):,}" + (" (1 row)" if rows_[i] == "embed" and ch[i, j] <= 2048 else "")
                ax.annotate(txt, (j, i), xytext=(8, 0), textcoords="offset points", fontsize=8, color=INK,
                            va="center", ha="left" if j < 4 else "right",
                            bbox={"boxstyle": "round,pad=0.2", "fc": SURF, "ec": INK2, "lw": 0.6})
        ax.grid(False)
        ax.set_xticks(np.arange(-.5, len(MODS)), minor=True); ax.set_yticks(np.arange(-.5, len(rows_)), minor=True)
        ax.grid(which="minor", color=SURF, lw=1.5); ax.tick_params(which="minor", length=0)
    axs[0].set_ylabel("layer (embed = tied embedding / unembedding)")
    cb = fig.colorbar(im, ax=axs, shrink=0.7, pad=0.01)
    cb.set_label("fraction of the cell's weights changed (log); gray = untouched")
    fig.suptitle("Where each repair changed the weights (vs the broken model)", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.savefig(FIG / "touched_heatmap.png", bbox_inches="tight"); plt.close(fig)


def timeline():
    d = json.load(open(FIG / "timeline_b25.json"))
    # baseline (probes 1-2, before any edit): the driver recorded 64/64 irrelevant prompts called and printed the
    # xLAM score {'call_rate': 1.0, 'name_match': 1.0} (see timeline_b25.json, probe 2 and the next python call)
    xs, irr, nm, lab = [1.8], [0.0], [1.0], ["baseline"]
    for e in d["events"]:
        if e["kind"] != "probe":
            continue
        for s in e.get("scores", []):
            v = eval(s["score"], {})  # a dict literal printed by the driver
            if "irr_call_rate" in v:
                xs.append(e["t"] / 60); irr.append(1 - v["irr_call_rate"]); nm.append(v["name_match"])
                lab.append(s["label"])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(xs, irr, color=C["driver"], lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label="abstains on irrelevant")
    ax.plot(xs, nm, color=C["script"], lw=2, marker="s", ms=6, mec=SURF, mew=1.5,
            label="correct function names on xLAM")
    ax.set_ylim(-0.05, 1.1); ax.set_xlim(0, 33.5)
    ax.set_xlabel("minutes into the session"); ax.set_ylabel("practice rate (32 + 32 prompts)")
    phases = [(1.8, "baseline"), (9.8, "scale <tool_call> row"), (15.2, "copy heads"), (21.6, "v_proj / MLPs"),
              (23.9, 'boost "I" row'), (25.4, 'mix <tool_call> into "I"'), (31.7, "save + submit")]
    for x, t in phases:
        ax.axvline(x, color=GRID, lw=1, zorder=0)
        ax.text(x + 0.25, 1.08, t, fontsize=7.8, color=INK2, rotation=90, va="top", ha="left")
    fin = [i for i, l in enumerate(lab) if l == "beta1.2 c0.3"]
    if fin:
        i = fin[0]
        ax.annotate("final edit chosen here:\nE[I] ← 1.2·E[I] + 0.30·E[<tool_call>]", (xs[i], irr[i]),
                    xytext=(19.0, 0.12), textcoords="data", fontsize=8.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.legend(loc="center left", frameon=False, fontsize=9, bbox_to_anchor=(0.0, 0.42))
    ax.set_title("Fable b25 session: practice scores after each edit (driver-printed)", loc="left", fontsize=12,
                 color=INK)
    fig.tight_layout(); fig.savefig(FIG / "timeline_b25.png"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    R_ = rows()
    md, js = table(R_)
    (FIG / "results.md").write_text(md + "\n")
    (FIG / "results.json").write_text(json.dumps(js, indent=1))
    print(md)
    scatter(R_); recovery_touched(R_); budget_curve(R_); heatmap(); timeline()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
