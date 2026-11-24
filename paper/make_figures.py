"""Figures for paper/weight_space_router.md. Numbers are copied from the reports named beside each block.

    <venv with matplotlib>/bin/python paper/make_figures.py      # writes paper/fig*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def fig1():
    # reports/R.1.md, R.2.md, R.2_b50.md. Astra b10 = their published p10c; Astra b50 = p50c rerun in our pipeline.
    data = {"MATH-500": {"base": 7.70, "Astra": (12.3, 50.85), "Fable": (39.15, 51.85)},
            "GSM8K": {"base": 13.76, "Astra": (25.7, 67.29), "Fable": (48.67, 64.71)}}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, (name, d) in zip(axes, data.items()):
        x = [0, 1]
        w = 0.34
        for k, (drv, col) in enumerate([("Astra", GRAY), ("Fable", BLUE)]):
            xs = [i + (k - 0.5) * w for i in x]
            bars = ax.bar(xs, d[drv], w - 0.03, color=col, label=f"{'GPT-6 Astra (MIT CSAIL)' if drv == 'Astra' else 'Claude Fable 5.1 (ours)'}")
            for b, v in zip(bars, d[drv]):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.axhline(d["base"], color=INK2, ls="--", lw=1)
        ax.text(0.5, d["base"] + 1.6, f"unedited {d['base']:.1f}", ha="center", fontsize=7.2, color=INK2)
        ax.set_xticks(x, ["budget 10", "budget 50"])
        ax.set_title(name, color=INK, fontsize=10.5)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 80)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_replication.png", dpi=220)


def fig2():
    # reports/R.4a_repeats_s1..s4.md (score_a*.json): (recovery, retention, good answers)
    runs = {"b10": [(0.786, 1.04, 12), (0.941, -1.84, 40), (0.738, -0.55, 23)],
            "b25": [(1.086, -3.01, 43), (0.904, -0.57, 31), (0.765, -0.71, 24)],
            "b50": [(0.989, -2.49, 37), (0.578, -0.41, 17), (0.770, -2.06, 29)]}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.add_patch(matplotlib.patches.Rectangle((0.8, -2), 0.4, 3.6, color="#eef4fb", zorder=0))
    ax.axvline(0.8, color=GRID, lw=1); ax.axhline(-2, color=GRID, lw=1)
    for (lab, pts), col in zip(runs.items(), [BLUE, ORANGE, AQUA]):
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=[30 + 3 * p[2] for p in pts], color=col,
                   edgecolor=SURF, lw=1.5, label=f"Fable, {lab} passes", zorder=3)
    ax.scatter([0.963], [-0.69], marker="*", s=260, color=INK, edgecolor=SURF, lw=1, zorder=4, label="synthesis (saw all nine)")
    ax.scatter([1.048], [-1.89], marker="s", s=60, color=GRAY, zorder=3, label="corrective LoRA (trained)")
    ax.scatter([0.0], [0.0], marker="X", s=70, color=INK2, zorder=3)
    ax.annotate("broken router", (0.0, 0.0), (0.04, 0.45), fontsize=8.5, color=INK2)
    ax.annotate("base model: recovery 1.0,\nretention −16.2 (off chart)", (1.0, -4.0), (0.52, -3.75), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.annotate("b10 r2 (demo run)", (0.941, -1.84), (0.83, -1.45), fontsize=8, color=INK2)
    ax.text(1.19, 1.35, "meets both targets", ha="right", fontsize=8, color=INK2)
    ax.set_xlim(-0.05, 1.2); ax.set_ylim(-4.0, 1.6)
    ax.set_xlabel("irrelevance recovery (fraction of the lost 'no tool' behaviour won back)")
    ax.set_ylabel("call retention (points vs broken)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tradeoff.png", dpi=220)


def fig3():
    # reports/A.10.md, R.4a_P0.md, R.4a_repeats_s1..s4.md
    edits = [12, 40, 23, 43, 31, 24, 37, 17, 29]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    rows = ["base model", "after xLAM LoRA\n(as served)", "after xLAM LoRA\n(tool token banned)",
            "+ Fable edit\n(9 runs + synthesis)", "+ corrective LoRA\n(trained, A.7)"]
    y = list(range(len(rows)))[::-1]
    ax.barh(y[0], 81, color=GRAY, height=0.5); ax.text(82, y[0], "81", va="center", fontsize=8.5)
    ax.barh(y[1], 0.6, color=GRAY, height=0.5); ax.text(2, y[1], "0 (calls a tool every time)", va="center", fontsize=8.5)
    ax.barh(y[2], 11, color=ORANGE, height=0.5); ax.text(12, y[2], "11", va="center", fontsize=8.5)
    ax.scatter(edits, [y[3]] * len(edits), color=BLUE, s=36, zorder=3, edgecolor=SURF, lw=1)
    ax.scatter([31], [y[3]], marker="*", s=180, color=INK, zorder=4, edgecolor=SURF)
    ax.text(46, y[3], "12–43 (synthesis ★ 31)", va="center", fontsize=8.5)
    ax.barh(y[4], 75, color=GRAY, height=0.5); ax.text(76, y[4], "75", va="center", fontsize=8.5)
    ax.axvline(40, color=INK2, ls=":", lw=1); ax.text(40.5, 4.35, "target 40", fontsize=8, color=INK2)
    ax.set_yticks(y, rows, fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("good answers (judge helpfulness ≥ 4) on 240 prompts where no tool fits")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_answers.png", dpi=220)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))
