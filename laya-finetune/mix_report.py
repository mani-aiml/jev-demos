"""The training class mix at each data stage, against the generation design, as a table and a plot.

Two checks, both learned the hard way on this project:
1. clean vs defective: the near-test filter dropped mostly clean items, so training ran 20%
   clean against the 1-in-3 the data was generated with;
2. defect type vs defect type: adding 40 natural examples to 11 types doubled their count, and
   Laya then named those types 125 times and was right on only 65 of them (52%).
The target is the generation design (1 clean in 3, every defect type equal), never test labels.
"""

import collections
import os
import sys
from typing import Any

from finetune_data import DESIGN_CLEAN_SHARE, labels

TYPE_BAND = (0.5, 1.5)  # a defect type's share may sit between half and 1.5x the equal share
CLEAN_TOLERANCE = 0.05  # clean share may sit within 5 points of the design
SURFACE, INK, INK_2, MUTED, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a"
BEFORE, AFTER = "#3987e5", "#d95926"  # dark-mode categorical slots 1 and 2, validated together

Stage = tuple[str, list[dict[str, Any]], bool]  # (name, rows, use each row's weight)


def shares(rows: list[dict[str, Any]], weighted: bool, reference: str = "planted") -> dict[str, float]:
    """Each label's share of the total training weight (weight 1 per row when unweighted)."""
    total, out = 0.0, collections.defaultdict(float)
    for r in rows:
        w = r.get("weight", 1.0) if weighted else 1.0
        out[labels(r, reference)["criterion"]] += w
        total += w
    return {k: v / total for k, v in out.items()}


def problems(mix: dict[str, float]) -> list[str]:
    """What is off target, in plain words; empty when the mix is safe to train on."""
    found = []
    if abs(mix.get("none", 0.0) - DESIGN_CLEAN_SHARE) > CLEAN_TOLERANCE:
        found.append(f"clean share {mix.get('none', 0.0):.1%}, design {DESIGN_CLEAN_SHARE:.1%}")
    types = {k: v for k, v in mix.items() if k != "none"}
    equal = (1 - DESIGN_CLEAN_SHARE) / max(1, len(types))
    off = sorted(k for k, v in types.items() if not TYPE_BAND[0] * equal <= v <= TYPE_BAND[1] * equal)
    if off:
        found.append(f"{len(off)} defect types outside {TYPE_BAND[0]}x-{TYPE_BAND[1]}x the equal share: {', '.join(off)}")
    return found


def print_table(stages: list[Stage]) -> None:
    width = max(len(name) for name, _, _ in stages) + 2
    print(f"{'stage':{width}s} {'rows':>6} {'clean':>7} {'defect type min / max':>24}")
    for name, rows, weighted in stages:
        mix = shares(rows, weighted)
        types = [v for k, v in mix.items() if k != "none"]
        print(f"{name:{width}s} {len(rows):>6} {mix.get('none', 0.0):>6.1%} {min(types):>11.2%} / {max(types):.2%}")
    print(f"{'design':{width}s} {'':>6} {DESIGN_CLEAN_SHARE:>6.1%} {(1 - DESIGN_CLEAN_SHARE) / len(types):>11.2%} each")


def _clean_panel(ax, stages: list[Stage]) -> None:
    clean = [shares(rows, w).get("none", 0.0) * 100 for _, rows, w in stages]
    ax.bar(range(len(stages)), clean, color=[AFTER if w else BEFORE for _, _, w in stages], width=0.6)  # weighted stages stand out
    for i, v in enumerate(clean):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", color=INK, fontsize=13)
    ax.axhline(DESIGN_CLEAN_SHARE * 100, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(-0.3, DESIGN_CLEAN_SHARE * 100 + 2.5, "design: 1 in 3", color=MUTED, ha="left")  # above every bar label
    ax.set_ylim(0, DESIGN_CLEAN_SHARE * 100 + 6)
    ax.set_xticks(range(len(stages)), [name for name, _, _ in stages], rotation=20, ha="right")
    ax.set_ylabel("clean answers, % of training weight")
    ax.set_title("Clean vs defective", color=INK, loc="left", fontsize=15)


def _types_panel(ax, before: Stage, after: Stage) -> None:
    first, last = shares(before[1], before[2]), shares(after[1], after[2])
    types = sorted((k for k in first if k != "none"), key=lambda k: first[k])
    equal = (1 - DESIGN_CLEAN_SHARE) / len(types)
    for i, k in enumerate(types):
        ax.plot([first[k] * 100, last[k] * 100], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.scatter([first[k] * 100 for k in types], range(len(types)), s=64, color=BEFORE, label=before[0], zorder=2)
    ax.scatter([last[k] * 100 for k in types], range(len(types)), s=64, color=AFTER, label=after[0], zorder=3)
    ax.axvline(equal * 100, color=MUTED, linestyle="--", linewidth=1.5)
    ax.set_yticks(range(len(types)), [k.replace("_", " ") for k in types], fontsize=10)
    ax.set_xlabel("% of training weight")
    ax.set_title(f"Each defect type (equal share = {equal:.2%})", color=INK, loc="left", fontsize=15)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(stages: list[Stage], path: str | None = None, compare: tuple[int, int] = (0, -1)):
    """Left: clean share per stage. Right: every defect type, stage `compare[0]` vs `compare[1]`.

    Saved to `path` when given; the figure is returned so a notebook can show it inline.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED,
                         "ytick.color": INK_2, "axes.edgecolor": GRID, "font.size": 12})
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 11), width_ratios=[1, 1.5], facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
    _clean_panel(left, stages)
    _types_panel(right, stages[compare[0]], stages[compare[1]])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def preflight(rows: list[dict], mix: str, out: str | None, allow_skew: bool) -> None:
    """Show the class mix this run will train on, and refuse a skewed one unless told otherwise."""
    import matplotlib
    matplotlib.use("Agg")  # headless: the plot goes to the run folder

    stages = [("as loaded", rows, False), (f"as trained (--mix {mix})", rows, mix != "none")]
    print_table(stages)
    if out:
        os.makedirs(out, exist_ok=True)
        plot(stages, os.path.join(out, "mix.png"))
    found = problems(shares(rows, weighted=mix != "none"))
    if found and not allow_skew:
        sys.exit("mix is off the design, not training:\n  " + "\n  ".join(found)
                 + "\nfix it with --mix design or --mix balanced, or pass --allow-skew")
