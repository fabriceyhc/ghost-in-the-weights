"""
Revised figure generation for "Ghost in the Weights" experiments.
Each figure is designed to make the key finding immediately visible.

Run from project root:
    python artifacts/figures/generate_figures.py
"""

import json, os, glob, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

ROOT   = "/data2/fabricehc/ghost-in-the-weights"
RDIR   = os.path.join(ROOT, "data/results")
OUTDIR = os.path.join(ROOT, "artifacts/figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ── model catalogue ───────────────────────────────────────────────────────────
# (display_name, file_stem, n_params_B, family, is_instruct)
MODELS = [
    ("Qwen3-0.6B",       "Qwen_Qwen3-0.6B",                      0.6,  "Qwen3",  True),
    ("Qwen3-0.6B-Base",  "Qwen_Qwen3-0.6B-Base",                 0.6,  "Qwen3",  False),
    ("Qwen3-1.7B",       "Qwen_Qwen3-1.7B",                      1.7,  "Qwen3",  True),
    ("Qwen3-1.7B-Base",  "Qwen_Qwen3-1.7B-Base",                 1.7,  "Qwen3",  False),
    ("Qwen3-4B",         "Qwen_Qwen3-4B",                        4.0,  "Qwen3",  True),
    ("Qwen3-4B-Base",    "Qwen_Qwen3-4B-Base",                   4.0,  "Qwen3",  False),
    ("Qwen3-8B",         "Qwen_Qwen3-8B",                        8.0,  "Qwen3",  True),
    ("Qwen3-8B-Base",    "Qwen_Qwen3-8B-Base",                   8.0,  "Qwen3",  False),
    ("Qwen3-14B",        "Qwen_Qwen3-14B",                      14.0,  "Qwen3",  True),
    ("Qwen3-14B-Base",   "Qwen_Qwen3-14B-Base",                 14.0,  "Qwen3",  False),
    ("Llama-3.2-1B",     "meta-llama_Llama-3.2-1B",              1.0,  "Llama3", False),
    ("Llama-3.2-1B-I",   "meta-llama_Llama-3.2-1B-Instruct",     1.0,  "Llama3", True),
    ("Llama-3.2-3B",     "meta-llama_Llama-3.2-3B",              3.0,  "Llama3", False),
    ("Llama-3.2-3B-I",   "meta-llama_Llama-3.2-3B-Instruct",     3.0,  "Llama3", True),
    ("Llama-3.1-8B",     "meta-llama_Llama-3.1-8B",              8.0,  "Llama3", False),
    ("Llama-3.1-8B-I",   "meta-llama_Llama-3.1-8B-Instruct",     8.0,  "Llama3", True),
    ("Llama-3.1-70B",    "meta-llama_Llama-3.1-70B",            70.0,  "Llama3", False),
    ("Llama-3.1-70B-I",  "meta-llama_Llama-3.1-70B-Instruct",   70.0,  "Llama3", True),
    ("Gemma-3-1b-pt",    "google_gemma-3-1b-pt",                 1.0,  "Gemma3", False),
    ("Gemma-3-1b-it",    "google_gemma-3-1b-it",                 1.0,  "Gemma3", True),
    ("Gemma-3-4b-pt",    "google_gemma-3-4b-pt",                 4.0,  "Gemma3", False),
    ("Gemma-3-4b-it",    "google_gemma-3-4b-it",                 4.0,  "Gemma3", True),
]
FAM_COLOR  = {"Qwen3": "#2166ac", "Llama3": "#d6604d", "Gemma3": "#4dac26"}
FAM_LABEL  = {"Qwen3": "Qwen3",   "Llama3": "Llama-3", "Gemma3": "Gemma-3"}

def load(exp, stem):
    p = os.path.join(RDIR, exp, f"{exp}_{stem}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

def fam_legend(ax, loc="best", extra=None):
    els = [mpatches.Patch(color=c, label=FAM_LABEL[f]) for f, c in FAM_COLOR.items()]
    if extra:
        els += extra
    ax.legend(handles=els, fontsize=8.5, loc=loc, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Overview heatmap, rows sorted by signal count (most interesting first)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_overview():
    EXP_COLS = ["E1\nTemporal", "E2\nGramm.", "E3\nGeometry",
                "E4\nEasy", "E4\nMed", "E4\nHard", "E5\nMeta"]

    def score_row(stem, family, nparams, is_inst):
        row, notes = [], []
        # E1
        d = load("exp1_temporal", stem)
        if d:
            p = d["comparisons"].get("Self vs Matched-AI (critical test)", {}).get("p_value", 1.0)
            row.append(1 if p < 0.05 else 0)
        else:
            row.append(3)
        # E2
        d = load("exp2_grammatical", stem)
        if d:
            st = d["decomposition"]["statistical_tests"]
            sig = any(float(st[k]["p"]) < 0.05 for k in st)
            row.append(1 if sig else 0)
        else:
            row.append(3)
        # E3
        d = load("exp3_geometry", stem)
        if d is None:
            row.append(3)
        elif d.get("significant") is True or str(d.get("significant")).lower() == "true":
            row.append(1)
        else:
            row.append(0)
        # E4 three difficulties
        d = load("exp4_behavioral", stem)
        gemma4b = (family == "Gemma3" and nparams >= 4)
        if d:
            for diff in ["easy", "medium", "hard"]:
                dd = d["difficulties"].get(diff, {})
                n, p, acc = dd.get("n", 0), dd.get("p_value", 1.0), dd.get("accuracy", 0.33)
                if n == 0:      row.append(3)
                elif gemma4b:   row.append(-1)
                else:           row.append(1 if (p < 0.05 and acc > 0.333) else 0)
        else:
            row += [3, 3, 3]
        # E5
        d = load("exp5_metacognitive", stem)
        if d:
            ps = d["causation"]["per_strength"]
            max_beh = max(v["behavior_change_rate"] for v in ps.values())
            row.append(2 if (max_beh >= 0.35 and is_inst) else 0)
        else:
            row.append(3)
        n_sig = sum(1 for v in row if v == 1)
        return row, n_sig

    rows_data = []
    for name, stem, nparams, family, is_inst in MODELS:
        row, n_sig = score_row(stem, family, nparams, is_inst)
        rows_data.append((n_sig, name, stem, nparams, family, is_inst, row))

    # Sort: most signals first, then alphabetically
    rows_data.sort(key=lambda x: (-x[0], x[4], x[2]))

    mat = np.array([r[-1] for r in rows_data], dtype=float)
    labels_y = [r[1] for r in rows_data]
    fams_y   = [r[4] for r in rows_data]
    insts_y  = [r[5] for r in rows_data]

    cmap = matplotlib.colors.ListedColormap(["#f0f0f0", "#1a9641", "#f4a900", "#e8e8e8"])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    mat_disp = np.where(mat == -1, 0, mat)

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(mat_disp, cmap=cmap, norm=norm, aspect="auto")

    for i in range(len(rows_data)):
        for j in range(len(EXP_COLS)):
            if mat[i, j] == -1:
                ax.add_patch(plt.Rectangle([j-.5, i-.5], 1, 1,
                             fill=False, hatch="////", edgecolor="#bbbbbb", lw=0))
            if mat[i, j] == 3:
                ax.add_patch(plt.Rectangle([j-.5, i-.5], 1, 1,
                             facecolor="#e8e8e8", edgecolor="none"))
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#aaaaaa")

    # Highlight rows with any signal
    for i, (n_sig, *_) in enumerate(rows_data):
        if n_sig > 0:
            ax.add_patch(plt.Rectangle([-0.5, i-0.5], len(EXP_COLS), 1,
                         fill=False, edgecolor="#333333", lw=1.2, zorder=5))

    ax.set_xticks(range(len(EXP_COLS)))
    ax.set_xticklabels(EXP_COLS, ha="center")
    ax.set_yticks(range(len(rows_data)))
    ax.set_yticklabels(labels_y, fontsize=8.5)
    for i, (fam, inst) in enumerate(zip(fams_y, insts_y)):
        ax.get_yticklabels()[i].set_color(FAM_COLOR[fam])
        if not inst:
            ax.get_yticklabels()[i].set_style("italic")

    # Signal count annotation
    for i, (n_sig, *_) in enumerate(rows_data):
        if n_sig > 0:
            ax.annotate(f"{n_sig}✓", xy=(len(EXP_COLS) - 0.5 + 0.35, i),
                        fontsize=8, va="center", color="#333333",
                        annotation_clip=False, xycoords="data")

    ax.set_xticks(np.arange(-.5, len(EXP_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows_data), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(2.5, color="white", lw=2.5)
    ax.axvline(5.5, color="white", lw=2.5)

    # Bracket for E4 sub-columns — must use y > n_rows because imshow y=0 is at top
    n_rows = len(rows_data)
    ax.annotate("", xy=(3-.45, n_rows + 0.9), xytext=(5+.45, n_rows + 0.9),
                arrowprops=dict(arrowstyle="-", color="#555555", lw=1.2),
                annotation_clip=False)
    ax.annotate("E4: Behavioral Self-Recognition", xy=(4, n_rows + 1.7),
                ha="center", fontsize=8, color="#555555",
                annotation_clip=False, xycoords="data")

    legend_els = [
        mpatches.Patch(facecolor="#1a9641",             label="Significant signal"),
        mpatches.Patch(facecolor="#f4a900",             label="RLHF artifact (E5 only)"),
        mpatches.Patch(facecolor="#f0f0f0", ec="#aaa",  label="Null"),
        mpatches.Patch(facecolor="#e8e8e8", hatch="////", ec="#bbb", label="Unreliable / N/A"),
        plt.Line2D([0],[0], color="#333333", lw=1.2, label="Row has ≥1 signal"),
    ]
    ax.legend(handles=legend_els, loc="lower left", bbox_to_anchor=(0, -0.22),
              ncol=3, frameon=False, fontsize=8.5)

    ax.set_title("Cross-Experiment Overview  ·  22 Models × 5 Experiments\n"
                 "(sorted by signal count; italic = base; color = family)",
                 fontsize=11, pad=14)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig1_overview.{ext}"))
    plt.close(fig)
    print("✓ fig1_overview")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — E1 v2: Corrected metric — last-token linear probe accuracy
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_e1_temporal():
    V2DIR = os.path.join(RDIR, "exp1_temporal_v2")

    # Collect all v2 results
    records = []
    for name, stem, nparams, family, is_inst in MODELS:
        path = os.path.join(V2DIR, f"exp1v2_temporal_{stem}.json")
        if not os.path.exists(path):
            continue
        d     = json.load(open(path))
        pts   = d["probe_points"]
        sva   = d["self_vs_matched_ai"]
        negf  = d["named_entity_vs_generic_fact"]
        records.append(dict(
            name=name, family=family, is_inst=is_inst, nparams=nparams,
            pts=pts,
            acc_sva  = [sva[str(T)]["linear_probe"]["accuracy"]  for T in pts],
            acc_negf = [negf[str(T)]["linear_probe"]["accuracy"] for T in pts],
        ))

    pts_arr = np.array(records[0]["pts"]) if records else np.array([100,250,500,1000,2000])

    fig, ax = plt.subplots(figsize=(8, 5))

    diffs = np.array([np.array(r["acc_sva"]) - np.array(r["acc_negf"])
                      for r in records])          # (n_models, n_pts)
    mean_diff = diffs.mean(0)
    std_diff  = diffs.std(0)

    # Shaded ±1 std band across all models
    ax.fill_between(pts_arr, mean_diff - std_diff, mean_diff + std_diff,
                    color="#888888", alpha=0.15, label="±1 SD across models")

    # Individual model lines (faint, coloured by family)
    for r, diff_row in zip(records, diffs):
        col    = FAM_COLOR[r["family"]]
        ls     = "-" if r["is_inst"] else "--"
        is_gem = r["family"] == "Gemma3"
        ax.plot(pts_arr, diff_row, color=col, ls=ls,
                lw=2.0 if is_gem else 1.0,
                alpha=1.0 if is_gem else 0.45,
                marker="o" if r["is_inst"] else "s", ms=4 if is_gem else 3,
                zorder=4 if is_gem else 2)

    # Mean line across all models
    ax.plot(pts_arr, mean_diff, color="#333333", lw=2.2, ls="-",
            marker="D", ms=5, zorder=5, label="Mean across models")

    # Zero reference
    ax.axhline(0, color="#333333", lw=1.2, ls="--", zorder=1)
    ax.fill_between(pts_arr, -0.03, 0.03, color="#333333", alpha=0.05,
                    label="±0.03 tolerance band")

    ax.set_xlabel("Context distance T (tokens after statement)")
    ax.set_ylabel("Probe accuracy difference\n(self vs. matched-AI)  −  (named-entity vs. generic-fact)")
    ax.set_title("Self-advantage over control = 0 at all distances\n"
                 "(zero line = null; positive = self persists longer)")
    ax.set_xticks(pts_arr)
    ax.set_ylim(-0.18, 0.22)
    ax.spines[["top","right"]].set_visible(False)

    legend_els = [mpatches.Patch(color=c, label=FAM_LABEL[f]) for f, c in FAM_COLOR.items()]
    legend_els += [
        plt.Line2D([0],[0], color="#333", lw=2.2, marker="D", ms=5, label="Mean"),
        plt.Line2D([0],[0], color="gray", lw=1.0, ls="-",  label="Instruct"),
        plt.Line2D([0],[0], color="gray", lw=1.0, ls="--", label="Base"),
    ]
    ax.legend(handles=legend_els, fontsize=8.5, frameon=False, ncol=2,
              loc="upper right")

    # Annotate Gemma
    for r, diff_row in zip(records, diffs):
        if r["family"] == "Gemma3":
            ax.annotate(r["name"], xy=(pts_arr[-1], diff_row[-1]),
                        xytext=(-55, 8 if diff_row[-1] > 0 else -18),
                        textcoords="offset points",
                        fontsize=8, color=FAM_COLOR["Gemma3"],
                        arrowprops=dict(arrowstyle="->",
                                        color=FAM_COLOR["Gemma3"], lw=0.8))

    fig.suptitle("Experiment 1: Temporal Persistence of Self-Representations",
                 fontsize=12, y=1.01, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig2_e1_temporal.{ext}"))
    plt.close(fig)
    print("✓ fig2_e1_temporal")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — E3: Geometry (size-scaling as centerpiece)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_e3_geometry():
    data = []
    for name, stem, nparams, family, is_inst in MODELS:
        d = load("exp3_geometry", stem)
        if d is None:
            continue
        eff = d.get("divergence_effect")
        pv  = d.get("p_value")
        if eff is None or pv is None:
            continue
        data.append((name, stem, nparams, family, is_inst, eff, pv))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for name, stem, nparams, family, is_inst, eff, pv in data:
        col    = FAM_COLOR[family]
        marker = "o" if is_inst else "s"
        ec     = "#111111" if pv < 0.05 else "none"
        lw     = 1.8 if pv < 0.05 else 0
        size   = 120 if pv < 0.05 else 60
        ax.scatter(nparams, eff, color=col, marker=marker, s=size,
                   edgecolors=ec, linewidths=lw, alpha=0.85, zorder=4)
        if pv < 0.05:
            label = (name.replace("-Base", "").replace("-Instruct", "").replace("Llama-3.", "L3.")
                     + (" (B)" if not is_inst else ""))
            # Instruct labels above point, base labels below — separates the 14B pair
            yoff = 8 if is_inst else -18
            ax.annotate(label, xy=(nparams, eff),
                        xytext=(8, yoff), textcoords="offset points",
                        fontsize=8.5, color=col,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ax.axhline(0, color="#999999", lw=1, ls="--")
    ax.axvspan(14, 80, color="#e8f5e9", alpha=0.45, zorder=1,
               label="≥14B: significant models appear here")
    ax.axvline(14, color="#4caf50", lw=1.5, ls=":", zorder=2)
    ax.text(14.5, ax.get_ylim()[0] if ax.get_ylim()[0] > -20 else -15,
            "14B threshold", fontsize=8, color="#388e3c", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Model size (B parameters)")
    ax.set_ylabel("Self/other divergence effect (permutation test)")
    ax.set_title("Geometry Signal Scales with Model Size\n"
                 "(only ≥14B models reach significance)")
    xticks = [0.6, 1, 3, 8, 14, 70]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])

    legend_els = [mpatches.Patch(color=c, label=FAM_LABEL[f]) for f, c in FAM_COLOR.items()]
    legend_els += [
        plt.Line2D([0],[0], marker="o", color="none", mfc="#555", ms=8, label="Instruct"),
        plt.Line2D([0],[0], marker="s", color="none", mfc="#555", ms=8, label="Base"),
        plt.Line2D([0],[0], marker="o", color="none", mfc="none",
                   mec="black", mew=1.8, ms=10, label="p < 0.05"),
    ]
    ax.legend(handles=legend_els, fontsize=8.5, frameon=False, loc="upper left")

    fig.suptitle("Experiment 3: Self/Other Geometry — Emerges Only at ≥14B Parameters",
                 fontsize=12, y=1.01, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig3_e3_geometry.{ext}"))
    plt.close(fig)
    print("✓ fig3_e3_geometry")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — E4: Self-recognition — strip plot (shows clustering at chance)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_e4_behavioral():
    diffs     = ["easy", "medium", "hard"]
    diff_x    = {"easy": 0, "medium": 1, "hard": 2}
    CHANCE    = 1/3

    # Collect data
    all_pts = []
    for name, stem, nparams, family, is_inst in MODELS:
        d = load("exp4_behavioral", stem)
        if d is None:
            continue
        gemma4b = (family == "Gemma3" and nparams >= 4)
        for diff in diffs:
            dd = d["difficulties"].get(diff, {})
            n, acc, pv = dd.get("n", 0), dd.get("accuracy"), dd.get("p_value", 1.0)
            if acc is None or n == 0:
                continue
            all_pts.append(dict(name=name, diff=diff, acc=acc, p=pv,
                                family=family, is_inst=is_inst,
                                nparams=nparams, unreliable=gemma4b, n=n))

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Jitter x within each difficulty group; record exact position for sig points
    rng = np.random.default_rng(42)
    sig_pos = {}   # (name, diff) -> actual (x, y) used in scatter
    for pt in all_pts:
        xbase = diff_x[pt["diff"]]
        jitter = rng.uniform(-0.28, 0.28)
        col  = FAM_COLOR[pt["family"]]
        mark = "o" if pt["is_inst"] else "s"
        ec   = "none"
        lw   = 0
        alpha = 0.35 if pt["unreliable"] else 0.80
        size  = 50

        if pt["p"] < 0.05 and pt["acc"] > CHANCE and not pt["unreliable"]:
            ec, lw, size = "#111111", 2.0, 110  # highlight significant
            sig_pos[(pt["name"], pt["diff"])] = (xbase + jitter, pt["acc"])

        ax.scatter(xbase + jitter, pt["acc"], color=col, marker=mark,
                   s=size, edgecolors=ec, linewidths=lw, alpha=alpha, zorder=3)

    # Chance line
    ax.axhline(CHANCE, color="#333333", lw=2, ls="--", zorder=2,
               label=f"Chance ({CHANCE:.2f})")
    ax.fill_between([-0.5, 2.5], CHANCE - 0.04, CHANCE + 0.04,
                    color="#999999", alpha=0.10, zorder=1)

    # Annotate the significant results — spread boxes across top of figure
    sig_pts = [p for p in all_pts if p["p"] < 0.05 and p["acc"] > CHANCE and not p["unreliable"]]
    # Sort by (column, acc desc) so each box is assigned to the side of its own column
    sig_pts_sorted = sorted(sig_pts, key=lambda p: (diff_x[p["diff"]], p["acc"]))
    # Place each box close to its column: two medium boxes straddle the column
    box_x = [0.55, 1.45, 2.45]   # left-of-medium, right-of-medium, right-of-hard
    box_y = 0.61
    for idx, pt in enumerate(sig_pts_sorted):
        label = f"{pt['name']}\nacc={pt['acc']:.2f}, p={pt['p']:.3f}"
        tx = box_x[idx % len(box_x)]
        arrow_xy = sig_pos.get((pt["name"], pt["diff"]),
                               (diff_x[pt["diff"]], pt["acc"]))
        ax.annotate(label, xy=arrow_xy,
                    xytext=(tx, box_y),
                    fontsize=8, color=FAM_COLOR[pt["family"]],
                    arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9,
                              ec=FAM_COLOR[pt["family"]], lw=0.8))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Easy\n(cross-family\ndistractors)",
                         "Medium\n(same-family\ndistractors)",
                         "Hard\n(same-size\ndistractors)"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0.12, 0.70)
    ax.set_ylabel("Accuracy")

    # Right-axis tick at chance
    ax.set_yticks([0.2, CHANCE, 0.4, 0.5, 0.6])
    ax.set_yticklabels(["0.20", "0.33\n(chance)", "0.40", "0.50", "0.60"])

    legend_els = [mpatches.Patch(color=c, label=FAM_LABEL[f]) for f, c in FAM_COLOR.items()]
    legend_els += [
        plt.Line2D([0],[0], marker="o", color="none", mfc="#555", ms=8, label="Instruct"),
        plt.Line2D([0],[0], marker="s", color="none", mfc="#555", ms=8, label="Base"),
        plt.Line2D([0],[0], marker="o", color="none", mfc="none",
                   mec="black", mew=2.0, ms=11, label="p < 0.05"),
        mpatches.Patch(color="#999999", alpha=0.35, label="Unreliable (Gemma-4b)"),
    ]
    ax.legend(handles=legend_els, fontsize=8.5, frameon=False,
              loc="lower right", ncol=2)

    ax.set_title("Experiment 4: Behavioral Self-Recognition\n"
                 "20/22 models at chance; 2 partial signals at medium difficulty",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig4_e4_behavioral.{ext}"))
    plt.close(fig)
    print("✓ fig4_e4_behavioral")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — E5: Metacognitive steering — dissociation is the story
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_e5_metacognitive():
    alphas = [0.5, 1.0, 2.0, 4.0]

    # Collect per-model data
    records = []
    for name, stem, nparams, family, is_inst in MODELS:
        d = load("exp5_metacognitive", stem)
        if d is None:
            continue
        ps = d["causation"]["per_strength"]
        beh  = [ps.get(str(a), {}).get("behavior_change_rate", 0.0) for a in alphas]
        mech = [ps.get(str(a), {}).get("mechanism_change_rate", 0.0) for a in alphas]
        records.append(dict(name=name, family=family, is_inst=is_inst,
                            nparams=nparams, beh=beh, mech=mech,
                            max_beh=max(beh), max_mech=max(mech)))

    # Sort records by max_beh descending
    sorted_rec = sorted(records, key=lambda r: -r["max_beh"])
    y = np.arange(len(sorted_rec))

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, r in enumerate(sorted_rec):
        col   = FAM_COLOR[r["family"]]
        # Line from mech (0) to beh
        ax.plot([r["max_mech"], r["max_beh"]], [y[i], y[i]],
                color=col, lw=2.0, alpha=0.5, zorder=2)
        # Behavioral dot (filled)
        marker = "o" if r["is_inst"] else "s"
        ax.scatter(r["max_beh"], y[i], color=col, s=70, marker=marker,
                   alpha=0.85, zorder=4)
        # Mechanistic dot (empty, grey)
        ax.scatter(r["max_mech"], y[i], color="#aaaaaa", s=55, marker=marker,
                   alpha=0.7, zorder=3, edgecolors="#777777", lw=1)

    ax.axvline(0, color="#555555", lw=1.5, ls="--")
    ax.axvline(0.35, color="#f4a900", lw=1.5, ls=":", label="RLHF threshold (beh)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in sorted_rec], fontsize=8)
    for i, r in enumerate(sorted_rec):
        ax.get_yticklabels()[i].set_color(FAM_COLOR[r["family"]])
        if not r["is_inst"]:
            ax.get_yticklabels()[i].set_style("italic")
    ax.set_xlabel("Rate (fraction of 30 prompts)")
    ax.set_title("Behavioral vs Mechanistic Change (max across steering strength α)\n"
                 "Colored dot = behavior  ·  Gray dot = mechanism (always 0)")
    ax.set_xlim(-0.04, 0.60)

    # Annotate zero-mech baseline — no arrow needed, text is self-explanatory
    ax.text(0.97, 0.64, "Mechanistic change\n= 0.00 for ALL models",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"))

    beh_patch  = plt.Line2D([0],[0], marker="o", color="none", mfc="#555", ms=9,
                             label="Max behavioral Δ (instruct)")
    beh_patch2 = plt.Line2D([0],[0], marker="s", color="none", mfc="#555", ms=9,
                             label="Max behavioral Δ (base)")
    mech_patch = plt.Line2D([0],[0], marker="o", color="none", mfc="#aaa",
                             mec="#777", mew=1, ms=9, label="Max mechanistic Δ")
    ax.legend(handles=[beh_patch, beh_patch2, mech_patch], fontsize=8.5, frameon=False)

    fig.suptitle("Experiment 5: Metacognitive Steering — Behavior Changes, Mechanism Never Does",
                 fontsize=12, y=1.01, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig5_e5_metacognitive.{ext}"))
    plt.close(fig)
    print("✓ fig5_e5_metacognitive")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — E2: Focus on the 3 exceptions vs a null contrast
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_e2_grammatical():
    # Show 2×2 bar plots for the 3 significant models + 1 null example
    focal = [
        ("Llama-3.2-1B-I",  "meta-llama_Llama-3.2-1B-Instruct",  "Llama3", True,
         "Referent effect*\n(other > self)", "#d73027"),
        ("Llama-3.2-3B",    "meta-llama_Llama-3.2-3B",           "Llama3", False,
         "Referent effect*\n(self > other; opposite dir.)", "#f46d43"),
        ("Llama-3.1-70B",   "meta-llama_Llama-3.1-70B",          "Llama3", False,
         "Person effect*\n(3rd > 1st; wrong direction)", "#4575b4"),
        ("Qwen3-8B",        "Qwen_Qwen3-8B",                     "Qwen3",  True,
         "Null example\n(all cells ≈ equal)", "#999999"),
    ]

    cell_keys    = ["1p_self_ai",  "1p_other_ai",  "2p_self_ai",  "2p_other_ai"]
    cell_xlabels = ["1st\nSelf", "1st\nOther", "2nd\nSelf", "2nd\nOther"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5.5), sharey=False)

    for ax, (mname, stem, family, is_inst, title, highlight_col) in zip(axes, focal):
        d = load("exp2_grammatical", stem)
        if d is None:
            ax.set_title(mname + "\n(no data)")
            continue
        cm    = d["decomposition"]["cell_means"]
        stats = d["decomposition"]["statistical_tests"]
        vals  = [cm.get(k, 0) for k in cell_keys]
        vmin  = min(vals) - 0.02
        vmax  = max(vals) + 0.02
        yrange = vmax - vmin

        # Color bars by 1st vs 2nd person
        colors = ["#c6dbef", "#9ecae1", "#a1d99b", "#74c476"]
        bars = ax.bar(range(4), vals, color=colors, width=0.65, edgecolor="#555555", lw=0.5)

        # Annotate max bar with highlight
        max_idx = int(np.argmax(vals))
        min_idx = int(np.argmin(vals))
        bars[max_idx].set_edgecolor(highlight_col)
        bars[max_idx].set_linewidth(2.5)

        # p-value annotations
        p_person   = float(stats["person_test"]["p"])
        p_referent = float(stats["referent_test"]["p"])
        p_inter    = float(stats["interaction_test"]["p"])
        ann_lines = [
            f"Person p={p_person:.3f}{sig_stars(p_person)}",
            f"Referent p={p_referent:.3f}{sig_stars(p_referent)}",
            f"Interact p={p_inter:.3f}{sig_stars(p_inter)}",
        ]
        ann_txt = "\n".join(ann_lines)
        ax.text(0.5, -0.26, ann_txt, transform=ax.transAxes,
                fontsize=7.5, ha="center", va="top", color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
                clip_on=False)

        ax.set_xticks(range(4))
        ax.set_xticklabels(cell_xlabels, fontsize=8)
        ax.set_ylim(vmin - yrange * 0.05, vmax + yrange * 0.45)
        ax.set_ylabel("Restoration slope" if ax == axes[0] else "")
        ax.set_title(f"{mname}\n{title}", fontsize=9,
                     color=highlight_col if highlight_col != "#999999" else "#555555")

        # Shade 1st-person columns
        ax.axvspan(-0.5, 1.5, color="#eeeeee", alpha=0.4, zorder=0)
        ax.axvspan(1.5, 3.5, color="#ffffff", alpha=0, zorder=0)
        if ax == axes[0]:
            ax.text(0.5, vmax + yrange * 0.12, "1st person", ha="center",
                    fontsize=7.5, color="#555555")
            ax.text(2.5, vmax + yrange * 0.12, "2nd person", ha="center",
                    fontsize=7.5, color="#555555")

    fig.suptitle("Experiment 2: Grammatical Person — 3 Exceptions, Inconsistent Across Models\n"
                 "(no model shows the expected 1st-person × self interaction)",
                 fontsize=11, y=1.03, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig6_e2_grammatical.{ext}"))
    plt.close(fig)
    print("✓ fig6_e2_grammatical")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig1_overview()
    fig2_e1_temporal()
    fig3_e3_geometry()
    fig4_e4_behavioral()
    fig5_e5_metacognitive()
    fig6_e2_grammatical()
    print(f"\nAll figures saved to {OUTDIR}/")
    for f in sorted(os.listdir(OUTDIR)):
        if f.endswith((".pdf", ".png")):
            sz = os.path.getsize(os.path.join(OUTDIR, f)) // 1024
            print(f"  {f}  ({sz} KB)")
