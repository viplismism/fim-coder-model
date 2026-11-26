#!/usr/bin/env python3
"""Charts for blog/blog.md, from the published result files only (results/, runs/). Writes blog/charts/*.png.
Usage: python blog/make_charts.py   (needs matplotlib; e.g. python3 -m venv .venv-blog && .venv-blog/bin/pip install matplotlib)
Palette: the validated reference categorical order; colour follows the entity across every chart."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "blog" / "charts"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = "#191919", "#666666", "#e6e6e6", "#ffffff"
C = {"base 14B": "#257de0", "random 14B": "#f47230", "AST 14B": "#00b894", "base 32B": "#f2a600", "DeepSeek 6.7B": "#ee7fac",
     "training holes": "#6a4ce2", "real edits (test)": "#008a17"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def frame(ax, ygrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)


def bars(ax, labels, values, fmt="{:.1f}", width=0.72, colors=None):
    """Thin-ish bars, 4px-ish rounded top, 2px surface gap via edgecolor, value label in text ink."""
    xs = range(len(labels))
    for x, lab, v in zip(xs, labels, values):
        c = (colors or C)[lab] if colors is None or isinstance(colors, dict) else colors[x]
        ax.add_patch(FancyBboxPatch((x - width / 2, 0), width, v, boxstyle="round,pad=0,rounding_size=0.06",
                                    mutation_aspect=max(values) / 6, fc=c, ec=SURF, lw=2))
        ax.text(x, v + max(values) * 0.015, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=11)
    ax.set_xlim(-0.6, len(labels) - 0.4); ax.set_xticks(list(xs)); ax.set_xticklabels(labels, color=INK)


fs = json.load(open(ROOT / "results/facet/summary.json"))
cp = json.load(open(ROOT / "results/controlplane/summary.json"))

# 1. pass@1, two repos
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
fr = fs["metrics"]["build_test_pass/all"]["extra"]["rate"]
cr = cp["metrics"]["build_test_pass/all"]["rate"]
panels = [("facet-rs/facet (Rust): 1,056 edits, 391 commits",
           ["base 14B", "random 14B", "AST 14B", "base 32B"],
           [fr["qwen14b-base"], fr["facet-random"], fr["facet-ast"], fr["qwen32b-base"]]),
          ("telekom/controlplane (Go): 632 edits, 193 commits",
           ["base 14B", "random 14B", "DeepSeek 6.7B", "base 32B"],
           [cr["qwen14b-base"], cr["qwen14b-ft"], cr["deepseek67b-base"], cr["qwen32b-base"]])]
for ax, (title, labs, vals) in zip(axs, panels):
    frame(ax); bars(ax, labs, [100 * v for v in vals], "{:.1f}%")
    ax.set_title(title, loc="left", fontsize=12); ax.set_ylim(0, 80)
axs[0].set_ylabel("pass@1: builds + tests pass (%)")
fig.text(0.01, 0.005, "'random 14B' = Qwen2.5-Coder-14B + LoRA trained on the repo's own pre-cutoff code (random FIM holes). "
         "Greedy, max 256 new tokens.", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "pass_at_1.png", dpi=200); plt.close(fig)

# 2. the stop effect
sc = fs["stop_check"]["by_model_that_hit_limit"]["qwen14b-base"]
labs = ["base 14B", "random 14B", "AST 14B", "base 32B"]
vals = [sc["passes"][k] for k in ("qwen14b-base", "facet-random", "facet-ast", "qwen32b-base")]
fig, ax = plt.subplots(figsize=(7.5, 4)); frame(ax); bars(ax, labs, vals, "{:d}")
ax.set_ylim(0, 100); ax.set_ylabel("examples passing")
ax.set_title(f"facet: the {sc['n']} examples where base 14B ran into the 256-token limit", loc="left", fontsize=12)
fig.text(0.01, 0.01, "Base 14B keeps writing past the end of the hole; the fine-tunes (and the 32B) know where to stop.", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "stop_effect.png", dpi=200); plt.close(fig)

# 3. AST - random across languages (Real-FIM-Eval Add, char perplexity), repo-level CIs
langs = [("Rust", "final_eval_summary.json"), ("Python", "python_final_eval_summary.json"),
         ("TypeScript", "typescript_final_eval_summary.json"), ("Go", "go_final_eval_summary.json"), ("C++", "cpp_final_eval_summary.json")]
# primary rows (Real-FIM-Eval Add) from the canonical summary table
rows = {}
for line in open(ROOT / "results/multilang/summary.md"):
    p = [x.strip() for x in line.split("|")]
    if len(p) > 8 and p[1] in dict(langs) and p[6].startswith(("+", "-")):
        lo, hi = [float(x) for x in p[7].strip("[]").split(",")]
        rows[p[1]] = (float(p[6]), lo, hi)
fig, ax = plt.subplots(figsize=(7.5, 3.8)); frame(ax, ygrid=False); ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
names = [n for n, _ in langs][::-1]
for y, n in enumerate(names):
    v, lo, hi = rows[n]
    ax.plot([lo * 1000, hi * 1000], [y, y], color=C["base 14B"], lw=2, solid_capstyle="round")
    ax.plot(v * 1000, y, "o", ms=8, color=C["base 14B"], mec=SURF, mew=2)
    ax.text(hi * 1000 + 0.12, y, f"{v * 1000:+.1f}", va="center", color=INK, fontsize=10)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, color=INK); ax.set_xlim(-1.2, 4.2)
ax.set_xlabel("AST − random, char perplexity × 1000  (right of 0 = random holes better)")
ax.set_title("AST holes never beat random holes", loc="left", fontsize=12)
fig.text(0.01, 0.01, "DeepSeek-Coder-6.7B + LoRA per arm, Real-FIM-Eval Add; dot = estimate, line = 95% CI over repos.", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "ast_vs_random_languages.png", dpi=200); plt.close(fig)

# 4. hole sizes: training holes vs real post-cutoff edits
hs = json.load(open(ROOT / "results/facet/hole_sizes.json"))
bk = hs["buckets"]; a = hs["train_ast"]["pct"]; t = hs["test_sensitive"]["pct"]
fig, ax = plt.subplots(figsize=(8.5, 4)); frame(ax)
w = 0.38
for i, (lab, d) in enumerate((("training holes", a), ("real edits (test)", t))):
    for j, b in enumerate(bk):
        x = j + (i - 0.5) * w
        ax.add_patch(FancyBboxPatch((x - w / 2 + 0.02, 0), w - 0.04, d[b], boxstyle="round,pad=0,rounding_size=0.04",
                                    mutation_aspect=12, fc=C[lab], ec=SURF, lw=2, label=lab if j == 0 else None))
        ax.text(x, d[b] + 0.8, f"{d[b]:.0f}", ha="center", va="bottom", color=INK, fontsize=9)
ax.set_xlim(-0.6, len(bk) - 0.4); ax.set_ylim(0, 56); ax.set_xticks(range(len(bk))); ax.set_xticklabels(bk, color=INK)
ax.set_xlabel("hole size (lines)"); ax.set_ylabel("% of holes")
ax.legend(frameon=False, loc="upper right", labelcolor=INK)
ax.set_title(f"facet: training holes (median {hs['train_ast']['median_lines']} lines) vs real edits (median "
             f"{hs['test_sensitive']['median_lines']})", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "hole_sizes.png", dpi=200); plt.close(fig)

# 5. training loss, both facet arms
fig, ax = plt.subplots(figsize=(8.5, 4)); frame(ax)
for lab, pat in (("random 14B", "*_facet-qwen14b-random_train_s42"), ("AST 14B", "*_facet-qwen14b-ast_train_s42")):
    d = next((ROOT / "runs/training").glob(pat))
    L = [json.loads(l) for l in open(d / "metrics.jsonl")]
    tr = [(x["update"], x["loss"]) for x in L if "loss" in x]
    ev = [(x["update"], x["eval_loss"]) for x in L if "eval_loss" in x]
    k = 25; sm = [(tr[i][0], sum(v for _, v in tr[max(0, i - k + 1):i + 1]) / len(tr[max(0, i - k + 1):i + 1])) for i in range(len(tr))]
    ax.plot(*zip(*sm), color=C[lab], lw=2, label=f"{lab}: training loss (25-update mean)")
    ax.plot(*zip(*ev), "o", ms=8, color=C[lab], mec=SURF, mew=2, label=f"{lab}: validation loss")
ax.set_xlabel("optimizer update (batch 16)"); ax.set_ylabel("loss (nats per answer token)")
ax.set_title("facet LoRA training, Qwen2.5-Coder-14B on one DGX Spark (~25 h per arm)", loc="left", fontsize=12)
ax.set_ylim(0.1, 0.52)
ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2, loc="upper right")
fig.text(0.01, 0.01, "Each arm is trained and validated on its own hole type, so the two curves are not directly comparable.", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "training_loss.png", dpi=200); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
