#!/usr/bin/env python3
"""Self-contained HTML: side-by-side fill-in-the-middle cards (expected / baseline / post-finetune) for the
repo-specific test set, in the style of prvnsmpth/finetune-code-assistant. Shows every build+test (sensitive)
example where the two models' pass@1 outcomes differ, both directions, plus the headline numbers.
The output contains telekom/controlplane source code (Apache-2.0); it is a local artefact, not committed.
Usage: python make_compare_html.py <full_dir> <examples.jsonl> <summary.json> <stop_check.json> <out.html>
"""
import html, json, sys
from pathlib import Path

BASE, FT = "qwen14b-base", "qwen14b-ft"
CTX_BEFORE, CTX_AFTER, MAX_LINES = 6, 4, 24


def lines_block(text, start, cls, limit=MAX_LINES):
    rows = text.rstrip("\n").split("\n") if text.strip() else ["(empty)"]
    extra = len(rows) - limit
    rows = rows[:limit]
    out = [f'<tr class="{cls}"><td class="ln">{start + k}</td><td class="code">{html.escape(r) or "&nbsp;"}</td></tr>'
           for k, r in enumerate(rows)]
    if extra > 0:
        out.append(f'<tr class="{cls} more"><td class="ln"></td><td class="code">… {extra} more lines</td></tr>')
    return "".join(out)


def status(s):
    bt = "pass" if s.get("build_test_pass") else ("tests fail" if s.get("build") else "build fails")
    return f'<span class="st {"ok" if s.get("build_test_pass") else "bad"}">build+tests: {bt}</span>' + \
           (' <span class="st ok">exact match</span>' if s.get("exact_match") else "")


def card(e, cb, cf, sb, sf):
    start = e["hole_start_line"]
    pre = e["prefix"].rstrip("\n").split("\n")[-CTX_BEFORE:] if e["prefix"] else []
    post = e["suffix"].split("\n")[:CTX_AFTER]
    n_exp = len(e["middle"].rstrip("\n").split("\n"))
    rows = "".join(f'<tr class="ctx"><td class="ln">{start - len(pre) + k}</td><td class="code">{html.escape(r) or "&nbsp;"}</td></tr>'
                   for k, r in enumerate(pre))
    rows += lines_block(e["middle"], start, "exp") + lines_block(cb["completion"], start, "base") + lines_block(cf["completion"], start, "ft")
    rows += "".join(f'<tr class="ctx"><td class="ln">{start + n_exp + k}</td><td class="code">{html.escape(r) or "&nbsp;"}</td></tr>'
                    for k, r in enumerate(post))
    kind = "ft" if sf["build_test_pass"] else "base"
    ftype = "test" if e["is_test_file"] else "source"
    hit = []
    if not cb["stopped"]:
        hit.append("baseline hit the 256-token limit")
    if not cf["stopped"]:
        hit.append("post-finetune hit the 256-token limit")
    return f'''<div class="card" data-kind="{kind}" data-ftype="{ftype}">
 <div class="hd"><span class="t">Code Comparison</span><span class="sub">Fill-in-middle comparison</span></div>
 <div class="meta"><b>{html.escape(e["path"])}</b> <span class="pill">{ftype} file</span>
  <div class="small">commit {e["commit"][:10]} · {html.escape(e["commit_date"][:10])} · {html.escape(e["subject"])} · hole: {e["hole_lines"]} line(s){" · " + "; ".join(hit) if hit else ""}</div></div>
 <div class="tw"><table>{rows}</table></div>
 <div class="foot"><span class="b exp">Expected</span>
  <span class="b base">Baseline</span> {status(sb)}
  <span class="b ftb">Post-Finetune</span> {status(sf)}</div>
</div>'''


def main():
    d, exf, sumf, stopf, outf = sys.argv[1:6]
    d = Path(d)
    ex = {e["id"]: e for e in map(json.loads, open(exf))}
    C = {n: {c["id"]: c for c in map(json.loads, open(d / f"{n}.completions.jsonl"))} for n in (BASE, FT)}
    S = {n: {s["id"]: s for s in map(json.loads, open(d / f"{n}.scores.jsonl"))} for n in (BASE, FT)}
    summ = json.load(open(sumf)); stop = json.load(open(stopf))
    diff = [i for i in sorted(S[FT]) if S[FT][i].get("sensitive") and S[FT][i]["build_test_pass"] != S[BASE][i]["build_test_pass"]]
    n_ft = sum(S[FT][i]["build_test_pass"] for i in diff); n_base = len(diff) - n_ft
    cards = "\n".join(card(ex[i], C[BASE][i], C[FT][i], S[BASE][i], S[FT][i]) for i in diff)
    m = summ["metrics"]
    def row(label, key, fmt="{:.1%}"):
        r = m[key]["rate"] if "rate" in m[key] else m[key]["char_ppl"]
        x = m[key]["finetuned_minus"][BASE]
        v = fmt.format
        return f'<tr><td>{label}</td><td>{m[key]["n"]}</td><td>{v(r[BASE])}</td><td><b>{v(r[FT])}</b></td><td>{x["value"]:+.4f}</td><td>[{x["ci"][0]:+.4f}, {x["ci"][1]:+.4f}]</td></tr>'
    s = stop["build_test_pass/all/neither_hit_limit"]
    table = "".join([row("pass@1 (build+tests), all", "build_test_pass/all"), row("pass@1, source files", "build_test_pass/source"),
                     row("pass@1, test files", "build_test_pass/test"), row("exact match, all", "exact_match/all"),
                     row("char perplexity (lower = better)", "char_ppl/all", "{:.4f}")])
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Fine-tuned vs base: telekom/controlplane</title>
<style>
body{{background:#2b2f36;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:20px;color:#1f2328;font-size:13px}}
.wrap{{max-width:980px;margin:0 auto}} h1{{color:#fff;font-size:18px;margin:0 0 6px}} .lead{{color:#c9d1d9;font-size:14px;margin:0 0 18px;line-height:1.5}}
.panel{{background:#fff;border-radius:10px;padding:16px 18px;margin-bottom:18px}}
.panel table{{border-collapse:collapse;width:100%;font-size:12.5px}} .panel td,.panel th{{padding:6px 8px;border-bottom:1px solid #eaeef2;text-align:left}}
.note{{font-size:13px;color:#57606a;margin-top:10px;line-height:1.5}}
.filters button{{border:1px solid #d0d7de;background:#f6f8fa;border-radius:16px;padding:5px 12px;margin:0 6px 6px 0;cursor:pointer;font-size:13px}}
.filters button.on{{background:#0969da;color:#fff;border-color:#0969da}}
.card{{background:#fff;border-radius:8px;margin-bottom:16px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.3)}}
.hd{{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;border-bottom:1px solid #d8dee4;background:#f6f8fa}}
.hd .t{{font-size:15px;font-weight:600}} .hd .sub{{color:#57606a;font-size:12px}}
.meta{{padding:8px 16px;border-bottom:1px solid #d8dee4;font-size:13px}} .small{{color:#57606a;font-size:11.5px;margin-top:3px}}
.pill{{background:#eaeef2;border-radius:10px;padding:1px 8px;font-size:12px;margin-left:6px}}
.tw{{overflow-x:auto}} .tw table{{border-collapse:collapse;width:max-content;min-width:100%}}
td.ln{{width:1%;min-width:36px;white-space:nowrap;text-align:right;padding:1px 10px 1px 6px;color:#6e7781;font:11.5px ui-monospace,SFMono-Regular,Menlo,monospace;user-select:none;vertical-align:top}}
td.code{{white-space:pre;tab-size:4;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;padding:1px 10px}}
tr.ctx{{background:#fff}} tr.exp{{background:#e6ffec}} tr.base{{background:#ffebe9}} tr.ft{{background:#ddf4ff}}
tr.more td.code{{color:#57606a;font-style:italic}}
.foot{{padding:9px 16px;border-top:1px solid #d8dee4;display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
.b{{color:#fff;border-radius:12px;padding:3px 10px;font-size:12px}} .b.exp{{background:#2da44e}} .b.base{{background:#cf222e}} .b.ftb{{background:#0969da}}
.st{{font-size:11.5px;border-radius:9px;padding:2px 7px;margin-right:8px}} .st.ok{{background:#dafbe1;color:#116329}} .st.bad{{background:#ffebe9;color:#a40e26}}
</style></head><body><div class="wrap">
<h1>Repo-specific fine-tuning: telekom/controlplane (Go)</h1>
<p class="lead">Qwen2.5-Coder-14B base vs the same model with a LoRA trained only on the repo as of 2026-05-01. Test holes are real lines added in commits
after that date; each model sees the file's prefix and suffix and writes the missing lines (greedy, one try). Expected = what the developer wrote.</p>
<div class="panel"><table><tr><th>metric</th><th>n</th><th>base 14B</th><th>fine-tuned 14B</th><th>difference</th><th>95% CI (commits)</th></tr>{table}</table>
<div class="note">Most of the pass@1 gain comes from stopping: the fine-tuned model hit the 256-token limit on 67 holes vs 162 for the base.
On the {s["n"]} examples where neither hit the limit, pass@1 is {s["ft"]:.1%} vs {s["base"]:.1%} ({s["diff"]:+.1%}, CI [{s["ci"][0]:+.1%}, {s["ci"][1]:+.1%}], not significant).
Below: every build+test example where the two models disagree ({len(diff)} examples: post-finetune passes and baseline fails on {n_ft}; baseline passes and post-finetune fails on {n_base}).</div></div>
<div class="filters"><button class="on" data-f="all">All ({len(diff)})</button><button data-f="ft">Post-finetune wins ({n_ft})</button><button data-f="base">Baseline wins ({n_base})</button>
<button data-g="all" class="on">source + test</button><button data-g="source">source files</button><button data-g="test">test files</button></div>
{cards}
</div><script>
let f="all",g="all";
document.querySelectorAll(".filters button").forEach(b=>b.onclick=()=>{{
 if(b.dataset.f){{f=b.dataset.f;document.querySelectorAll("[data-f]").forEach(x=>x.classList.toggle("on",x===b));}}
 else{{g=b.dataset.g;document.querySelectorAll("[data-g]").forEach(x=>x.classList.toggle("on",x===b));}}
 document.querySelectorAll(".card").forEach(c=>c.style.display=((f==="all"||c.dataset.kind===f)&&(g==="all"||c.dataset.ftype===g))?"":"none");}});
</script></body></html>'''
    Path(outf).write_text(page)
    print(outf, len(diff), "cards;", n_ft, "post-finetune wins,", n_base, "baseline wins")


if __name__ == "__main__":
    main()
