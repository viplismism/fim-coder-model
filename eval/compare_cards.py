#!/usr/bin/env python3
"""Self-contained HTML for facet-rs/facet: fill-in-the-middle cards (expected / base / AST fine-tune / random
fine-tune), in the style of prvnsmpth/finetune-code-assistant. Shows every build+test (sensitive) example where the
three models' pass@1 outcomes are not all the same, plus the headline table from summarize_facet.py.
The output contains facet source code (MIT/Apache-2.0); it is a local artefact, not committed.
Usage: python -m eval.compare_cards <scores_dir> <examples.jsonl> <summary.json> <out.html>
"""
import html, json, sys
from pathlib import Path

BASE, AST, RND = "qwen14b-base", "facet-ast", "facet-random"
LABEL = {BASE: "Base", AST: "AST fine-tune", RND: "Random fine-tune"}
CLS = {BASE: "base", AST: "ast", RND: "rnd"}
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


def card(e, C, S):
    start = e["hole_start_line"]
    pre = e["prefix"].rstrip("\n").split("\n")[-CTX_BEFORE:] if e["prefix"] else []
    post = e["suffix"].split("\n")[:CTX_AFTER]
    n_exp = len(e["middle"].rstrip("\n").split("\n"))
    rows = "".join(f'<tr class="ctx"><td class="ln">{start - len(pre) + k}</td><td class="code">{html.escape(r) or "&nbsp;"}</td></tr>'
                   for k, r in enumerate(pre))
    rows += lines_block(e["middle"], start, "exp") + "".join(lines_block(C[n]["completion"], start, CLS[n]) for n in (BASE, AST, RND))
    rows += "".join(f'<tr class="ctx"><td class="ln">{start + n_exp + k}</td><td class="code">{html.escape(r) or "&nbsp;"}</td></tr>'
                    for k, r in enumerate(post))
    p = {n: bool(S[n]["build_test_pass"]) for n in (BASE, AST, RND)}
    kind = ("both" if p[AST] and p[RND] else "ast" if p[AST] else "rnd" if p[RND] else "none") if not p[BASE] else "basewin"
    ftype = "test" if e["is_test_file"] else "source"
    auth = "ai" if e.get("ai_tagged") else ("bot" if e.get("bot_author") else "human")
    hit = [f"{LABEL[n]} hit the 256-token limit" for n in (BASE, AST, RND) if not C[n].get("stopped")]
    foot = '<span class="b exp">Expected</span>' + "".join(f' <span class="b {CLS[n]}b">{LABEL[n]}</span> {status(S[n])}' for n in (BASE, AST, RND))
    return f'''<div class="card" data-kind="{kind}" data-ftype="{ftype}" data-auth="{auth}">
 <div class="hd"><span class="t">Code Comparison</span><span class="sub">Fill-in-middle comparison</span></div>
 <div class="meta"><b>{html.escape(e["path"])}</b> <span class="pill">{ftype} file</span> <span class="pill">{ {"ai": "AI-tagged commit", "bot": "bot commit", "human": "human commit"}[auth]}</span>
  <div class="small">commit {e["commit"][:10]} · {html.escape(e["commit_date"][:10])} · {html.escape(e["subject"])} · hole: {e["hole_lines"]} line(s){" · " + "; ".join(hit) if hit else ""}</div></div>
 <div class="tw"><table>{rows}</table></div>
 <div class="foot">{foot}</div>
</div>'''


def main():
    d, exf, sumf, outf = sys.argv[1:5]
    d = Path(d)
    ex = {e["id"]: e for e in map(json.loads, open(exf))}
    M = (BASE, AST, RND)
    C = {n: {c["id"]: c for c in map(json.loads, open(d / f"{n}.completions.jsonl"))} for n in M}
    S = {n: {s["id"]: s for s in map(json.loads, open(d / f"{n}.scores.jsonl"))} for n in M}
    summ = json.load(open(sumf)); m = summ["metrics"]
    ids = [i for i in sorted(S[BASE]) if all(i in S[n] and not S[n][i].get("skipped") for n in M) and S[BASE][i].get("sensitive")]
    diff = [i for i in ids if len({bool(S[n][i]["build_test_pass"]) for n in M}) > 1]
    cards_l = [card(ex[i], {n: C[n][i] for n in M}, {n: S[n][i] for n in M}) for i in diff]
    cnt = {k: sum(f'data-kind="{k}"' in c for c in cards_l) for k in ("both", "ast", "rnd", "basewin")}
    cards = "\n".join(cards_l)
    def cell(x):
        return f'{x["value"]:+.3f} <span class="ci">[{x["ci"][0]:+.3f}, {x["ci"][1]:+.3f}]</span>' + (" ✓" if x.get("beats") or x.get("better") else "")
    def row(label, key, fmt="{:.1%}"):
        if key not in m or not m[key]["n"]:
            return ""
        r = m[key].get("rate") or m[key].get("char_ppl"); dd = m[key]["diffs"]
        return (f'<tr><td>{label}</td><td>{m[key]["n"]}</td>' + "".join(f"<td>{fmt.format(r[n])}</td>" for n in M) +
                "".join(f"<td>{cell(dd[k])}</td>" for k in (f"{AST} - {BASE}", f"{RND} - {BASE}", f"{AST} - {RND}")) + "</tr>")
    table = "".join(row(lbl, key, *f) for lbl, key, *f in [
        ("pass@1, all", "build_test_pass/all"), ("pass@1, source files", "build_test_pass/source"), ("pass@1, test files", "build_test_pass/test"),
        ("pass@1, human commits", "build_test_pass/human"), ("pass@1, AI-tagged commits", "build_test_pass/ai_tagged"),
        ("pass@1, bot-only commits", "build_test_pass/bot_only"), ("pass@1, no model hit 256-token limit", "build_test_pass/neither_hit_limit"),
        ("exact match, all", "exact_match/all"), ("exact match, source", "exact_match/source"), ("exact match, test", "exact_match/test"),
        ("exact match, human", "exact_match/human"), ("exact match, AI-tagged", "exact_match/ai_tagged"),
        ("char perplexity, all (lower = better)", "char_ppl/all", "{:.4f}")])
    g = summ["generation"]
    stopnote = ", ".join(f'{LABEL[n]} {g[n]["hit_256_token_limit"]}' for n in M)
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Facet fine-tune comparison</title>
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
tr.ctx{{background:#fff}} tr.exp{{background:#e6ffec}} tr.base{{background:#ffebe9}} tr.ast{{background:#ddf4ff}} tr.rnd{{background:#fbefff}}
tr.more td.code{{color:#57606a;font-style:italic}}
.foot{{padding:9px 16px;border-top:1px solid #d8dee4;display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
.b{{color:#fff;border-radius:12px;padding:3px 10px;font-size:12px}} .b.exp{{background:#2da44e}} .b.baseb{{background:#cf222e}} .b.astb{{background:#0969da}} .b.rndb{{background:#8250df}}
.ci{{color:#6e7781;font-size:11px}} .wrap{{max-width:1180px}}
.st{{font-size:11.5px;border-radius:9px;padding:2px 7px;margin-right:8px}} .st.ok{{background:#dafbe1;color:#116329}} .st.bad{{background:#ffebe9;color:#a40e26}}
</style></head><body><div class="wrap">
<h1>Repo-specific fine-tuning: facet-rs/facet (Rust)</h1>
<p class="lead">Qwen2.5-Coder-14B base vs two LoRA fine-tunes with identical settings, trained only on the repo as of 2026-01-01: one on AST-node holes,
one on random holes of identical lengths. Test holes are real lines added in commits after that date; each model sees the file's prefix and suffix and
writes the missing lines (greedy, one try). Expected = what the developer wrote. pass@1 = cargo build + cargo test of the crate pass.</p>
<div class="panel"><table><tr><th>metric</th><th>n</th><th>base</th><th>AST ft</th><th>random ft</th><th>AST − base</th><th>random − base</th><th>AST − random</th></tr>{table}</table>
<div class="note">Differences with 95% CIs from a paired bootstrap over commits; ✓ = significant (CI excludes 0 in the better direction).
Hits of the 256-token limit: {stopnote}. Below: every build+test example where the three models do not all agree ({len(diff)} examples).</div></div>
<div class="filters"><button class="on" data-f="all">All ({len(diff)})</button><button data-f="both">Both fine-tunes win ({cnt["both"]})</button>
<button data-f="ast">Only AST wins ({cnt["ast"]})</button><button data-f="rnd">Only random wins ({cnt["rnd"]})</button><button data-f="basewin">Base passes, a fine-tune fails ({cnt["basewin"]})</button><br>
<button data-g="all" class="on">source + test</button><button data-g="source">source files</button><button data-g="test">test files</button>
<button data-a="all" class="on">all authors</button><button data-a="human">human</button><button data-a="ai">AI-tagged</button><button data-a="bot">bot</button></div>
{cards}
</div><script>
let f="all",g="all",au="all";
document.querySelectorAll(".filters button").forEach(b=>b.onclick=()=>{{
 const k=b.dataset.f?"f":b.dataset.g?"g":"a";
 document.querySelectorAll("[data-"+k+"]").forEach(x=>x.classList.toggle("on",x===b));
 if(k==="f")f=b.dataset.f;else if(k==="g")g=b.dataset.g;else au=b.dataset.a;
 document.querySelectorAll(".card").forEach(c=>c.style.display=((f==="all"||c.dataset.kind===f)&&(g==="all"||c.dataset.ftype===g)&&(au==="all"||c.dataset.auth===au))?"":"none");}});
</script></body></html>'''
    Path(outf).write_text(page)
    print(outf, len(diff), "cards", cnt)


if __name__ == "__main__":
    main()
