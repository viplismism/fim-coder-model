#!/usr/bin/env python3
"""Self-contained HTML report: expected answer vs base / AST / random completions, side by side.
Usage: python -m fimexp.report_html results/generations results/completions_report.html
"""

import difflib, html, json, statistics, sys
from pathlib import Path

MODELS = ("base", "ast", "random")
LABELS = {"base": "Base (untuned)", "ast": "AST-trained", "random": "Random-trained"}
SETS = {"reth_ast": "reth test — AST holes", "reth_random": "reth test — random holes (same spots)",
        "bench_add": "Real-FIM-Eval Rust Add (other repos)"}
E = html.escape


def diff_html(expected, got):
    """Completion with characters matching the expected answer in green, others in red; gaps marked."""
    sm = difflib.SequenceMatcher(None, expected, got, autojunk=False)
    parts = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            parts.append(f'<span class="ok">{E(got[b0:b1])}</span>')
        else:
            if b1 > b0:
                parts.append(f'<span class="bad">{E(got[b0:b1])}</span>')
            if a1 > a0 and op in ("delete", "replace"):
                parts.append(f'<span class="miss" title="missing: {E(expected[a0:a1])}">⌫</span>')
    return "".join(parts) or '<span class="empty">(empty)</span>'


def tail(text, n):
    lines = text.split("\n")
    return "\n".join(lines[-n:])


def head(text, n):
    return "\n".join(text.split("\n")[:n])


def main(gen_dir, out_path):
    d = Path(gen_dir)
    items = [json.loads(l) for l in open(d / "items.jsonl")]
    meta = json.loads((d / "meta.json").read_text())
    gens = {m: {r["id"]: r for r in map(json.loads, open(d / f"{m}.jsonl"))} for m in MODELS}

    rows = []
    for s in SETS:
        its = [i for i in items if i["set"] == s]
        cells = []
        for m in MODELS:
            g = [gens[m][i["id"]] for i in its]
            cells.append(f"<td>{100 * sum(x['exact_match'] for x in g) / len(g):.0f}%</td>"
                         f"<td>{statistics.fmean(x['edit_similarity'] for x in g):.3f}</td>")
        rows.append(f"<tr><th>{E(SETS[s])} <small>({len(its)})</small></th>{''.join(cells)}</tr>")

    cards = []
    for k, it in enumerate(items):
        outs = {m: gens[m][it["id"]] for m in MODELS}
        differ = len({o["text"] for o in outs.values()}) > 1
        cols = []
        for m in MODELS:
            o = outs[m]
            badge = '<span class="b exact">exact</span>' if o["exact_match"] else f'<span class="b">sim {o["edit_similarity"]:.2f}</span>'
            stop = "" if o["stopped"] else '<span class="b warn" title="hit the length limit before the end-of-answer token">no stop</span>'
            cols.append(f'<div class="col"><div class="ch">{LABELS[m]} {badge}{stop}</div><pre class="code">{diff_html(it["middle"], o["text"])}</pre></div>')
        where = E(it["path"] or "") if it["path"] else ""
        node = f' · node <code>{E(it["node_type"])}</code>' if it.get("node_type") else ""
        cards.append(f'''
<div class="card" data-set="{it["set"]}" data-differ="{int(differ)}" data-anyexact="{int(any(o["exact_match"] for o in outs.values()))}">
  <div class="meta"><b>#{k + 1}</b> · {E(SETS[it["set"]])} · <code>{E(it["group"])}</code> {where}{node}</div>
  <pre class="ctx">{E(tail(it["prefix"], 12))}<span class="hole">▌hole▐</span>{E(head(it["suffix"], 6))}</pre>
  <div class="col exp"><div class="ch">Expected (what the developer wrote)</div><pre class="code">{E(it["middle"]) or "(empty)"}</pre></div>
  <div class="cols">{"".join(cols)}</div>
</div>''')

    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>FIM completions: base vs AST vs random</title>
<style>
body{{font:14px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;background:#f6f7f9;color:#1d2330}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#5b6475;margin-bottom:16px}}
table{{border-collapse:collapse;background:#fff;margin:8px 0 16px}} th,td{{border:1px solid #dde1e8;padding:6px 10px;text-align:center}}
th:first-child{{text-align:left}} thead th{{background:#eef1f5}}
.controls{{position:sticky;top:0;background:#f6f7f9;padding:8px 0;z-index:2;border-bottom:1px solid #dde1e8}}
.card{{background:#fff;border:1px solid #dde1e8;border-radius:8px;padding:12px;margin:14px 0}}
.meta{{color:#5b6475;margin-bottom:6px}} code{{background:#eef1f5;padding:0 4px;border-radius:3px}}
pre{{margin:0;white-space:pre-wrap;word-break:break-word;font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace}}
.ctx{{background:#fafbfc;color:#7a8396;border:1px solid #eceff3;padding:8px;border-radius:6px;max-height:260px;overflow:auto}}
.hole{{background:#ffe9a8;color:#7a5a00;font-weight:600;padding:0 3px;border-radius:3px}}
.cols{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}}
.col{{border:1px solid #e3e7ee;border-radius:6px;overflow:hidden}} .col .code{{padding:8px}}
.exp{{margin-top:10px;border-color:#a8d5b5}} .exp .ch{{background:#e6f4ea}}
.ch{{background:#f1f3f7;padding:5px 8px;font-weight:600;font-size:13px}}
.ok{{background:#e6f4ea}} .bad{{background:#fde2e1;color:#8a1c14}} .miss{{color:#b26a00;font-weight:700}}
.empty{{color:#9aa3b2}} .b{{font-weight:500;font-size:11px;margin-left:6px;padding:1px 6px;border-radius:9px;background:#e3e7ee}}
.b.exact{{background:#1e8e3e;color:#fff}} .b.warn{{background:#fff1c2;color:#7a5a00}}
.legend span{{padding:0 6px;margin-right:8px;border-radius:3px}}
</style></head><body>
<h1>FIM completions — base vs AST-trained vs random-trained</h1>
<div class="sub">DeepSeek-Coder-6.7B base ({E(meta["base_precision"])}), greedy decoding, same prompts for all models. {len(items)} examples:
held-out reth <b>test</b> packages and Real-FIM-Eval Rust Add. Qualitative sample; the experiment's results are the perplexity scores in
<code>results/final_eval_summary.json</code>. Generated {E(meta["created"])}.</div>
<table><thead><tr><th>Set</th>{"".join(f'<th colspan="2">{LABELS[m]}</th>' for m in MODELS)}</tr>
<tr><th></th>{"<th>exact</th><th>edit sim</th>" * 3}</tr></thead><tbody>{"".join(rows)}</tbody></table>
<div class="legend"><span class="ok">matches expected</span><span class="bad">differs from expected</span><span class="miss">⌫</span>part of the expected answer missing (hover)
<span class="b warn">no stop</span> hit the length limit</div>
<div class="controls">Show: <select id="set"><option value="">all sets</option>{"".join(f'<option value="{s}">{E(v)}</option>' for s, v in SETS.items())}</select>
<label><input type="checkbox" id="differ"> only where models differ</label>
<label><input type="checkbox" id="anyexact"> only where some model is exact</label> <span id="count"></span></div>
{"".join(cards)}
<script>
function apply(){{const s=document.getElementById('set').value, d=document.getElementById('differ').checked, a=document.getElementById('anyexact').checked;let n=0;
document.querySelectorAll('.card').forEach(c=>{{const show=(!s||c.dataset.set===s)&&(!d||c.dataset.differ==='1')&&(!a||c.dataset.anyexact==='1');c.style.display=show?'':'none';n+=show}});
document.getElementById('count').textContent=n+' shown'}}
['set','differ','anyexact'].forEach(i=>document.getElementById(i).addEventListener('change',apply));apply();
</script></body></html>'''
    Path(out_path).write_text(page)
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
