"""Per-language tests: parsing, eligibility rules, UTF-8 offsets, reconstruction, aligned spans, pairing, tokenizer."""

import random

import pytest

from fimexp import sources
from fimexp.holes import SourceText, aligned_span, eligible_nodes, parser, sample_file
from fimexp.langs import LANGS, get

CFG = sources.load_config()

SNIPPETS = {
    "rust": (
        "/// Adds one — é\nfn add(x: u32) -> u32 {\n    // inner\n    let y = x + 1;\n    if y > 2 { return y; }\n    y\n}\n",
        ["self", "return", "pub", "true", "mut"],
    ),
    "python": (
        "# Module — é\nimport os\n\n\nclass Box:\n    def __init__(self, x):\n        # store\n        self.x = x\n\n"
        "    def grow(self, n=1):\n        if n is None:\n            return None\n        self.x += n\n        return [self.x, True]\n",
        ["return", "None", "True", "is", "def"],
    ),
    "typescript": (
        "// Totals — é\nexport function total(items: number[]): number {\n  /* sum */\n  let s = 0;\n"
        "  for (const it of items) {\n    if (it === null) { continue; }\n    s += it;\n  }\n  return s;\n}\n",
        ["return", "const", "let", "null", "continue"],
    ),
    "go": (
        "// Package m — é\npackage m\n\nimport \"fmt\"\n\n// Total sums.\nfunc Total(xs []int) int {\n\ts := 0\n"
        "\tfor _, x := range xs {\n\t\tif x == 0 {\n\t\t\tcontinue\n\t\t}\n\t\ts += x\n\t}\n\tfmt.Println(s, nil, true)\n\treturn s\n}\n",
        ["return", "continue", "nil", "true", "func"],
    ),
    "cpp": (
        "// Totals — é\n#include <vector>\n\nnamespace m {\nint total(const std::vector<int>& xs) {\n  /* sum */\n"
        "  int s = 0;\n  for (int x : xs) {\n    if (x == 0) { continue; }\n    s += x;\n  }\n  return s;\n}\n}  // namespace m\n",
        ["return", "continue", "const", "namespace", "true"],
    ),
}


@pytest.fixture(scope="module")
def psm():
    from transformers import AutoTokenizer
    from fimexp.psm import PSM
    tok = AutoTokenizer.from_pretrained(CFG["model"]["name"], revision=CFG["model"]["revision"])
    return PSM(tok, CFG["format"])


def test_every_language_has_a_benchmark_subset():
    assert sorted(LANGS) == ["cpp", "go", "python", "rust", "typescript"]
    assert all(c["real_fim_eval"] for c in LANGS.values())
    assert ".c" not in LANGS["cpp"]["benchmark_extensions"]


@pytest.mark.parametrize("lang", sorted(SNIPPETS))
def test_snippet_parses_cleanly(lang):
    text, _ = SNIPPETS[lang]
    assert not parser(lang).parse(text.encode()).root_node.has_error


@pytest.mark.parametrize("lang", sorted(SNIPPETS))
def test_eligibility_rules(lang):
    text, kws = SNIPPETS[lang]
    cfg = get(lang)
    src = SourceText(text)
    tree = parser(lang).parse(src.data)
    nodes = eligible_nodes(tree, src, cfg["exempt_types"], lang)
    texts = {src.data[n["start_byte"]:n["end_byte"]].decode() for n in nodes}
    types = {n["type"] for n in nodes}
    assert tree.root_node.type not in types and text not in texts              # root excluded
    assert any(n["is_comment"] for n in nodes)                                  # standalone comments eligible
    for n in nodes:                                                             # nothing inside a comment except the comment
        if not n["is_comment"]:
            inside = [c for c in nodes if c["is_comment"] and c["start_byte"] <= n["start_byte"] and n["end_byte"] <= c["end_byte"]]
            assert not inside
    for kw in kws:
        assert kw not in texts, f"{lang}: keyword leaf {kw!r} should be excluded"
    assert not any(t.strip() in ("(", ")", "{", "}", ";", "()", "{}") for t in texts)   # punctuation excluded
    assert any("\n" in t and len(t) > 20 for t in texts)                        # composite blocks stay eligible


@pytest.mark.parametrize("lang", sorted(SNIPPETS))
def test_utf8_offsets_and_reconstruction(lang):
    text, _ = SNIPPETS[lang]
    src = SourceText(text)
    assert not src.ascii                                                        # snippets contain non-ASCII text
    for n in eligible_nodes(parser(lang).parse(src.data), src, get(lang)["exempt_types"], lang):
        s, e = src.char_of_byte(n["start_byte"]), src.char_of_byte(n["end_byte"])
        assert text[s:e] == src.data[n["start_byte"]:n["end_byte"]].decode()
        assert text[:s] + text[s:e] + text[e:] == text and src.byte_of_char(s) == n["start_byte"]


@pytest.mark.parametrize("lang", sorted(SNIPPETS))
def test_aligned_span_boundaries(lang):
    text, _ = SNIPPETS[lang]
    src = SourceText(text)
    tree = parser(lang).parse(src.data)
    starts, ends, stack = set(), set(), [tree.root_node]
    while stack:
        n = stack.pop(); starts.add(n.start_byte); ends.add(n.end_byte); stack.extend(n.children)
    for seed in range(40):
        for kw in ({}, {"target_chars": 30}):
            a = aligned_span(tree, src, random.Random(seed), get(lang)["exempt_types"], lang=lang, **kw)
            b = aligned_span(tree, src, random.Random(seed), get(lang)["exempt_types"], lang=lang, **kw)
            assert a == b and a is not None
            s, e, _ = a
            assert src.byte_of_char(s) in starts and src.byte_of_char(e) in ends and text[s:e].strip()


@pytest.mark.parametrize("lang", sorted(SNIPPETS))
def test_pairs_matched(lang):
    text, _ = SNIPPETS[lang]
    tree = parser(lang).parse(text.encode())
    pairs = sample_file("x", text, tree, 30, 7, get(lang)["exempt_types"], lang)
    assert pairs and pairs == sample_file("x", text, tree, 30, 7, get(lang)["exempt_types"], lang)
    for p in pairs:
        (s, e), (rs, re_) = p["ast"], p["random"]
        assert e - s == re_ - rs and 0 <= rs and re_ <= len(text)


@pytest.mark.parametrize("lang", sorted(SNIPPETS))
def test_tokenizer_round_trip_and_segment_equivalence(lang, psm):
    text, _ = SNIPPETS[lang]
    ids = psm.encode(text)
    assert psm.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text
    k = len(text) // 3
    f = CFG["format"]
    joint = psm.tok(f["bos"] + f["fim_begin"] + text[:k] + f["fim_hole"] + text[2 * k:] + f["fim_end"] + text[k:2 * k],
                    add_special_tokens=False)["input_ids"]
    assert joint == psm.build(text[:k], text[2 * k:], text[k:2 * k], add_eos=False)["input_ids"]


def test_known_tokenizer_quirk_is_limited_to_byte_fragment_tokens(psm):
    # DeepSeek's tokenizer.json maps 13 single characters to added tokens that decode to U+FFFD (see env/README.md).
    # Scoring never decodes, so scores are unaffected; this pins the quirk so a change is noticed.
    quirky = set("õ÷ÁýÀÿøúþüùöû")
    for ch in quirky:
        assert psm.tok.decode(psm.encode(ch), skip_special_tokens=False, clean_up_tokenization_spaces=False) == "�"
    for ch in "éñçß—中":
        assert psm.tok.decode(psm.encode(ch), skip_special_tokens=False, clean_up_tokenization_spaces=False) == ch


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_benchmark_rows_are_that_language_only(lang):
    rows = sources.benchmark_rows(CFG, lang)
    assert rows and all(r["lang"] == lang for r in rows)
    assert all(r["path"].endswith(get(lang)["benchmark_extensions"]) for r in rows)
    if lang == "cpp":
        assert not any(r["path"].endswith(".c") for r in rows) and len(rows) == 2950
    if lang == "rust":
        assert len(rows) == 4727 and rows == sources.benchmark_rows(CFG)   # default unchanged


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_benchmark_prompt_is_language_neutral(lang, psm):
    from fimexp.score import bench_items
    f = CFG["format"]
    for split in ("add", "edit"):
        it = next(bench_items(CFG, split, lang))
        assert it["lang"] == lang
        b = psm.build(it["prefix"], it["suffix"], it["middle"], add_eos=False)
        text = psm.tok.decode(b["input_ids"][:b["n_prompt"]], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # exact layout: BOS, FIM begin, prefix, FIM hole, suffix, FIM end — nothing else is added
        assert text == f["bos"] + f["fim_begin"] + it["prefix"] + f["fim_hole"] + it["suffix"] + f["fim_end"] or "�" in text
        for word in ("Rust", "rust", "Python", "TypeScript", "language", "Complete", "Fill"):
            assert word not in (text.replace(it["prefix"], "").replace(it["suffix"], ""))
