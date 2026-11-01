"""Small tests: PSM label mask, tokenization boundaries, AST eligibility, UTF-8 offsets, windows, pairing."""

import math
import random

import pytest

from fimexp import sources
from fimexp.holes import SourceText, Windower, eligible_nodes, line_token_counts, parser, sample_file
from fimexp.psm import IGNORE, PSM, char_perplexity

CFG = sources.load_config()
EXEMPT = set(CFG["data"]["keyword_text_exempt_types"])


@pytest.fixture(scope="module")
def psm():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["model"]["name"], revision=CFG["model"]["revision"])
    return PSM(tok, CFG["format"])


def test_label_mask_middle_plus_eos_only(psm):
    prefix, middle, suffix = "fn f() -> u32 {\n    ", "a + 1", "\n}\n"
    b = psm.build(prefix, suffix, middle, add_eos=True)
    i = psm.ids
    assert b["input_ids"][:2] == [i["bos"], i["fim_begin"]]
    assert b["input_ids"][b["n_prompt"] - 1] == i["fim_end"]
    sup = [t for t, l in zip(b["input_ids"], b["labels"]) if l != IGNORE]
    assert sup == psm.encode(middle) + [i["eos"]]
    assert all(l == IGNORE for l in b["labels"][: b["n_prompt"]])
    assert b["labels"][-1] == i["eos"] and b["input_ids"][-1] == i["eos"]
    assert psm.tok.decode(sup[:-1], skip_special_tokens=False) == middle


def test_scoring_build_has_no_eos(psm):
    b = psm.build("a", "b", "c", add_eos=False)
    assert psm.ids["eos"] not in b["input_ids"]


def test_separate_segment_tokenization_equals_joint(psm):
    # Sentinels are added tokens, so segment-wise encoding must equal encoding the joined string.
    f = CFG["format"]
    for prefix, suffix, middle in [("let x = ", ";\n", "foo(1)"), ("  ", "  }", " \n  bar"), ("ø ÷", "é", "ü")]:
        joint = psm.tok(f["bos"] + f["fim_begin"] + prefix + f["fim_hole"] + suffix + f["fim_end"] + middle,
                        add_special_tokens=False)["input_ids"]
        assert joint == psm.build(prefix, suffix, middle, add_eos=False)["input_ids"]


def test_char_perplexity():
    assert char_perplexity(0.0, 5) == 1.0
    assert math.isclose(char_perplexity(-10.0, 5), math.exp(2.0))


def test_utf8_offsets_and_reconstruction():
    text = "// ø comment\nfn café(x: u8) -> u8 {\n    x // ü\n}\n"
    src = SourceText(text)
    tree = parser().parse(src.data)
    for n in eligible_nodes(tree, src, EXEMPT):
        s, e = src.char_of_byte(n["start_byte"]), src.char_of_byte(n["end_byte"])
        assert text[s:e] == src.data[n["start_byte"]:n["end_byte"]].decode()
        assert text[:s] + text[s:e] + text[e:] == text
        assert src.byte_of_char(s) == n["start_byte"]
    with pytest.raises(ValueError):
        src.char_of_byte(4)  # inside the two-byte 'ø'


def test_eligibility_rules():
    text = "/// doc\npub fn f(&mut self) -> bool { return true; }\n"
    src = SourceText(text)
    nodes = eligible_nodes(parser().parse(src.data), src, EXEMPT)
    types = {n["type"] for n in nodes}
    texts = {text[n["start_byte"]:n["end_byte"]] for n in nodes}
    assert "source_file" not in types                       # root excluded
    assert "line_comment" in types                          # standalone comment eligible
    assert "doc_comment" not in types and "outer_doc_comment_marker" not in types  # comment internals excluded
    for kw in ("pub", "self", "mut", "true"):
        assert kw not in texts                              # keyword leaves excluded
    assert "()" not in texts and "{" not in texts           # punctuation excluded
    assert "return true" in texts and "&mut self" in texts  # composite nodes stay eligible
    assert "function_item" in types and "block" in types


def test_pairs_reproducible_and_matched():
    text = "fn a() {}\n" * 50 + "/* block */\nstruct S { x: u8 }\n"
    data = text.encode()
    tree = parser().parse(data)
    p1 = sample_file("x.rs", text, tree, 20, 7, EXEMPT)
    p2 = sample_file("x.rs", text, tree, 20, 7, EXEMPT)
    assert p1 == p2
    for p in p1:
        (s, e), (rs, re_) = p["ast"], p["random"]
        assert e - s == re_ - rs and 0 <= rs and re_ <= len(text)
    # size-proportional: large nodes should dominate over 2000 draws
    many = sample_file("x.rs", text, tree, 2000, 7, EXEMPT)
    big = sum(p["node_type"] == "source_file" for p in many)
    assert big == 0


def test_window_fits_and_keeps_answer(psm):
    text = "".join(f"fn f{i}() -> u32 {{ {i} }}\n" for i in range(800))
    src = SourceText(text)
    lt = line_token_counts(src, psm)
    w = Windower(psm, 256)
    s = text.index("fn f400"); e = text.index("\n", s)
    mt = len(psm.encode(text[s:e]))
    ws, we, total = w.window(src, lt, s, e, mt)
    assert ws <= s and e <= we and total <= 256
    b = psm.build(text[ws:s], text[e:we], text[s:e], add_eos=True)
    assert len(b["input_ids"]) == total
    assert ws in src.line_starts and (we == len(text) or text[we - 1] == "\n")
    # an answer that cannot fit is rejected, never truncated
    assert w.window(src, lt, 0, len(text), len(psm.encode(text))) is None
