"""Single-node AST holes and matched random holes, with a shared context-window rule."""

import bisect, math, random, re

import tree_sitter as ts
import tree_sitter_rust as tsr

COMMENT_TYPES = {"line_comment", "block_comment"}
RUST_KEYWORDS = frozenset(
    "as break const continue crate else enum extern false fn for if impl in let loop match mod move mut pub ref "
    "return self Self static struct super trait true type unsafe use where while async await dyn abstract become "
    "box do final macro override priv typeof unsized virtual yield try gen union".split()
)
PUNCT_ONLY = re.compile(r"[^\w\s]+")

_LANG = ts.Language(tsr.language())


def parser() -> ts.Parser:
    return ts.Parser(_LANG)


class SourceText:
    """A file's exact text with byte<->char offset conversion (UTF-8)."""

    def __init__(self, text: str):
        self.text = text
        self.data = text.encode("utf-8")
        self.ascii = len(self.data) == len(text)
        if not self.ascii:
            # byte offset of each char boundary; index = char offset
            self._byte_at = [0]
            for ch in text:
                self._byte_at.append(self._byte_at[-1] + len(ch.encode("utf-8")))
        nl = [i for i, c in enumerate(text) if c == "\n"]
        self.line_starts = [0] + [i + 1 for i in nl if i + 1 < len(text)]

    def char_of_byte(self, b: int) -> int:
        if self.ascii:
            return b
        i = bisect.bisect_left(self._byte_at, b)
        if self._byte_at[i] != b:
            raise ValueError(f"byte offset {b} is not on a character boundary")
        return i

    def byte_of_char(self, c: int) -> int:
        return c if self.ascii else self._byte_at[c]


def eligible_nodes(tree, src: SourceText, exempt_types) -> list[dict]:
    """Named nodes eligible as a hole (see configs/experiment.yaml, data: node eligibility)."""
    out, stack = [], [(tree.root_node, False)]
    while stack:
        n, in_comment = stack.pop()
        is_root = n.parent is None
        if not is_root and n.is_named and not in_comment and not n.is_missing and n.type != "ERROR" and n.end_byte > n.start_byte:
            ok = True
            if not any(c.is_named for c in n.children) and n.type not in exempt_types:
                txt = src.data[n.start_byte:n.end_byte].decode("utf-8").strip()
                ok = txt not in RUST_KEYWORDS and not PUNCT_ONLY.fullmatch(txt)
            if ok:
                out.append({"type": n.type, "start_byte": n.start_byte, "end_byte": n.end_byte,
                            "is_comment": n.type in COMMENT_TYPES})
        child_in_comment = in_comment or n.type in COMMENT_TYPES
        stack.extend((c, child_in_comment) for c in reversed(n.children))
    return out


class Windower:
    """Choose prefix/suffix context at line boundaries so the PSM sequence fits `max_tokens`.

    Budget after the answer is split evenly between prefix and suffix; a side that reaches the file
    boundary passes its unused budget to the other. Lengths are verified with the real tokenizer.
    """

    def __init__(self, psm, max_tokens: int):
        self.psm, self.max_tokens = psm, max_tokens

    def fixed_tokens(self, middle: str) -> int:
        return self.psm.overhead + len(self.psm.encode(middle)) + 1  # + EOS

    def window(self, src: SourceText, line_tokens: list[int], start: int, end: int, middle_tokens: int):
        """Return (window_start, window_end, total_tokens) in chars, or None if the answer cannot fit."""
        budget = self.max_tokens - self.psm.overhead - middle_tokens - 1
        ls, n = src.line_starts, len(src.text)
        line_end = lambda i: ls[i + 1] if i + 1 < len(ls) else n
        first = bisect.bisect_right(ls, start) - 1               # line holding the hole's first char
        last = bisect.bisect_right(ls, max(end - 1, start)) - 1  # line holding the hole's last char
        # The partial lines around the hole are always kept; whole lines beyond them are optional.
        base = len(self.psm.encode(src.text[ls[first]:start])) + len(self.psm.encode(src.text[end:line_end(last)]))
        remaining = budget - base
        if remaining < 0:
            return None
        pre_cost = [line_tokens[i] for i in range(first - 1, -1, -1)]
        suf_cost = [line_tokens[i] for i in range(last + 1, len(ls))]
        n_pre = n_suf = used_pre = used_suf = 0
        half = remaining / 2
        while n_pre < len(pre_cost) and used_pre + pre_cost[n_pre] <= half:
            used_pre += pre_cost[n_pre]; n_pre += 1
        while n_suf < len(suf_cost) and used_suf + suf_cost[n_suf] <= half:
            used_suf += suf_cost[n_suf]; n_suf += 1
        # leftover budget flows to whichever side still has lines
        while n_pre < len(pre_cost) and used_pre + used_suf + pre_cost[n_pre] <= remaining:
            used_pre += pre_cost[n_pre]; n_pre += 1
        while n_suf < len(suf_cost) and used_pre + used_suf + suf_cost[n_suf] <= remaining:
            used_suf += suf_cost[n_suf]; n_suf += 1
        # Per-line counts are estimates (BPE can merge across lines); verify exactly and shrink if needed.
        while True:
            ws, we = ls[first - n_pre], line_end(last + n_suf)
            total = (self.psm.overhead + len(self.psm.encode(src.text[ws:start])) + middle_tokens
                     + len(self.psm.encode(src.text[end:we])) + 1)
            if total <= self.max_tokens:
                return ws, we, total
            if n_pre == 0 and n_suf == 0:
                return None
            if n_pre >= n_suf:
                n_pre -= 1
            else:
                n_suf -= 1


def line_token_counts(src: SourceText, psm) -> list[int]:
    ls = src.line_starts + [len(src.text)]
    return [len(psm.encode(src.text[ls[i]:ls[i + 1]])) for i in range(len(ls) - 1)]


def sample_file(path: str, text: str, tree, n_examples: int, seed: int, exempt_types) -> list[dict]:
    """Draw `n_examples` AST holes (prob. proportional to byte size) and a matched random hole for each."""
    src = SourceText(text)
    nodes = eligible_nodes(tree, src, exempt_types)
    if not nodes:
        return []
    rng = random.Random(f"{seed}:{path}:ast")
    picks = rng.choices(nodes, weights=[n["end_byte"] - n["start_byte"] for n in nodes], k=n_examples)
    pairs = []
    for k, node in enumerate(picks):
        s, e = src.char_of_byte(node["start_byte"]), src.char_of_byte(node["end_byte"])
        length = e - s
        r = random.Random(f"{seed}:{path}:{k}:random").randint(0, len(text) - length)
        pairs.append({"k": k, "node_type": node["type"], "is_comment": node["is_comment"],
                      "ast": (s, e), "random": (r, r + length)})
    return pairs


def n_examples_for(n_bytes: int, bytes_per_example: int) -> int:
    return max(1, math.ceil(n_bytes / bytes_per_example))


def aligned_span(tree, src: SourceText, rng: random.Random, exempt_types, max_tries: int = 20):
    """Aligned-span masking (AST-FIM): sample a character span with uniform random endpoints, take the smallest
    eligible named node containing it, and return the contiguous run of that node's direct named children with the
    largest character-level IoU with the span (the node itself if it has no named children).
    Returns (start_char, end_char, node_type) on exact node boundaries, or None."""
    n = len(src.text)
    if n < 2:
        return None
    for _ in range(max_tries):
        a, b = sorted(rng.sample(range(n + 1), 2))
        if src.text[a:b].strip() == "":
            continue
        bs, be = src.byte_of_char(a), src.byte_of_char(b)
        node = tree.root_node.named_descendant_for_byte_range(bs, be)
        while node is not None and node.parent is not None and node.type in COMMENT_TYPES | {"ERROR"}:
            node = node.parent
        if node is None:
            continue
        kids = [c for c in node.named_children if c.end_byte > c.start_byte]
        if not kids:
            if node.parent is None:
                continue
            return src.char_of_byte(node.start_byte), src.char_of_byte(node.end_byte), node.type
        best, best_iou = None, -1.0
        for i in range(len(kids)):
            for j in range(i, len(kids)):
                s, e = kids[i].start_byte, kids[j].end_byte
                inter = max(0, min(e, be) - max(s, bs)); union = max(e, be) - min(s, bs)
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best, best_iou = (s, e, kids[i].type if i == j else f"{node.type}:children[{i}:{j + 1}]"), iou
        s, e, t = best
        txt = src.data[s:e].decode("utf-8").strip()
        if txt in RUST_KEYWORDS or PUNCT_ONLY.fullmatch(txt or "x"):
            continue
        return src.char_of_byte(s), src.char_of_byte(e), t
    return None
