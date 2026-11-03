"""Structural metrics for FIM completions: parse-clean after splicing, bracket balance, exact match, edit similarity."""

import difflib

from .holes import parser

BRACKETS = {"(": ("paren", 1), ")": ("paren", -1), "[": ("square", 1), "]": ("square", -1), "{": ("brace", 1), "}": ("brace", -1)}


def bracket_delta(code: str) -> tuple:
    """Net open-minus-close count per bracket type, skipping string/char literals and comments (simple Rust lexer).
    Fragments may start or end inside a literal; the lexer simply treats them from the fragment start."""
    d = {"paren": 0, "square": 0, "brace": 0}
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if code.startswith("//", i):
            j = code.find("\n", i); i = n if j == -1 else j
        elif code.startswith("/*", i):
            j = code.find("*/", i + 2); i = n if j == -1 else j + 2
        elif c == "r" and i + 1 < n and code[i + 1] in '#"' and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] == "_")):
            j = i + 1; hashes = 0
            while j < n and code[j] == "#":
                hashes += 1; j += 1
            if j < n and code[j] == '"':
                end = code.find('"' + "#" * hashes, j + 1); i = n if end == -1 else end + 1 + hashes
            else:
                i += 1
        elif c == '"':
            j = i + 1
            while j < n and code[j] != '"':
                j += 2 if code[j] == "\\" else 1
            i = j + 1
        elif c == "'":
            # char literal ('x', '\n', '\u{..}') vs lifetime ('a)
            if i + 2 < n and code[i + 1] == "\\":
                j = code.find("'", i + 2); i = n if j == -1 else j + 1
            elif i + 2 < n and code[i + 2] == "'":
                i += 3
            else:
                i += 1
        else:
            if c in BRACKETS:
                k, v = BRACKETS[c]; d[k] += v
            i += 1
    return (d["paren"], d["square"], d["brace"])


def error_count(text: str) -> int:
    """Number of ERROR or MISSING nodes in a tree-sitter-rust parse."""
    tree = parser().parse(text.encode("utf-8"))
    n, stack = 0, [tree.root_node]
    while stack:
        x = stack.pop()
        if x.type == "ERROR" or x.is_missing:
            n += 1
        stack.extend(x.children)
    return n


def edit_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if (a or b) else 1.0


def score_completion(file_text: str, hole_chars, expected: str, got: str, original_errors: int) -> dict:
    hs, he = hole_chars
    spliced = file_text[:hs] + got + file_text[he:]
    errs = error_count(spliced)
    return {"exact_match": got == expected, "edit_similarity": round(edit_similarity(got, expected), 4),
            "parse_clean": errs == 0, "parse_no_worse": errs <= original_errors,
            "bracket_balance_match": bracket_delta(got) == bracket_delta(expected)}
