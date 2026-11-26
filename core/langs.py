"""Per-language configuration for hole sampling, parsing and evaluation (Real-FIM-Eval languages only).

Eligibility rule (all languages): named tree-sitter nodes only (anonymous keyword/punctuation tokens excluded), not the
root, not inside a comment (comments are atomic holes), and not a named node without named children whose whole text is
a keyword of that language or punctuation only. Exempt node types (string contents etc.) are never removed by the
keyword-text rule. Rust keeps exactly the configuration used in the earlier Rust experiment.
"""

from functools import lru_cache

import tree_sitter as ts

RUST_KEYWORDS = frozenset(
    "as break const continue crate else enum extern false fn for if impl in let loop match mod move mut pub ref "
    "return self Self static struct super trait true type unsafe use where while async await dyn abstract become "
    "box do final macro override priv typeof unsized virtual yield try gen union".split()
)
PYTHON_KEYWORDS = frozenset(
    "False None True and as assert async await break class continue def del elif else except finally for from global "
    "if import in is lambda nonlocal not or pass raise return try while with yield".split()
)
# Reserved words, strict-mode reserved words and TypeScript modifiers/operators. Built-in type names (number, string,
# boolean, ...) stay eligible, like Rust primitive types (u32) do.
TYPESCRIPT_KEYWORDS = frozenset(
    "break case catch class const continue debugger default delete do else enum export extends false finally for "
    "function if import in instanceof new null return super switch this throw true try typeof var void while with "
    "implements interface let package private protected public static yield as async await declare abstract readonly "
    "keyof infer is satisfies override accessor get set of from module namespace type undefined".split()
)
# Go keywords plus the predeclared constants true/false/nil/iota (keyword-like literals, as true/false are in Rust).
GO_KEYWORDS = frozenset(
    "break case chan const continue default defer else fallthrough for func go goto if import interface map package "
    "range return select struct switch type var true false nil iota".split()
)
CPP_KEYWORDS = frozenset(
    "alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t class compl "
    "concept const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default "
    "delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long "
    "mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected public register "
    "reinterpret_cast requires return short signed sizeof static static_assert static_cast struct switch template this "
    "thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while xor "
    "xor_eq override final".split()
)


def _rust():
    import tree_sitter_rust as m
    return m.language()


def _python():
    import tree_sitter_python as m
    return m.language()


def _typescript():
    import tree_sitter_typescript as m
    return m.language_typescript()


def _go():
    import tree_sitter_go as m
    return m.language()


def _cpp():
    import tree_sitter_cpp as m
    return m.language()


LANGS = {
    "rust": {"grammar": _rust, "grammar_package": "tree-sitter-rust", "extensions": (".rs",),
             "comment_types": frozenset({"line_comment", "block_comment"}), "keywords": RUST_KEYWORDS,
             "exempt_types": frozenset({"string_content", "escape_sequence", "char_literal"}),
             "package_markers": ("Cargo.toml",), "real_fim_eval": "rust", "benchmark_extensions": (".rs",)},
    "python": {"grammar": _python, "grammar_package": "tree-sitter-python", "extensions": (".py",),
               "comment_types": frozenset({"comment"}), "keywords": PYTHON_KEYWORDS,
               "exempt_types": frozenset({"string_content", "escape_sequence"}),
               "package_markers": ("pyproject.toml", "setup.py", "setup.cfg"), "real_fim_eval": "python",
               "benchmark_extensions": (".py",)},
    "typescript": {"grammar": _typescript, "grammar_package": "tree-sitter-typescript", "extensions": (".ts",),
                   "comment_types": frozenset({"comment"}), "keywords": TYPESCRIPT_KEYWORDS,
                   "exempt_types": frozenset({"string_fragment", "escape_sequence", "template_string"}),
                   "package_markers": ("package.json",), "real_fim_eval": "typescript", "benchmark_extensions": (".ts",)},
    "go": {"grammar": _go, "grammar_package": "tree-sitter-go", "extensions": (".go",),
           "comment_types": frozenset({"comment"}), "keywords": GO_KEYWORDS,
           "exempt_types": frozenset({"interpreted_string_literal_content", "raw_string_literal_content", "escape_sequence"}),
           "package_markers": ("go.mod",), "real_fim_eval": "go", "benchmark_extensions": (".go",)},
    "cpp": {"grammar": _cpp, "grammar_package": "tree-sitter-cpp", "extensions": (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"),
            "comment_types": frozenset({"comment"}), "keywords": CPP_KEYWORDS,
            "exempt_types": frozenset({"string_content", "escape_sequence", "raw_string_content", "char_literal"}),
            "package_markers": ("CMakeLists.txt",), "real_fim_eval": "cpp",
            # C is not a Real-FIM-Eval language: the .c files inside its cpp subset are excluded from evaluation.
            "benchmark_extensions": (".cpp", ".cc")},
}


def get(lang: str) -> dict:
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}; known: {sorted(LANGS)}")
    return LANGS[lang]


@lru_cache(maxsize=None)
def language(lang: str) -> ts.Language:
    return ts.Language(get(lang)["grammar"]())


def parser(lang: str) -> ts.Parser:
    return ts.Parser(language(lang))
