"""Exact encode/decode round trip of the pinned DeepSeek tokenizer on all pinned Rust text."""
import gzip, json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer

MODEL, REV = "deepseek-ai/deepseek-coder-6.7b-base", "ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912"
root = Path(__file__).resolve().parents[2]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
raw = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json", revision=REV))
print(type(tok).__name__, "bos", tok.bos_token, tok.bos_token_id, "eos", tok.eos_token, tok.eos_token_id)

texts = [("reth:" + str(p.relative_to(root / "sources/reth")), p.read_text(encoding="utf-8"))
         for p in sorted((root / "sources/reth").rglob("*.rs"))]
for split in ("add", "edit"):
    for i, line in enumerate(gzip.open(root / f"sources/real_fim_eval/{split}/test.jsonl.gz", "rt")):
        r = json.loads(line)
        if r["lang"] == "rust":
            for k in ("prompt", "canonical_solution", "suffix", "to_remove"):
                if r.get(k):
                    texts.append((f"{split}:{i}:{k}", r[k]))

fails, mismatch, n_tok = [], [], 0
for name, t in texts:
    ids = tok(t, add_special_tokens=False)["input_ids"]
    n_tok += len(ids)
    if tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != t:
        fails.append(name)
    if raw.encode(t, add_special_tokens=False).ids != ids:
        mismatch.append(name)
print(f"texts {len(texts):,}  tokens {n_tok:,}  round-trip failures {len(fails)}  ids differ from raw tokenizers backend {len(mismatch)}")
print("first failures:", fails[:5], "first mismatches:", mismatch[:5])

# Special tokens and default BOS/EOS behaviour
fim = {"begin": "<｜fim▁begin｜>", "hole": "<｜fim▁hole｜>", "end": "<｜fim▁end｜>"}
ids = tok(fim["begin"] + "a" + fim["hole"] + "b" + fim["end"])["input_ids"]
print("default encode adds BOS:", ids[0] == tok.bos_token_id, "| adds EOS:", ids[-1] == tok.eos_token_id)
print("FIM token ids:", {k: tok.convert_tokens_to_ids(v) for k, v in fim.items()}, "| full ids:", ids)
# Failures caused only by DeepSeek's byte-fragment added tokens (ids 32000-32012) are reported separately.
frag = {t.content for t in raw.get_added_tokens_decoder().values() if len(t.content) == 1}
def only_fragment_chars(name):
    t = dict(texts)[name]
    d = tok.decode(tok(t, add_special_tokens=False)["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return len(d) == len(t) and all(a == b or (a in frag and b == "\ufffd") for a, b in zip(t, d))
explained = [n for n in fails if only_fragment_chars(n)]
report = {"model": MODEL, "revision": REV, "texts": len(texts), "tokens": n_tok, "roundtrip_failures": fails,
          "failures_explained_by_byte_fragment_added_tokens": len(explained), "unexplained_failures": sorted(set(fails) - set(explained)),
          "ids_differ_from_raw_backend": mismatch, "byte_fragment_added_tokens": sorted(frag),
          "default_encode_adds_bos": ids[0] == tok.bos_token_id, "default_encode_adds_eos": ids[-1] == tok.eos_token_id}
(Path(__file__).parent / "tokenizer_roundtrip.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
print("explained by byte-fragment tokens:", len(explained), "unexplained:", report["unexplained_failures"])
sys.exit(1 if report["unexplained_failures"] or mismatch else 0)
