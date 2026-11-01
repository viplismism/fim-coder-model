"""Scoring tests on a tiny random model: answer boundaries, EOS exclusion, char normalization, Edit layout,
L2R windows, collator masking."""

import math

import pytest
import torch

from fimexp import sources
from fimexp.holes import SourceText
from fimexp.psm import IGNORE, PSM, char_perplexity
from fimexp.score import answer_logprobs, edit_layout, l2r_windows
from fimexp.train import Collator, missing_settings

CFG = sources.load_config()


@pytest.fixture(scope="module")
def psm():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["model"]["name"], revision=CFG["model"]["revision"])
    return PSM(tok, CFG["format"])


@pytest.fixture(scope="module")
def tiny(psm):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=len(psm.tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096))
    return m.eval()


def test_answer_logprobs_match_full_forward(psm, tiny):
    b = psm.build("fn main() {\n    ", "\n}\n", "let x = 1;", add_eos=False)
    lps = answer_logprobs(tiny, b)
    assert len(lps) == b["n_answer"] == len(psm.encode("let x = 1;"))
    ids = torch.tensor([b["input_ids"]])
    with torch.no_grad():
        full = torch.log_softmax(tiny(input_ids=ids).logits[0].float(), -1)
    ref = [full[i - 1, ids[0, i]].item() for i in range(b["n_prompt"], len(b["input_ids"]))]
    assert lps == pytest.approx(ref, abs=1e-5)
    # matches HF's own shifted loss over the same labels
    with torch.no_grad():
        loss = tiny(input_ids=ids, labels=torch.tensor([b["labels"]])).loss.item()
    assert -sum(lps) / len(lps) == pytest.approx(loss, rel=1e-4)


def test_scoring_excludes_eos_and_normalizes_by_chars(psm, tiny):
    middle = "x + ü"
    with_eos = psm.build("a", "b", middle, add_eos=True)
    without = psm.build("a", "b", middle, add_eos=False)
    assert with_eos["n_answer"] == without["n_answer"] + 1
    lps = answer_logprobs(tiny, without)
    assert char_perplexity(sum(lps), len(middle)) == pytest.approx(math.exp(-sum(lps) / 5))
    assert len(middle) == 5 and len(middle.encode()) == 6  # characters, not bytes


def test_edit_layout_is_line_aligned():
    t = CFG["evaluation"]["edit_template"]
    r = {"prompt": "mod a;\n", "to_remove": "pub use h::H;\n", "canonical_solution": "pub use crate::h::H;\n", "suffix": "fn f() {}\n"}
    prefix, suffix = edit_layout(r, t)
    assert prefix == "mod a;\n<<<<<<< ORIGINAL\npub use h::H;\n=======\n"
    assert suffix == ">>>>>>> UPDATED\nfn f() {}\n"
    assert t["status"] == "inferred"


def test_edit_layout_on_real_rows():
    t = CFG["evaluation"]["edit_template"]
    rows = [r for r in sources.benchmark_rows(CFG) if r["split"] == "edit"][:200]
    for r in rows:
        prefix, suffix = edit_layout(r, t)
        assert prefix.endswith("=======\n") and suffix.startswith(">>>>>>> UPDATED\n")
        assert (prefix + r["canonical_solution"] + suffix).count("<<<<<<< ORIGINAL\n") >= 1


def test_l2r_windows_reconstruct_file(psm):
    text = "".join(f"fn f{i}() -> u32 {{ {i} }}\n" for i in range(600))
    windows, skipped = l2r_windows(SourceText(text), psm, 256)
    assert not skipped and "".join(windows) == text and len(windows) > 1
    assert all(len(psm.build_l2r(w)["input_ids"]) <= 256 for w in windows)


def test_collator_masks_padding():
    c = Collator(pad_id=7)
    out = c([{"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3]}, {"input_ids": [4], "labels": [4]}])
    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 7, 7]]
    assert out["labels"].tolist() == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_trainer_refuses_undecided_settings():
    import copy
    cfg = copy.deepcopy(CFG)
    assert missing_settings(cfg) == []
    cfg["training"]["learning_rate"] = None
    assert missing_settings(cfg) == ["training.learning_rate"]


def test_update_counter_counts_model_inputs_per_update(psm, tmp_path):
    """Real Trainer, tiny model: logged examples/tokens per update equal what each update consumed."""
    import json as _json
    from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
    from fimexp.train import FIMDataset, UpdateCounter, make_callback, make_trainer_class
    rows = []
    for i in range(40):
        pre, mid, suf = f"fn f{i}() {{\n    ", "x" * (1 + i % 7), "\n}\n"
        rows.append({"id": str(i), "prefix": pre, "middle": mid, "suffix": suf,
                     "tokens": {"total_with_eos": len(psm.build(pre, suf, mid, add_eos=True)["input_ids"])}})
    ds = FIMDataset(rows, psm, 4096)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(psm.tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=2))
    counter = UpdateCounter(4096)
    from fimexp.train import Collator
    args = TrainingArguments(output_dir=str(tmp_path / "o"), per_device_train_batch_size=1, gradient_accumulation_steps=16,
                             num_train_epochs=1, logging_steps=1, report_to="none", use_cpu=True, save_strategy="no",
                             train_sampling_strategy="sequential", remove_unused_columns=False)
    t = make_trainer_class(counter)(model=model, args=args, train_dataset=ds, data_collator=Collator(psm.ids["eos"]),
                                    callbacks=[make_callback(tmp_path / "m.jsonl", counter)])
    t.train()
    logged = [_json.loads(l) for l in open(tmp_path / "m.jsonl") if '"loss"' in l]
    sizes = [r["tokens"]["total_with_eos"] for r in rows]
    assert [r["examples"] for r in logged] == [16, 16, 8]
    assert [r["tokens"] for r in logged] == [sum(sizes[:16]), sum(sizes[16:32]), sum(sizes[32:])]
