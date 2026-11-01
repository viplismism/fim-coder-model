#!/usr/bin/env python3
"""Environment checks for the shared .venv: CUDA, bitsandbytes 4-bit path, LoRA forward/backward, PSM labels.

No optimizer step is taken: weights are never updated. The LoRA rank here only exercises the code path
and is not an experiment setting. Writes checks/env/report.json.
"""

import json, platform, subprocess, sys, time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from fimexp import sources  # noqa: E402
from fimexp.psm import IGNORE, PSM  # noqa: E402

PKGS = ["torch", "transformers", "tokenizers", "trl", "peft", "accelerate", "datasets", "huggingface-hub",
        "bitsandbytes", "datatrove", "tree-sitter", "tree-sitter-rust", "xxhash", "numpy", "pytest"]


def main():
    cfg = sources.load_config()
    rep = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0],
           "platform": platform.platform(), "packages": {p: version(p) for p in PKGS}}
    rep["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
                                       capture_output=True, text=True).stdout.strip()
    rep["cuda"] = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
                   "arch_list": torch.cuda.get_arch_list(), "bf16_supported": torch.cuda.is_bf16_supported(),
                   "cusparselt_available": torch.backends.cusparselt.is_available()}

    import bitsandbytes as bnb
    from bitsandbytes.cextension import lib
    rep["bitsandbytes"] = {"version": bnb.__version__, "native_lib": getattr(lib, "_name", None) or str(getattr(lib, "lib", lib))}
    # 4-bit matmul round trip against bf16 reference
    w = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(8, 1024, dtype=torch.bfloat16, device="cuda")
    q, state = bnb.functional.quantize_4bit(w, quant_type="nf4")
    y4 = bnb.matmul_4bit(x, q.t(), quant_state=state)
    rel = ((y4.float() - (x @ w.t()).float()).norm() / (x @ w.t()).float().norm()).item()
    rep["bitsandbytes"]["nf4_matmul_rel_error"] = round(rel, 4)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    name, rev = cfg["model"]["name"], cfg["model"]["revision"]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    psm = PSM(tok, cfg["format"])
    qc = cfg["quantization"]
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=qc["load_in_4bit"], bnb_4bit_quant_type=qc["bnb_4bit_quant_type"],
                                 bnb_4bit_use_double_quant=qc["bnb_4bit_use_double_quant"],
                                 bnb_4bit_compute_dtype=getattr(torch, qc["bnb_4bit_compute_dtype"]))

    prefix = "impl Counter {\n    pub fn increment(&mut self) -> u64 {\n        "
    middle = "self.value += 1;\n        self.value"
    suffix = "\n    }\n}\n"
    b = psm.build(prefix, suffix, middle, add_eos=True)
    ids = torch.tensor([b["input_ids"]], device="cuda")
    labels = torch.tensor([b["labels"]], device="cuda")
    rep["label_mask"] = {"n_tokens": len(b["input_ids"]), "n_supervised": sum(l != IGNORE for l in b["labels"]),
                         "supervised_decoded": tok.decode([t for t, l in zip(b["input_ids"], b["labels"]) if l != IGNORE],
                                                          skip_special_tokens=False)}

    def nll(model):
        with torch.no_grad():
            return model(input_ids=ids, labels=labels).loss.item()

    t0 = time.time()
    bf16 = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map={"": 0})
    rep["bf16_base_nll"] = round(nll(bf16), 4)
    del bf16
    torch.cuda.empty_cache()
    base4 = AutoModelForCausalLM.from_pretrained(name, revision=rev, quantization_config=bnb_cfg, device_map={"": 0})
    rep["load_seconds"] = round(time.time() - t0, 1)
    rep["4bit_base_nll"] = round(nll(base4), 4)
    rep["4bit_memory_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    with torch.no_grad():
        base_logits = base4(input_ids=ids).logits.float()

    lcfg = dict(r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    # (a) adapter on the unmodified 4-bit base: a fresh LoRA (B=0) must reproduce the base exactly
    model = get_peft_model(base4, LoraConfig(**lcfg))
    model.eval()
    with torch.no_grad():
        plain_logits = model(input_ids=ids).logits.float()
    rep["fresh_adapter_vs_base_max_abs_logit_diff"] = {"no_prepare": (plain_logits - base_logits).abs().max().item()}

    # backward through the 4-bit base + LoRA with gradient checkpointing (no prepare_model_for_kbit_training)
    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()  # gradient check only; no optimizer, no weight update
    grads = [p.grad for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rep["lora_backward"] = {"loss": round(out.loss.item(), 4), "trainable_params": trainable,
                            "params_with_grad": len(grads), "all_finite": all(torch.isfinite(g).all().item() for g in grads),
                            "grad_norm": round(torch.norm(torch.stack([g.float().norm() for g in grads])).item(), 4),
                            "optimizer_step_taken": False}
    del model, base4
    torch.cuda.empty_cache()

    # (b) same, after prepare_model_for_kbit_training (upcasts non-quantized params to fp32)
    base4 = AutoModelForCausalLM.from_pretrained(name, revision=rev, quantization_config=bnb_cfg, device_map={"": 0})
    dtypes_before = {n: str(p.dtype) for n, p in base4.named_parameters() if p.dtype != torch.uint8}
    prepped = prepare_model_for_kbit_training(base4, use_gradient_checkpointing=True)
    changed = sorted({n.split(".")[-2] if "." in n else n for n, p in prepped.named_parameters()
                      if n in dtypes_before and str(p.dtype) != dtypes_before[n]})
    prepped = get_peft_model(prepped, LoraConfig(**lcfg))
    prepped.eval()
    with torch.no_grad():
        prep_logits = prepped(input_ids=ids).logits.float()
    rep["fresh_adapter_vs_base_max_abs_logit_diff"]["with_prepare_model_for_kbit_training"] = (prep_logits - base_logits).abs().max().item()
    rep["prepare_model_for_kbit_training_upcasts"] = changed
    rep["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    (Path(__file__).parent / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
