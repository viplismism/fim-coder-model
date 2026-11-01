"""Model loading shared by training and scoring: the same base preparation for both arms and the baseline."""

import torch


def bnb_config(cfg):
    from transformers import BitsAndBytesConfig
    q = cfg["quantization"]
    return BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
                              bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
                              bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]))


def load_tokenizer(cfg):
    """Pinned tokenizer; refuses to return one that alters code whitespace on a round trip."""
    from transformers import AutoTokenizer
    from .tokenizer_whitespace import assert_tokenizer_preserves_code_whitespace
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
    assert_tokenizer_preserves_code_whitespace(tok)
    return tok


def load_base(cfg):
    """The base model exactly as both arms and the baseline see it (weights frozen by the caller / PEFT)."""
    from transformers import AutoModelForCausalLM
    name, rev, precision = cfg["model"]["name"], cfg["model"]["revision"], cfg["base_precision"]
    if precision == "bf16":
        return AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map={"": 0})
    if precision == "nf4":
        model = AutoModelForCausalLM.from_pretrained(name, revision=rev, quantization_config=bnb_config(cfg), device_map={"": 0})
        if cfg["prepare_model_for_kbit_training"]:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        return model
    raise ValueError(f"base_precision must be bf16 or nf4, got {precision!r}")


def load_for_scoring(cfg, adapter=None):
    """Base (same preparation as training), optionally with a trained adapter. Base and tuned use this path."""
    model = load_base(cfg)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model
