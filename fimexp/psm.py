"""PSM (prefix-suffix-middle) FIM formatting, shared by training, scoring and generation.

DeepSeek: BOS <fim_begin> prefix <fim_hole> suffix <fim_end> middle [EOS]; Qwen2.5-Coder: see class PSM.
Segments are tokenized separately (no special tokens added by the tokenizer) and concatenated,
so the answer tokens are identical whether a sample is trained on or scored.
"""

import math

IGNORE = -100


class PSM:
    """fmt["style"] selects the documented FIM layout of the model family (default "deepseek"):
      deepseek: BOS <｜fim▁begin｜> prefix <｜fim▁hole｜> suffix <｜fim▁end｜> middle [EOS]
      qwen:     <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle [<|endoftext|>]   (no BOS;
                Qwen2.5-Coder technical report / model card, file-level FIM)
    Both put prefix then suffix in the prompt and the middle last (PSM)."""

    def __init__(self, tokenizer, fmt: dict):
        self.tok = tokenizer
        self.style = fmt.get("style", "deepseek")
        keys = {"deepseek": ("bos", "eos", "fim_begin", "fim_hole", "fim_end"),
                "qwen": ("eos", "fim_prefix", "fim_suffix", "fim_middle")}[self.style]
        ids = {k: tokenizer.convert_tokens_to_ids(fmt[k]) for k in keys}
        for k, v in ids.items():
            if v is None or v == tokenizer.unk_token_id:
                raise ValueError(f"format token {k}={fmt[k]!r} is not a single vocabulary token")
        if self.style == "deepseek":
            if ids["bos"] != tokenizer.bos_token_id or ids["eos"] != tokenizer.eos_token_id:
                raise ValueError("configured BOS/EOS differ from the tokenizer's")
            self.overhead = 4  # BOS + three FIM sentinels (EOS counted separately)
            self.stop_ids = [ids["eos"], ids["fim_begin"], ids["fim_hole"], ids["fim_end"]]
        else:
            self.overhead = 3  # three FIM sentinels (EOS counted separately)
            extra = [tokenizer.convert_tokens_to_ids(t) for t in fmt.get("extra_stop", [])]
            self.stop_ids = [ids["eos"], ids["fim_prefix"], ids["fim_suffix"], ids["fim_middle"], *extra]
        self.ids = ids

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"] if text else []

    def build(self, prefix: str, suffix: str, middle: str, add_eos: bool) -> dict:
        """Token ids with labels: only middle (+ EOS if add_eos) is supervised."""
        prompt = self.prompt(prefix, suffix)
        i = self.ids
        answer = self.encode(middle) + ([i["eos"]] if add_eos else [])
        return {"input_ids": prompt + answer, "labels": [IGNORE] * len(prompt) + answer,
                "n_prompt": len(prompt), "n_answer": len(answer)}

    def build_l2r(self, text: str) -> dict:
        """Ordinary left-to-right: BOS text; every text token is scored."""
        ids = self.encode(text)
        start = self.ids["bos"] if self.style == "deepseek" else self.ids["eos"]  # Qwen: <|endoftext|> as document start
        return {"input_ids": [start, *ids], "labels": [IGNORE, *ids], "n_prompt": 1, "n_answer": len(ids)}

    def prompt(self, prefix: str, suffix: str) -> list[int]:
        """Prompt ids up to (and including) the token after which the middle is generated."""
        i = self.ids
        if self.style == "deepseek":
            return [i["bos"], i["fim_begin"], *self.encode(prefix), i["fim_hole"], *self.encode(suffix), i["fim_end"]]
        return [i["fim_prefix"], *self.encode(prefix), i["fim_suffix"], *self.encode(suffix), i["fim_middle"]]


def char_perplexity(sum_logprob: float, n_chars: int) -> float:
    """exp(-sum of answer-token log probs / number of answer characters)."""
    return math.exp(-sum_logprob / n_chars)
