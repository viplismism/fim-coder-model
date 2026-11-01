"""DeepSeek-Coder PSM formatting, shared by training and scoring.

Sequence: BOS <fim_begin> prefix <fim_hole> suffix <fim_end> middle [EOS]
Segments are tokenized separately (no special tokens added by the tokenizer) and concatenated,
so the answer tokens are identical whether a sample is trained on or scored.
"""

import math

IGNORE = -100


class PSM:
    def __init__(self, tokenizer, fmt: dict):
        self.tok = tokenizer
        ids = {k: tokenizer.convert_tokens_to_ids(fmt[k]) for k in ("bos", "eos", "fim_begin", "fim_hole", "fim_end")}
        for k, v in ids.items():
            if v is None or v == tokenizer.unk_token_id:
                raise ValueError(f"format token {k}={fmt[k]!r} is not a single vocabulary token")
        if ids["bos"] != tokenizer.bos_token_id or ids["eos"] != tokenizer.eos_token_id:
            raise ValueError("configured BOS/EOS differ from the tokenizer's")
        self.ids = ids
        self.overhead = 4  # BOS + three FIM sentinels (EOS counted separately)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"] if text else []

    def build(self, prefix: str, suffix: str, middle: str, add_eos: bool) -> dict:
        """Token ids with labels: only middle (+ EOS if add_eos) is supervised."""
        i = self.ids
        prompt = [i["bos"], i["fim_begin"], *self.encode(prefix), i["fim_hole"], *self.encode(suffix), i["fim_end"]]
        answer = self.encode(middle) + ([i["eos"]] if add_eos else [])
        return {"input_ids": prompt + answer, "labels": [IGNORE] * len(prompt) + answer,
                "n_prompt": len(prompt), "n_answer": len(answer)}

    def build_l2r(self, text: str) -> dict:
        """Ordinary left-to-right: BOS text; every text token is scored."""
        ids = self.encode(text)
        return {"input_ids": [self.ids["bos"], *ids], "labels": [IGNORE, *ids], "n_prompt": 1, "n_answer": len(ids)}


def char_perplexity(sum_logprob: float, n_chars: int) -> float:
    """exp(-sum of answer-token log probs / number of answer characters)."""
    return math.exp(-sum_logprob / n_chars)
