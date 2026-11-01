"""Tokenizer checks and decode helpers for code whitespace preservation."""


WHITESPACE_PROBES = [
    "pub use builder::*;\n",
    "fn clone(&self) -> Self {\n    Self { value: self.value.clone() }\n}\n",
    "<｜fim▁begin｜>impl X {\n    <｜fim▁hole｜>\n}<｜fim▁end｜>Self { value }\n<｜end▁of▁sentence｜>",
]


def decode_preserving_whitespace(tokenizer, token_ids, *, skip_special_tokens: bool = True) -> str:
    """Decode without tokenizer cleanup that can normalize code whitespace."""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def assert_tokenizer_preserves_code_whitespace(tokenizer, probes: list[str] | None = None) -> None:
    """Fail fast if encode/decode changes spaces or newlines in code samples."""
    probes = probes or WHITESPACE_PROBES
    for text in probes:
        encoded = tokenizer(text, add_special_tokens=False)
        decoded = decode_preserving_whitespace(
            tokenizer,
            encoded["input_ids"],
            skip_special_tokens=False,
        )
        if decoded != text:
            raise RuntimeError(
                "Tokenizer does not preserve code whitespace on round trip.\n"
                f"Input:  {text!r}\n"
                f"Output: {decoded!r}\n"
                "Do not train or evaluate with this tokenizer stack."
            )
