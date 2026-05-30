#!/usr/bin/env python3
"""Quick FIM model testing - sanity checks before full evaluation."""

import torch
import time
import argparse
from pathlib import Path

# Test cases for FIM completion
TEST_CASES = [
    {
        "name": "Rust: Simple function body",
        "prefix": "fn calculate_sum(numbers: &[i32]) -> i32 {\n    ",
        "suffix": "\n}",
        "expected_contains": ["iter", "sum"],
        "language": "rust"
    },
    {
        "name": "Rust: Struct field access",
        "prefix": "impl PaymentData {\n    fn get_amount(&self) -> u64 {\n        ",
        "suffix": "\n    }\n}",
        "expected_contains": ["self.", "amount"],
        "language": "rust"
    },
    {
        "name": "Rust: Match arm",
        "prefix": "match status {\n    Status::Success => ",
        "suffix": ",\n    Status::Failed => Err(\"failed\"),\n}",
        "expected_contains": ["Ok"],
        "language": "rust"
    },
    {
        "name": "Rust: Error handling",
        "prefix": "fn process_payment(data: &PaymentRequest) -> Result<Response, Error> {\n    let amount = data.amount.ok_or(",
        "suffix": ")?;\n    Ok(Response::new(amount))\n}",
        "expected_contains": ["Error", "missing", "amount"],
        "language": "rust"
    },
    {
        "name": "Rust: Async function",
        "prefix": "async fn fetch_data(client: &Client, url: &str) -> Result<Data, Error> {\n    let response = client.get(url)",
        "suffix": ";\n    response.json().await\n}",
        "expected_contains": [".await", "send"],
        "language": "rust"
    },
    {
        "name": "Rust: Vector operations",
        "prefix": "fn filter_active(items: Vec<Item>) -> Vec<Item> {\n    items.into_iter()",
        "suffix": "\n}",
        "expected_contains": ["filter", "collect"],
        "language": "rust"
    },
    {
        "name": "Rust: Trait implementation",
        "prefix": "impl Display for Transaction {\n    fn fmt(&self, f: &mut Formatter) -> fmt::Result {\n        ",
        "suffix": "\n    }\n}",
        "expected_contains": ["write!", "self"],
        "language": "rust"
    },
    {
        "name": "Rust: Option handling",
        "prefix": "fn get_user_name(user: Option<&User>) -> String {\n    user.map(|u| ",
        "suffix": ").unwrap_or_default()\n}",
        "expected_contains": ["name", "clone", "to_string"],
        "language": "rust"
    },
]


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(adapter_path: str, base_model: str = "deepseek-ai/deepseek-coder-6.7b-base", use_quantization: bool = True):
    """Load the fine-tuned model with LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter: {adapter_path}")
    
    # Quantization only works on CUDA
    if use_quantization and device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # MPS or CPU - no quantization, use float16
        dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if device == "mps":
            model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    print("Model loaded successfully\n")
    return model, tokenizer


def generate_fim(model, tokenizer, prefix: str, suffix: str, max_tokens: int = 64, temperature: float = 0.2):
    """Generate FIM completion."""
    prompt = f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time
    
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    # Stop at newline or end of logical completion
    if "\n\n" in generated:
        generated = generated.split("\n\n")[0]
    
    return generated.strip(), elapsed


def run_test(model, tokenizer, test_case: dict, verbose: bool = True):
    """Run a single test case and return results."""
    generated, elapsed = generate_fim(
        model, tokenizer,
        test_case["prefix"],
        test_case["suffix"]
    )
    
    # Check if expected keywords are in the output
    contains_expected = any(
        kw.lower() in generated.lower() 
        for kw in test_case["expected_contains"]
    )
    
    result = {
        "name": test_case["name"],
        "passed": contains_expected,
        "generated": generated,
        "time_ms": elapsed * 1000,
        "expected_keywords": test_case["expected_contains"]
    }
    
    if verbose:
        status = "PASS" if contains_expected else "FAIL"
        print(f"[{status}] {test_case['name']}")
        print(f"  Prefix: ...{test_case['prefix'][-50:]}".replace("\n", "\\n"))
        print(f"  Generated: {generated[:100]}".replace("\n", "\\n"))
        print(f"  Time: {elapsed*1000:.1f}ms")
        if not contains_expected:
            print(f"  Expected to contain one of: {test_case['expected_contains']}")
        print()
    
    return result


def run_all_tests(model, tokenizer, verbose: bool = True):
    """Run all test cases and summarize results."""
    print("=" * 60)
    print("FIM MODEL QUICK TEST")
    print("=" * 60 + "\n")
    
    results = []
    for test_case in TEST_CASES:
        result = run_test(model, tokenizer, test_case, verbose)
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_time = sum(r["time_ms"] for r in results) / total
    
    print("=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Average generation time: {avg_time:.1f}ms")
    print("=" * 60)
    
    return results


def interactive_mode(model, tokenizer):
    """Interactive testing mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE FIM TESTING")
    print("Enter prefix and suffix to test FIM completion")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")
    
    while True:
        print("Enter PREFIX (end with empty line):")
        prefix_lines = []
        while True:
            line = input()
            if line == "":
                break
            prefix_lines.append(line)
        prefix = "\n".join(prefix_lines)
        
        if prefix.lower() == "quit":
            break
        
        print("Enter SUFFIX (end with empty line):")
        suffix_lines = []
        while True:
            line = input()
            if line == "":
                break
            suffix_lines.append(line)
        suffix = "\n".join(suffix_lines)
        
        print("\nGenerating...")
        generated, elapsed = generate_fim(model, tokenizer, prefix, suffix)
        
        print(f"\n--- RESULT ({elapsed*1000:.1f}ms) ---")
        print(f"{prefix}{generated}{suffix}")
        print("--- END ---\n")


def main():
    parser = argparse.ArgumentParser(description="Quick FIM model testing")
    parser.add_argument("--adapter", default="viplismism/deepseek-coder-6.7b-fim",
                       help="HuggingFace adapter path or local path")
    parser.add_argument("--base_model", default="deepseek-ai/deepseek-coder-6.7b-base",
                       help="Base model name (use 7B or 14B for Mac)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only show summary")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Disable quantization (required for MPS)")
    args = parser.parse_args()
    
    # Auto-detect: disable quantization on non-CUDA
    use_quant = not args.no_quantize and torch.cuda.is_available()
    
    model, tokenizer = load_model(args.adapter, args.base_model, use_quantization=use_quant)
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        run_all_tests(model, tokenizer, verbose=not args.quiet)


if __name__ == "__main__":
    main()
