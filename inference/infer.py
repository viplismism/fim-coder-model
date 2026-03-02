#!/usr/bin/env python3
"""FIM model evaluation: compare tuned vs baseline."""

import os, json, time, random, argparse
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Result:
    text: str
    time: float
    tokens: int

class Evaluator:
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        
        is_local = Path(model_path).is_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=is_local)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto", trust_remote_code=True, local_files_only=is_local)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print(f"Loaded: {model_path}")

    def generate(self, prompt: str, max_tokens: int = 128) -> Result:
        start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                       pad_token_id=self.tokenizer.pad_token_id)
        gen_ids = out[0][input_len:] if out.dim() > 1 else out[input_len:]
        return Result(self.tokenizer.decode(gen_ids, skip_special_tokens=True), time.time() - start, len(gen_ids))

    def generate_k(self, prompt: str, k: int = 5, max_tokens: int = 128) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                       temperature=0.8, top_p=0.95, num_return_sequences=k,
                                       pad_token_id=self.tokenizer.pad_token_id)
        return [self.tokenizer.decode(out[i][input_len:], skip_special_tokens=True) for i in range(k)]

def baseline_prompt(prefix: str, suffix: str) -> str:
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

def tuned_prompt(prefix: str, suffix: str, path: str, repo: str = "reth") -> str:
    return f"<|repo_name|>{repo}\n<|file_sep|>{path}\n<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

def edit_sim(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if a[i-1] == b[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return 1.0 - dp[m][n] / max(m, n)

def word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / max(len(wa), len(wb)) if wa and wb else 0.0

def evaluate(tuned_path: str, baseline_path: str, test_path: str, max_samples: int = None):
    tuned, baseline = Evaluator(tuned_path), Evaluator(baseline_path)
    
    with open(test_path) as f:
        data = [json.loads(line) for line in f]
    if max_samples and max_samples < len(data):
        random.seed(42)
        data = random.sample(data, max_samples)
    
    results = []
    for i, item in enumerate(data):
        print(f"\r[{i+1}/{len(data)}]", end="", flush=True)
        prefix, suffix, expected, path = item['prefix'], item['suffix'], item['middle'], item.get('filePath', 'unknown.rs')
        
        t_res = tuned.generate(tuned_prompt(prefix, suffix, path))
        b_res = baseline.generate(baseline_prompt(prefix, suffix))
        t_k = tuned.generate_k(tuned_prompt(prefix, suffix, path), k=5)
        b_k = baseline.generate_k(baseline_prompt(prefix, suffix), k=5)
        
        def metrics(gen, exp, samples):
            exact = gen.strip() == exp.strip()
            return {'generated': gen, 'exact': exact, 'pass@1': exact,
                    'pass@3': any(s.strip() == exp.strip() for s in samples[:3]),
                    'pass@5': any(s.strip() == exp.strip() for s in samples),
                    'edit_sim': edit_sim(gen, exp), 'word_overlap': word_overlap(gen, exp)}
        
        results.append({'id': i, 'path': path, 'expected': expected,
            'tuned': metrics(t_res.text, expected, t_k) | {'time': t_res.time, 'tokens': t_res.tokens},
            'baseline': metrics(b_res.text, expected, b_k) | {'time': b_res.time, 'tokens': b_res.tokens}})
    print()
    return results

def summary(results: List[Dict]):
    n = len(results)
    agg = lambda k, f: sum(r[k][f] for r in results)
    avg = lambda k, f: agg(k, f) / n
    
    print(f"\n{'='*50}\nRESULTS ({n} samples)\n{'='*50}")
    for name, key in [("TUNED", "tuned"), ("BASELINE", "baseline")]:
        print(f"\n{name}: pass@1={agg(key,'pass@1')}/{n} ({agg(key,'pass@1')/n*100:.1f}%) | "
              f"pass@5={agg(key,'pass@5')}/{n} | edit_sim={avg(key,'edit_sim'):.3f}")
    print(f"\nΔ pass@1: {(agg('tuned','pass@1')-agg('baseline','pass@1'))/n*100:+.1f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--baseline", default="Qwen/Qwen2.5-Coder-32B")
    parser.add_argument("--test_data", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    test_path = args.test_data or str(root / "data" / "reth_test.jsonl")
    
    print(f"Tuned: {args.tuned}\nBaseline: {args.baseline}\nTest: {test_path}\n")
    results = evaluate(args.tuned, args.baseline, test_path, args.max_samples)
    summary(results)
    
    out = root / "inference" / "results" / (args.output or f"results_{int(time.time())}.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()