#!/usr/bin/env python3
"""
Comprehensive FIM model benchmarking with pass@k metrics.
Compares fine-tuned model against baseline on test dataset.
"""

import os
import json
import time
import random
import argparse
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


@dataclass
class GenerationResult:
    text: str
    time_ms: float
    tokens: int


@dataclass 
class SampleMetrics:
    exact_match: bool
    pass_at_1: bool
    pass_at_3: bool
    pass_at_5: bool
    edit_similarity: float
    token_overlap: float
    bleu_score: float
    generation_time_ms: float
    generated_tokens: int


@dataclass
class BenchmarkResult:
    sample_id: int
    file_path: str
    node_type: str
    expected: str
    tuned_output: str
    baseline_output: str
    tuned_metrics: SampleMetrics
    baseline_metrics: SampleMetrics


class FIMEvaluator:
    """Evaluator for FIM models."""
    
    def __init__(self, model_path: str, is_adapter: bool = False, base_model: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        if is_adapter:
            from peft import PeftModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print(f"Loaded: {model_path}")
    
    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.0) -> GenerationResult:
        """Generate single completion."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start
        
        gen_tokens = outputs[0][input_len:]
        generated = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        return GenerationResult(
            text=generated.strip(),
            time_ms=elapsed * 1000,
            tokens=len(gen_tokens)
        )
    
    def generate_k(self, prompt: str, k: int = 5, max_tokens: int = 128, temperature: float = 0.8) -> List[str]:
        """Generate k completions for pass@k evaluation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        return [
            self.tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True).strip()
            for i in range(k)
        ]


def build_fim_prompt(prefix: str, suffix: str) -> str:
    """Build FIM prompt with DeepSeek Coder tokens."""
    return f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def edit_similarity(generated: str, expected: str) -> float:
    """Compute edit similarity (1 - normalized edit distance)."""
    gen, exp = generated.strip(), expected.strip()
    if not gen and not exp:
        return 1.0
    if not gen or not exp:
        return 0.0
    
    dist = edit_distance(gen, exp)
    return 1.0 - dist / max(len(gen), len(exp))


def token_overlap(generated: str, expected: str) -> float:
    """Compute token-level overlap score."""
    gen_tokens = set(generated.split())
    exp_tokens = set(expected.split())
    
    if not gen_tokens or not exp_tokens:
        return 0.0
    
    intersection = gen_tokens & exp_tokens
    union = gen_tokens | exp_tokens
    
    return len(intersection) / len(union)


def simple_bleu(generated: str, expected: str, max_n: int = 4) -> float:
    """Compute simple BLEU-like score."""
    def get_ngrams(text: str, n: int) -> Dict[tuple, int]:
        tokens = text.split()
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i:i+n])] += 1
        return ngrams
    
    if not generated.strip() or not expected.strip():
        return 0.0
    
    scores = []
    for n in range(1, max_n + 1):
        gen_ngrams = get_ngrams(generated, n)
        exp_ngrams = get_ngrams(expected, n)
        
        if not gen_ngrams:
            continue
        
        matches = sum(min(gen_ngrams[ng], exp_ngrams.get(ng, 0)) for ng in gen_ngrams)
        total = sum(gen_ngrams.values())
        
        if total > 0:
            scores.append(matches / total)
    
    if not scores:
        return 0.0
    
    # Geometric mean
    import math
    return math.exp(sum(math.log(s) for s in scores if s > 0) / len(scores)) if all(s > 0 for s in scores) else 0.0


def compute_metrics(
    generated: str, 
    expected: str, 
    k_samples: List[str],
    gen_time_ms: float,
    gen_tokens: int
) -> SampleMetrics:
    """Compute all metrics for a single sample."""
    gen_clean = generated.strip()
    exp_clean = expected.strip()
    
    exact = gen_clean == exp_clean
    
    return SampleMetrics(
        exact_match=exact,
        pass_at_1=exact,
        pass_at_3=any(s.strip() == exp_clean for s in k_samples[:3]),
        pass_at_5=any(s.strip() == exp_clean for s in k_samples[:5]),
        edit_similarity=edit_similarity(gen_clean, exp_clean),
        token_overlap=token_overlap(gen_clean, exp_clean),
        bleu_score=simple_bleu(gen_clean, exp_clean),
        generation_time_ms=gen_time_ms,
        generated_tokens=gen_tokens
    )


def run_benchmark(
    tuned_evaluator: FIMEvaluator,
    baseline_evaluator: FIMEvaluator,
    test_data: List[Dict],
    max_samples: Optional[int] = None,
    k_samples: int = 5
) -> List[BenchmarkResult]:
    """Run full benchmark comparison."""
    
    if max_samples and max_samples < len(test_data):
        random.seed(42)
        test_data = random.sample(test_data, max_samples)
    
    results = []
    total = len(test_data)
    
    for i, item in enumerate(test_data):
        prefix = item['prefix']
        suffix = item['suffix']
        expected = item['middle']
        file_path = item.get('filePath', 'unknown')
        node_type = item.get('nodeType', 'unknown')
        
        prompt = build_fim_prompt(prefix, suffix)
        
        # Generate with tuned model
        tuned_result = tuned_evaluator.generate(prompt)
        tuned_k = tuned_evaluator.generate_k(prompt, k=k_samples)
        
        # Generate with baseline
        baseline_result = baseline_evaluator.generate(prompt)
        baseline_k = baseline_evaluator.generate_k(prompt, k=k_samples)
        
        # Compute metrics
        tuned_metrics = compute_metrics(
            tuned_result.text, expected, tuned_k,
            tuned_result.time_ms, tuned_result.tokens
        )
        baseline_metrics = compute_metrics(
            baseline_result.text, expected, baseline_k,
            baseline_result.time_ms, baseline_result.tokens
        )
        
        results.append(BenchmarkResult(
            sample_id=i,
            file_path=file_path,
            node_type=node_type,
            expected=expected,
            tuned_output=tuned_result.text,
            baseline_output=baseline_result.text,
            tuned_metrics=tuned_metrics,
            baseline_metrics=baseline_metrics
        ))
        
        # Progress
        print(f"\r[{i+1}/{total}] Tuned pass@1: {sum(1 for r in results if r.tuned_metrics.pass_at_1)}/{i+1}", end="", flush=True)
    
    print()
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary."""
    n = len(results)
    
    def agg(model: str, metric: str) -> float:
        metrics = [getattr(r.tuned_metrics if model == "tuned" else r.baseline_metrics, metric) for r in results]
        return sum(metrics)
    
    def avg(model: str, metric: str) -> float:
        return agg(model, metric) / n
    
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS ({n} samples)")
    print("=" * 70)
    
    # Main metrics table
    print("\n{:<20} {:>15} {:>15} {:>15}".format("Metric", "Tuned", "Baseline", "Delta"))
    print("-" * 70)
    
    metrics_to_show = [
        ("Pass@1", "pass_at_1", True),
        ("Pass@3", "pass_at_3", True),
        ("Pass@5", "pass_at_5", True),
        ("Edit Similarity", "edit_similarity", False),
        ("Token Overlap", "token_overlap", False),
        ("BLEU Score", "bleu_score", False),
        ("Avg Time (ms)", "generation_time_ms", False),
    ]
    
    for name, metric, is_count in metrics_to_show:
        if is_count:
            tuned_val = int(agg("tuned", metric))
            base_val = int(agg("baseline", metric))
            delta = tuned_val - base_val
            print("{:<20} {:>12}/{:<2} {:>12}/{:<2} {:>+15}".format(
                name, tuned_val, n, base_val, n, delta
            ))
        else:
            tuned_val = avg("tuned", metric)
            base_val = avg("baseline", metric)
            delta = tuned_val - base_val
            print("{:<20} {:>15.3f} {:>15.3f} {:>+15.3f}".format(
                name, tuned_val, base_val, delta
            ))
    
    # Per node type breakdown
    print("\n" + "-" * 70)
    print("PASS@1 BY NODE TYPE:")
    print("-" * 70)
    
    by_type = defaultdict(list)
    for r in results:
        by_type[r.node_type].append(r)
    
    print("{:<25} {:>10} {:>10} {:>10}".format("Node Type", "Tuned", "Baseline", "Delta"))
    for node_type, type_results in sorted(by_type.items(), key=lambda x: -len(x[1])):
        tuned_pass = sum(1 for r in type_results if r.tuned_metrics.pass_at_1)
        base_pass = sum(1 for r in type_results if r.baseline_metrics.pass_at_1)
        total = len(type_results)
        delta = tuned_pass - base_pass
        print("{:<25} {:>7}/{:<2} {:>7}/{:<2} {:>+10}".format(
            node_type, tuned_pass, total, base_pass, total, delta
        ))
    
    print("=" * 70)


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save detailed results to JSON."""
    output = {
        "metadata": {
            "total_samples": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {
            "tuned": {
                "pass_at_1": sum(1 for r in results if r.tuned_metrics.pass_at_1),
                "pass_at_3": sum(1 for r in results if r.tuned_metrics.pass_at_3),
                "pass_at_5": sum(1 for r in results if r.tuned_metrics.pass_at_5),
                "avg_edit_similarity": sum(r.tuned_metrics.edit_similarity for r in results) / len(results),
                "avg_time_ms": sum(r.tuned_metrics.generation_time_ms for r in results) / len(results),
            },
            "baseline": {
                "pass_at_1": sum(1 for r in results if r.baseline_metrics.pass_at_1),
                "pass_at_3": sum(1 for r in results if r.baseline_metrics.pass_at_3),
                "pass_at_5": sum(1 for r in results if r.baseline_metrics.pass_at_5),
                "avg_edit_similarity": sum(r.baseline_metrics.edit_similarity for r in results) / len(results),
                "avg_time_ms": sum(r.baseline_metrics.generation_time_ms for r in results) / len(results),
            }
        },
        "results": [
            {
                "sample_id": r.sample_id,
                "file_path": r.file_path,
                "node_type": r.node_type,
                "expected": r.expected,
                "tuned_output": r.tuned_output,
                "baseline_output": r.baseline_output,
                "tuned_metrics": asdict(r.tuned_metrics),
                "baseline_metrics": asdict(r.baseline_metrics),
            }
            for r in results
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FIM Model Benchmark")
    parser.add_argument("--tuned", default="viplismism/deepseek-coder-6.7b-fim",
                       help="Fine-tuned model/adapter path")
    parser.add_argument("--baseline", default="deepseek-ai/deepseek-coder-6.7b-base",
                       help="Baseline model path")
    parser.add_argument("--test_data", default=None,
                       help="Test data JSONL path")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to evaluate")
    parser.add_argument("--output", default=None,
                       help="Output JSON path")
    parser.add_argument("--k", type=int, default=5,
                       help="K for pass@k evaluation")
    args = parser.parse_args()
    
    # Paths
    root = Path(__file__).parent.parent
    test_path = Path(args.test_data) if args.test_data else root / "data" / "reth_test.jsonl"
    output_path = Path(args.output) if args.output else root / "evaluation" / "results" / f"benchmark_{int(time.time())}.json"
    
    print("FIM Model Benchmark")
    print("=" * 50)
    print(f"Tuned:    {args.tuned}")
    print(f"Baseline: {args.baseline}")
    print(f"Test:     {test_path}")
    print(f"Samples:  {args.max_samples or 'all'}")
    print("=" * 50 + "\n")
    
    # Load test data
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("Generate test data first with: python datagen/datagen.py")
        return
    
    with open(test_path) as f:
        test_data = [json.loads(line) for line in f]
    print(f"Loaded {len(test_data)} test samples\n")
    
    # Load models
    print("Loading models...")
    tuned = FIMEvaluator(args.tuned, is_adapter=True, base_model=args.baseline)
    baseline = FIMEvaluator(args.baseline, is_adapter=False)
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = run_benchmark(tuned, baseline, test_data, args.max_samples, args.k)
    
    # Print and save results
    print_summary(results)
    save_results(results, output_path)


if __name__ == "__main__":
    main()
