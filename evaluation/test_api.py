#!/usr/bin/env python3
"""
VS Code Extension / IDE integration test.
Simulates how a code editor would call the FIM API.
"""

import requests
import json
import time
from typing import Optional

API_BASE = "http://localhost:8000"


def test_fim_completion(prefix: str, suffix: str, max_tokens: int = 64) -> dict:
    """Test FIM completion endpoint."""
    response = requests.post(
        f"{API_BASE}/v1/fim/completions",
        json={
            "prefix": prefix,
            "suffix": suffix,
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
    )
    return response.json()


def test_openai_compatible(prompt: str, suffix: Optional[str] = None) -> dict:
    """Test OpenAI-compatible endpoint."""
    response = requests.post(
        f"{API_BASE}/v1/openai/completions",
        json={
            "model": "fim-32b",
            "prompt": prompt,
            "suffix": suffix,
            "max_tokens": 64,
            "temperature": 0.2
        }
    )
    return response.json()


def test_health() -> dict:
    """Test health endpoint."""
    response = requests.get(f"{API_BASE}/health")
    return response.json()


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("FIM API Integration Tests")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n[Test 1] Health Check")
    try:
        health = test_health()
        print(f"  Status: {health['status']}")
        print(f"  Model: {health['model_name']}")
        print(f"  Device: {health['device']}")
        assert health['status'] == 'healthy', "Health check failed"
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    # Test 2: Basic FIM completion
    print("\n[Test 2] Basic FIM Completion")
    try:
        result = test_fim_completion(
            prefix="fn add(a: i32, b: i32) -> i32 {\n    ",
            suffix="\n}"
        )
        print(f"  Completion: {result['completion']}")
        print(f"  Time: {result['time_ms']:.1f}ms")
        print(f"  Tokens: {result['tokens']}")
        assert result['completion'], "Empty completion"
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 3: Rust struct completion
    print("\n[Test 3] Struct Implementation")
    try:
        result = test_fim_completion(
            prefix="""impl Payment {
    fn new(amount: u64, currency: &str) -> Self {
        Self {
            """,
            suffix="""
        }
    }
}"""
        )
        print(f"  Completion: {result['completion'][:100]}...")
        print(f"  Time: {result['time_ms']:.1f}ms")
        assert 'amount' in result['completion'].lower(), "Expected 'amount' in completion"
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 4: OpenAI-compatible format
    print("\n[Test 4] OpenAI-Compatible Endpoint")
    try:
        result = test_openai_compatible(
            prompt="<｜fim▁begin｜>fn main() {\n    let x = <｜fim▁hole｜>;\n    println!(\"{}\", x);\n}<｜fim▁end｜>"
        )
        completion = result['choices'][0]['text']
        print(f"  Completion: {completion}")
        print(f"  Tokens: {result['usage']['completion_tokens']}")
        assert completion, "Empty completion"
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 5: Error handling with function body
    print("\n[Test 5] Complex Function Body")
    try:
        result = test_fim_completion(
            prefix="""async fn process_transaction(tx: &Transaction) -> Result<Receipt, Error> {
    let validated = tx.validate()?;
    """,
            suffix="""
    Ok(Receipt::new(validated))
}"""
        )
        print(f"  Completion: {result['completion'][:150]}...")
        print(f"  Time: {result['time_ms']:.1f}ms")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 6: Latency test (multiple requests)
    print("\n[Test 6] Latency Test (10 requests)")
    try:
        times = []
        for i in range(10):
            start = time.time()
            result = test_fim_completion(
                prefix="let x = ",
                suffix=";",
                max_tokens=16
            )
            times.append((time.time() - start) * 1000)
        
        avg = sum(times) / len(times)
        p50 = sorted(times)[len(times)//2]
        p99 = sorted(times)[int(len(times)*0.99)]
        
        print(f"  Avg: {avg:.1f}ms")
        print(f"  P50: {p50:.1f}ms")
        print(f"  P99: {p99:.1f}ms")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Integration tests completed")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_tests()
