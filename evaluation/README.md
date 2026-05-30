# Evaluation Module

This module contains tools for testing and evaluating the fine-tuned FIM model.

## Quick Test

Run sanity checks on the model with predefined test cases:

```bash
python evaluation/quick_test.py --adapter viplismism/deepseek-coder-6.7b-fim

# Interactive mode
python evaluation/quick_test.py -i
```

## Full Benchmark

Compare tuned model against baseline with pass@k metrics:

```bash
# Full benchmark on test set
python evaluation/benchmark.py --tuned viplismism/deepseek-coder-6.7b-fim --max_samples 100

# With custom test data
python evaluation/benchmark.py --tuned viplismism/deepseek-coder-6.7b-fim --test_data data/reth_test.jsonl
```

Output includes:
- Pass@1, Pass@3, Pass@5
- Edit similarity
- Token overlap
- BLEU score
- Per node-type breakdown

## API Server

Production-ready FastAPI server:

```bash
# Start server
python evaluation/api_server.py

# Or with uvicorn
uvicorn evaluation.api_server:app --host 0.0.0.0 --port 8000

# With custom model
FIM_ADAPTER=viplismism/deepseek-coder-6.7b-fim FIM_BASE_MODEL=deepseek-ai/deepseek-coder-6.7b-base uvicorn evaluation.api_server:app
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/fim/completions` | POST | FIM completion |
| `/v1/openai/completions` | POST | OpenAI-compatible |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/fim/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "fn add(a: i32, b: i32) -> i32 {\n    ",
    "suffix": "\n}",
    "max_tokens": 64,
    "temperature": 0.2
  }'
```

## Integration Tests

Test the API server:

```bash
# Start server first, then:
python evaluation/test_api.py
```
