#!/usr/bin/env python3
"""
FastAPI server for FIM model inference.
Production-ready API for code completion.
"""

import os
import time
import torch
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

# Suppress transformers warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response models
class FIMRequest(BaseModel):
    """FIM completion request."""
    prefix: str = Field(..., description="Code before the cursor")
    suffix: str = Field(..., description="Code after the cursor")
    max_tokens: int = Field(128, ge=1, le=512, description="Maximum tokens to generate")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Sampling temperature")
    stop_sequences: List[str] = Field(default=[], description="Stop sequences")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prefix": "fn calculate_sum(numbers: &[i32]) -> i32 {\n    ",
                "suffix": "\n}",
                "max_tokens": 64,
                "temperature": 0.2
            }
        }


class FIMResponse(BaseModel):
    """FIM completion response."""
    completion: str = Field(..., description="Generated code completion")
    tokens: int = Field(..., description="Number of tokens generated")
    time_ms: float = Field(..., description="Generation time in milliseconds")
    model: str = Field(..., description="Model used for generation")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_name: str


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = ""
    
    def load(
        self, 
        adapter_path: str = "viplismism/deepseek-coder-6.7b-fim",
        base_model: str = "deepseek-ai/deepseek-coder-6.7b-base"
    ):
        """Load the model with LoRA adapter."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        logger.info(f"Loading model from {adapter_path}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.model_name = adapter_path
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate(
        self, 
        prefix: str, 
        suffix: str, 
        max_tokens: int = 128,
        temperature: float = 0.2,
        stop_sequences: List[str] = None
    ) -> tuple[str, int, float]:
        """Generate FIM completion."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        prompt = f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = (time.time() - start) * 1000
        
        gen_tokens = outputs[0][input_len:]
        completion = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop_sequences:
            for stop in stop_sequences:
                if stop in completion:
                    completion = completion.split(stop)[0]
        
        return completion.strip(), len(gen_tokens), elapsed


# Global model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    adapter = os.environ.get("FIM_ADAPTER", "viplismism/deepseek-coder-6.7b-fim")
    base = os.environ.get("FIM_BASE_MODEL", "deepseek-ai/deepseek-coder-6.7b-base")
    model_manager.load(adapter, base)
    yield
    # Cleanup on shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="FIM Code Completion API",
    description="Production API for Fill-in-the-Middle code completion",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "unhealthy",
        model_loaded=model_manager.model is not None,
        device=model_manager.device,
        model_name=model_manager.model_name
    )


@app.post("/v1/fim/completions", response_model=FIMResponse)
async def fim_completion(request: FIMRequest):
    """
    Generate FIM code completion.
    
    Takes code prefix and suffix, returns the completion for the middle.
    """
    try:
        completion, tokens, time_ms = model_manager.generate(
            prefix=request.prefix,
            suffix=request.suffix,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_sequences=request.stop_sequences
        )
        
        return FIMResponse(
            completion=completion,
            tokens=tokens,
            time_ms=time_ms,
            model=model_manager.model_name
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=FIMResponse)
async def legacy_completion(request: FIMRequest):
    """Legacy endpoint for compatibility."""
    return await fim_completion(request)


# OpenAI-compatible endpoint for IDE integration
class OpenAIRequest(BaseModel):
    model: str = "fim-32b"
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.2
    suffix: Optional[str] = None


class OpenAIResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


@app.post("/v1/openai/completions", response_model=OpenAIResponse)
async def openai_compatible(request: OpenAIRequest):
    """
    OpenAI-compatible completion endpoint for IDE integration.
    
    Expects DeepSeek FIM format in prompt: <｜fim▁begin｜>...<｜fim▁hole｜>...<｜fim▁end｜>
    Or uses suffix parameter if provided.
    """
    # Parse FIM format from prompt
    if "<｜fim▁begin｜>" in request.prompt:
        parts = request.prompt.split("<｜fim▁begin｜>")[1].split("<｜fim▁hole｜>")
        prefix = parts[0]
        suffix = parts[1].split("<｜fim▁end｜>")[0] if len(parts) > 1 else ""
    else:
        prefix = request.prompt
        suffix = request.suffix or ""
    
    completion, tokens, time_ms = model_manager.generate(
        prefix=prefix,
        suffix=suffix,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    return OpenAIResponse(
        id=f"cmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "text": completion,
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": 0,  # Not tracked
            "completion_tokens": tokens,
            "total_tokens": tokens
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
