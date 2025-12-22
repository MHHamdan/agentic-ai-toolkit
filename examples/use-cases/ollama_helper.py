#!/usr/bin/env python3
"""
Ollama Helper Module

Provides utilities for calling local Ollama models with real token tracking.
Used by use-case examples for actual LLM inference instead of mocks.

Supported models (run `ollama list` to see available):
- gemma2:2b (fast, lightweight)
- phi3:latest (fast, good reasoning)
- llama3.2:3b (balanced)
- mistral:latest (good quality)
- llama3.1:8b (high quality)
- qwen2.5:14b (highest quality, slower)
"""

import requests
import time
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class OllamaResponse:
    """Response from Ollama model with token metrics."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model: str
    success: bool
    error: Optional[str] = None


class OllamaClient:
    """Client for local Ollama API with token tracking."""

    def __init__(
        self,
        model: str = "gemma2:2b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "gemma2:2b", "phi3:latest")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._total_tokens_used = 0
        self._total_requests = 0

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> OllamaResponse:
        """
        Generate text completion with token tracking.

        Args:
            prompt: User prompt
            system: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            OllamaResponse with text and metrics
        """
        start_time = time.time()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            latency = time.time() - start_time

            # Extract token counts from response
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens

            self._total_tokens_used += total_tokens
            self._total_requests += 1

            return OllamaResponse(
                text=data.get("response", ""),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                model=self.model,
                success=True
            )

        except requests.exceptions.ConnectionError:
            return OllamaResponse(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=time.time() - start_time,
                model=self.model,
                success=False,
                error="Connection failed. Is Ollama running? Try: ollama serve"
            )
        except requests.exceptions.Timeout:
            return OllamaResponse(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=time.time() - start_time,
                model=self.model,
                success=False,
                error=f"Request timed out after {self.timeout}s"
            )
        except Exception as e:
            return OllamaResponse(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=time.time() - start_time,
                model=self.model,
                success=False,
                error=str(e)
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> OllamaResponse:
        """
        Chat completion with message history.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            OllamaResponse with text and metrics
        """
        start_time = time.time()

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            latency = time.time() - start_time

            # Extract token counts
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens

            self._total_tokens_used += total_tokens
            self._total_requests += 1

            return OllamaResponse(
                text=data.get("message", {}).get("content", ""),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                model=self.model,
                success=True
            )

        except Exception as e:
            return OllamaResponse(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=time.time() - start_time,
                model=self.model,
                success=False,
                error=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get cumulative usage statistics."""
        return {
            "total_tokens": self._total_tokens_used,
            "total_requests": self._total_requests,
            "model": self.model
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self._total_tokens_used = 0
        self._total_requests = 0


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_ollama_ready(model: str = "gemma2:2b", base_url: str = "http://localhost:11434") -> tuple:
    """
    Check if Ollama is ready for inference (server running + model loadable).

    Returns:
        (ready: bool, error_message: str or None)
    """
    if not check_ollama_available(base_url):
        return False, "Ollama server not running. Start with: ollama serve"

    # Try a minimal generation to check if model can load
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1}
            },
            timeout=30
        )
        if response.status_code == 200:
            return True, None
        else:
            data = response.json()
            error = data.get("error", "Unknown error")
            if "out of memory" in error.lower():
                return False, f"GPU out of memory. Free GPU resources or use smaller model."
            return False, error
    except requests.exceptions.Timeout:
        return False, "Model loading timed out"
    except Exception as e:
        return False, str(e)


def list_models(base_url: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except:
        pass
    return []


def select_best_model(
    preferred: List[str] = None,
    base_url: str = "http://localhost:11434"
) -> Optional[str]:
    """
    Select best available model from preferred list.

    Args:
        preferred: List of models in order of preference
        base_url: Ollama server URL

    Returns:
        Best available model name or None
    """
    if preferred is None:
        # Default preference: fast to slow
        preferred = [
            "gemma2:2b",
            "phi3:latest",
            "llama3.2:3b",
            "llama3.2:latest",
            "mistral:latest",
            "llama3.1:8b",
            "qwen2.5:14b"
        ]

    available = list_models(base_url)
    for model in preferred:
        if model in available:
            return model

    # Return first available if no preferred match
    return available[0] if available else None


# Token cost estimation (for local models, actual cost is compute time)
# These rates are for comparison with cloud API pricing
TOKEN_RATES = {
    "gemma2:2b": 0.00001,      # Very cheap (small model)
    "phi3:latest": 0.00001,    # Very cheap (small model)
    "llama3.2:3b": 0.00002,    # Cheap
    "llama3.2:latest": 0.00002,
    "mistral:latest": 0.00003, # Moderate
    "llama3.1:8b": 0.00005,    # Higher
    "qwen2.5:14b": 0.0001,     # Highest (large model)
}


def estimate_cost(tokens: int, model: str) -> float:
    """Estimate inference cost based on token count."""
    rate = TOKEN_RATES.get(model, 0.00003)  # Default rate
    return tokens * rate


if __name__ == "__main__":
    # Test the helper
    print("Checking Ollama availability...")

    if not check_ollama_available():
        print("ERROR: Ollama not running. Start with: ollama serve")
        exit(1)

    print("Available models:", list_models())

    model = select_best_model()
    print(f"Selected model: {model}")

    if model:
        client = OllamaClient(model=model)
        print(f"\nTesting {model}...")

        response = client.generate(
            "What is 2 + 2? Answer in one word.",
            max_tokens=10
        )

        if response.success:
            print(f"Response: {response.text}")
            print(f"Tokens: {response.total_tokens}")
            print(f"Latency: {response.latency_seconds:.2f}s")
            print(f"Estimated cost: ${estimate_cost(response.total_tokens, model):.6f}")
        else:
            print(f"Error: {response.error}")
