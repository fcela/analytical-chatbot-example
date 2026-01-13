"""LLM wrapper utility for the chatbot with multi-provider support.

Supports:
- OpenAI (default)
- Anthropic (Claude)
- Ollama (local)

Provider selection via LLM_PROVIDER environment variable.
"""

import os
import time
from typing import List, Optional

# Provider constants
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OLLAMA = "ollama"


def _mock_response(prompt: str) -> str:
    """Return a deterministic mock response for local testing."""
    lower = prompt.lower()
    if "2 + 2" in lower or "what is 2 + 2" in lower:
        return "4"
    if "average" in lower or "mean" in lower:
        return "The average is 42 (mock)."
    if "plot" in lower or "chart" in lower:
        return "I generated a plot (mock)."
    return "MOCK: This is a synthetic reply for local testing."


def _call_openai(
    prompt: str,
    system_prompt: Optional[str],
    messages: Optional[List[dict]],
    max_retries: int
) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        import httpx
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    message_list = []
    if system_prompt:
        message_list.append({"role": "system", "content": system_prompt})
    if messages:
        message_list.extend(messages)
    message_list.append({"role": "user", "content": prompt})

    backoff = 1.0
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=message_list,
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            )
            return response.choices[0].message.content
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            
    raise last_exception or RuntimeError("Max retries exceeded")


def _call_anthropic(
    prompt: str,
    system_prompt: Optional[str],
    messages: Optional[List[dict]],
    max_retries: int
) -> str:
    """Call Anthropic (Claude) API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)

    # Build messages list for Anthropic format
    message_list = []
    if messages:
        for msg in messages:
            role = msg.get("role", "user")
            # Anthropic only accepts "user" and "assistant" roles
            if role == "system":
                continue  # System prompt handled separately
            message_list.append({
                "role": role,
                "content": msg.get("content", "")
            })
    message_list.append({"role": "user", "content": prompt})

    backoff = 1.0
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
                system=system_prompt or "",
                messages=message_list,
            )
            return response.content[0].text
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            
    raise last_exception or RuntimeError("Max retries exceeded")


def _call_ollama(
    prompt: str,
    system_prompt: Optional[str],
    messages: Optional[List[dict]],
    max_retries: int
) -> str:
    """Call Ollama local API."""
    debug = os.getenv("LLM_DEBUG", "false").lower() in ("1", "true", "yes")

    if debug:
        print(f"[LLM_DEBUG] Ollama: Loading ollama package...")

    try:
        from ollama import chat
    except ImportError:
        raise ImportError("Ollama package not installed. Run: pip install ollama")

    message_list = []
    if system_prompt:
        message_list.append({"role": "system", "content": system_prompt})
    if messages:
        message_list.extend(messages)
    message_list.append({"role": "user", "content": prompt})

    model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    if debug:
        print(f"[LLM_DEBUG] Ollama: Using model '{model}'")
        print(f"[LLM_DEBUG] Ollama: Message count: {len(message_list)}")
        print(f"[LLM_DEBUG] Ollama: Prompt length: {len(prompt)} chars")
        if system_prompt:
            print(f"[LLM_DEBUG] Ollama: System prompt length: {len(system_prompt)} chars")

    backoff = 1.0
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if debug:
                print(f"[LLM_DEBUG] Ollama: Starting API call (attempt {attempt + 1}/{max_retries + 1})...")
                start_time = time.time()

            # Enable streaming to avoid timeouts/empty replies with some models
            response_stream = chat(
                model=model,
                messages=message_list,
                stream=True,
            )

            full_content = []
            timeout = float(os.getenv("LLM_TIMEOUT", "120")) # 2 minutes default timeout
            
            for chunk in response_stream:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Ollama generation timed out after {timeout}s")
                
                content = chunk.get('message', {}).get('content', '')
                if content:
                    full_content.append(content)
            
            result = "".join(full_content)

            if debug:
                elapsed = time.time() - start_time
                print(f"[LLM_DEBUG] Ollama: Response received in {elapsed:.2f}s")
                print(f"[LLM_DEBUG] Ollama: Response length: {len(result)} chars")

            return result
        except Exception as e:
            last_exception = e
            if debug:
                print(f"[LLM_DEBUG] Ollama: Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                if debug:
                    print(f"[LLM_DEBUG] Ollama: Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            
    raise last_exception or RuntimeError("Max retries exceeded")


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    *,
    max_retries: int = 2
) -> str:
    """
    Call the LLM with a prompt and optional system prompt.

    Provider is selected via LLM_PROVIDER environment variable:
    - "openai" (default): Uses OpenAI API
    - "anthropic": Uses Anthropic (Claude) API
    - "ollama": Uses local Ollama server

    Environment Variables:
    - LLM_PROVIDER: Provider selection (openai, anthropic, ollama)
    - MOCK_LLM: Set to "true" to return mock responses
    - LLM_TEMPERATURE: Temperature for generation (default 0.7)

    Provider-specific variables:
    - OpenAI: OPENAI_API_KEY, OPENAI_MODEL (default: gpt-4o)
    - Anthropic: ANTHROPIC_API_KEY, ANTHROPIC_MODEL (default: claude-sonnet-4-20250514)
    - Ollama: OLLAMA_MODEL (default: qwen3:4b)

    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt for context
        messages: Optional list of previous messages for context
        max_retries: Number of retries for transient errors

    Returns:
        The LLM's response text
    """
    debug = os.getenv("LLM_DEBUG", "false").lower() in ("1", "true", "yes")

    # Mock mode (useful for local dev without network or API keys)
    if os.getenv("MOCK_LLM", "false").lower() in ("1", "true", "yes"):
        if debug:
            print(f"[LLM_DEBUG] Mock mode enabled, returning mock response")
        return _mock_response(prompt)

    provider = os.getenv("LLM_PROVIDER", PROVIDER_OPENAI).lower().strip()

    if debug:
        print(f"[LLM_DEBUG] Provider: {provider}")
        print(f"[LLM_DEBUG] Prompt preview: {prompt[:100]}..." if len(prompt) > 100 else f"[LLM_DEBUG] Prompt: {prompt}")

    if provider == PROVIDER_OPENAI:
        return _call_openai(prompt, system_prompt, messages, max_retries)
    elif provider == PROVIDER_ANTHROPIC:
        return _call_anthropic(prompt, system_prompt, messages, max_retries)
    elif provider == PROVIDER_OLLAMA:
        return _call_ollama(prompt, system_prompt, messages, max_retries)
    else:
        return f"[LLM_ERROR] Unknown provider: {provider}. Use 'openai', 'anthropic', or 'ollama'."


def get_provider_info() -> dict:
    """Get information about the current LLM provider configuration."""
    provider = os.getenv("LLM_PROVIDER", PROVIDER_OPENAI).lower().strip()

    info = {
        "provider": provider,
        "mock_mode": os.getenv("MOCK_LLM", "false").lower() in ("1", "true", "yes"),
    }

    if provider == PROVIDER_OPENAI:
        info["model"] = os.getenv("OPENAI_MODEL", "gpt-4o")
        info["api_key_set"] = bool(os.getenv("OPENAI_API_KEY"))
    elif provider == PROVIDER_ANTHROPIC:
        info["model"] = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        info["api_key_set"] = bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == PROVIDER_OLLAMA:
        info["model"] = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        info["api_key_set"] = True  # Ollama doesn't need an API key

    return info


if __name__ == "__main__":
    # Quick manual tests
    print("Provider info:", get_provider_info())
    print("\nTest query:")
    print(call_llm("What is 2 + 2?"))
