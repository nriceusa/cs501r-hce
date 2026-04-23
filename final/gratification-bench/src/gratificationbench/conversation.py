"""Multi-turn LLM conversation runner supporting multiple providers.

Providers are imported lazily so you only need to install the SDK for
the provider you're actually using.

Usage:
    client = get_client("gemini", api_key="...")
    result = client.run_conversation(test_case)
    print(result["scored_response"])
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from gratificationbench.prompts import get_system_prompt


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send a list of messages and return the model's response text.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}

        Returns:
            The model's response as a plain string.
        """
        ...

    def run_conversation(
        self,
        test_case: dict[str, Any],
        system_prompt_variant: str = "default",
    ) -> dict[str, Any]:
        """Run a test case as a multi-turn conversation.

        Builds the conversation from the test case's setup_turns, generating
        model responses for each user turn. Then injects the sabotage_trigger
        and captures the model's response — this is the "scored_response."

        Args:
            test_case: A dict conforming to the Gratification Bench test case schema.
            system_prompt_variant: Which system prompt to use (see prompts.py).

        Returns:
            The test_case dict augmented with:
                conversation_history: Full list of {role, content} dicts.
                scored_response: The model's response at the sabotage turn.
        """
        system_prompt = get_system_prompt(system_prompt_variant)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Play out setup turns: user speaks, model responds
        for turn in test_case.get("setup_turns", []):
            messages.append({"role": "user", "content": turn["content"]})
            response = self.chat(messages)
            messages.append({"role": "assistant", "content": response})

        # Inject the sabotage trigger and capture the scored response
        messages.append({"role": "user", "content": test_case["sabotage_trigger"]})
        scored_response = self.chat(messages)
        messages.append({"role": "assistant", "content": scored_response})

        return {
            **test_case,
            "conversation_history": messages,
            "scored_response": scored_response,
        }


class GeminiClient(LLMClient):
    """Google Gemini client via AI Studio API.

    Install: pip install google-genai>=1.51.0
    NOTE: Uses the new google-genai SDK (not the deprecated google-generativeai).
    Free tier: generous daily limits on gemini-3-flash-preview
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from google import genai  # type: ignore
        from google.genai import types as _types  # type: ignore

        # 120s timeout so hung connections raise rather than sleep forever
        http_options = _types.HttpOptions(timeout=120_000)
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self.model_name = model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        import time
        from google.genai import types  # type: ignore
        from google.genai.errors import ClientError  # type: ignore

        # Separate system prompt from conversation messages
        system_instruction = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                chat_messages.append(msg)

        # Build config with system instruction if present
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
        ) if system_instruction else None

        # Convert to google-genai Content format
        contents = [
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part(text=msg["content"])],
            )
            for msg in chat_messages
        ]

        for attempt in range(1, 10):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                return response.text
            except Exception as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                retryable = code in (429, 500, 503) or any(
                    x in str(e) for x in ("429", "500", "503", "UNAVAILABLE", "INTERNAL",
                                          "timed out", "timeout", "DeadlineExceeded",
                                          "ConnectError", "EOF", "ConnectionError",
                                          "RemoteDisconnected", "Connection reset")
                )
                if retryable and attempt < 9:
                    wait = min(20 * attempt, 120)  # 20, 40, 60 … capped at 120s
                    print(f"    Gemini {code or 'error'} (attempt {attempt}/9), retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Gemini chat failed after 9 attempts")


class GroqClient(LLMClient):
    """Groq client for fast open-source model inference.

    Install: pip install groq
    Free tier: generous daily limits; 30,000 TPM on llama-4-scout
    Default model: meta-llama/llama-4-scout-17b-16e-instruct (Llama 4 Scout)
    """

    DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from groq import Groq  # type: ignore

        self.client = Groq(api_key=api_key)
        self.model_name = model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        import re
        import time

        for attempt in range(1, 9):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    **kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e)
                err_type = type(e).__name__
                retryable = (
                    "429" in err_str or "rate_limit" in err_str.lower() or "503" in err_str
                    or "ConnectError" in err_type or "APIConnectionError" in err_type
                    or any(x in err_str for x in ("nodename nor servname", "Connection error",
                                                   "ConnectionError", "timed out", "timeout"))
                )
                if not retryable or attempt >= 8:
                    raise

                # Parse "Please try again in Xm Ys" or "in Xs" from the error message
                wait = self._parse_retry_after(err_str)
                if wait is None:
                    wait = min(30 * attempt, 300)  # fallback: 30/60/90…300s
                else:
                    wait = int(wait) + 5  # add 5s buffer

                is_conn = "ConnectError" in err_type or "APIConnectionError" in err_type
                label = "connection error" if is_conn else "rate limit"
                print(f"    Groq {label} (attempt {attempt}/8), retrying in {wait}s…")
                time.sleep(wait)
        raise RuntimeError("Groq chat failed after 8 attempts")

    @staticmethod
    def _parse_retry_after(err_str: str) -> float | None:
        """Extract the suggested retry delay (in seconds) from a Groq error string."""
        import re
        # "Please try again in 4m27.49s"
        m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", err_str)
        if m:
            minutes = int(m.group(1) or 0)
            seconds = float(m.group(2))
            return minutes * 60 + seconds
        return None


class HFInferenceClient(LLMClient):
    """HuggingFace Inference API client.

    Install: pip install huggingface-hub
    Set HUGGINGFACE_API_KEY (or HF_TOKEN) environment variable.
    """

    DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from huggingface_hub import InferenceClient  # type: ignore

        self.client = InferenceClient(model=model, token=api_key)
        self.model_name = model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return response.choices[0].message.content


class GitHubModelsClient(LLMClient):
    """GitHub Models client — free for GitHub Education students.

    Install: pip install openai
    Set GITHUB_TOKEN environment variable (Settings → Developer settings →
    Personal access tokens, or use the token from 'gh auth token').

    Browse models at: https://github.com/marketplace/models
    Default model: meta-llama/Llama-3.3-70B-Instruct (free tier)
    """

    BASE_URL = "https://models.inference.ai.azure.com"
    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(base_url=self.BASE_URL, api_key=api_key, timeout=120.0)
        self.model_name = model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            **kwargs,
        )
        return response.choices[0].message.content


class TogetherClient(LLMClient):
    """Together AI client — $25 free credits for new accounts.

    Install: pip install openai
    Set TOGETHER_API_KEY environment variable.
    Sign up at: https://api.together.ai

    Browse models at: https://api.together.ai/models
    Default model: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free (free tier)
    """

    BASE_URL = "https://api.together.xyz/v1"
    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(base_url=self.BASE_URL, api_key=api_key, timeout=120.0)
        self.model_name = model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            **kwargs,
        )
        return response.choices[0].message.content


class OpenRouterClient(LLMClient):
    """OpenRouter client — OpenAI-compatible gateway to 300+ models.

    Install: pip install openai
    Set OPENROUTER_API_KEY environment variable.

    Free tier: models with the `:free` suffix, 50 requests/day without credits.
    Browse models at: https://openrouter.ai/models

    Default model: meta-llama/llama-3.3-70b-instruct:free
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "google/gemma-4-31b-it:free"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from openai import OpenAI  # type: ignore

        # 120s timeout so hung connections raise rather than sleep forever
        self.client = OpenAI(base_url=self.BASE_URL, api_key=api_key, timeout=120.0)
        self.model_name = model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        import re
        import time

        last_err = None
        for attempt in range(1, 9):  # up to 8 attempts
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    **kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e)
                code = getattr(e, "status_code", None)
                retryable = code in (429, 500, 502, 503, 504) or any(
                    x in err_str for x in ("429", "502", "503", "rate limit", "upstream",
                                           "timed out", "timeout", "RemoteDisconnected",
                                           "ConnectError", "APIConnectionError",
                                           "Connection error", "EOF", "Connection reset")
                ) or type(e).__name__ in ("APIConnectionError", "ConnectError")
                if not retryable or attempt >= 8:
                    raise

                # Try to parse a retry-after delay from the error body
                wait = self._parse_retry_after(err_str)
                if wait is None:
                    wait = min(30 * attempt, 180)  # 30, 60, 90 … 180s cap
                else:
                    wait = int(wait) + 5

                print(f"    [OpenRouter] {code or 'err'} (attempt {attempt}/8), retrying in {wait}s…")
                time.sleep(wait)
                last_err = e
        raise last_err  # type: ignore

    @staticmethod
    def _parse_retry_after(err_str: str) -> float | None:
        """Parse 'retry after Xs' or 'Xm Ys' from an error string."""
        import re
        m = re.search(r"retry.{0,10}?(\d+(?:\.\d+)?)\s*s(?:econds?)?", err_str, re.I)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+)m\s*(\d+(?:\.\d+)?)s", err_str)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        return None


# Provider registry
_PROVIDERS: dict[str, type[LLMClient]] = {
    "gemini": GeminiClient,
    "groq": GroqClient,
    "huggingface": HFInferenceClient,
    "hf": HFInferenceClient,
    "openrouter": OpenRouterClient,
    "or": OpenRouterClient,
    "github": GitHubModelsClient,
    "together": TogetherClient,
}

# Environment variable names for API keys
_API_KEY_ENV_VARS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "hf": "HUGGINGFACE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "or": "OPENROUTER_API_KEY",
    "github": "GITHUB_TOKEN",
    "together": "TOGETHER_API_KEY",
}


def get_client(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Factory function for LLM clients.

    Args:
        provider: One of "gemini", "groq", "huggingface" (or "hf")
        api_key: API key. If None, reads from environment variable.
        model: Model name override. Uses provider default if None.

    Returns:
        An LLMClient instance for the requested provider.
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {list(_PROVIDERS.keys())}"
        )

    if api_key is None:
        env_var = _API_KEY_ENV_VARS[provider]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"No API key found for provider '{provider}'. "
                f"Pass api_key= or set the {env_var} environment variable."
            )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if model:
        kwargs["model"] = model

    return _PROVIDERS[provider](**kwargs)
