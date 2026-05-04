"""OpenAI-compatible LLM client."""

import json
import os
from pathlib import Path
import re
from datetime import datetime
from typing import Any

from openai import OpenAI

from src.shared.schemas import OPENAI_MESSAGE
from src.shared.conversations import pretty_print_conversation_using_tab


class LLM:
    """Simple OpenAI-compatible LLM client.

    Attributes:
        client: OpenAI client instance
        model: Model name
        temperature: Sampling temperature
        max_tokens: Optional max tokens for generation

    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        ensure_nonempty: bool = True,
    ) -> None:
        """Initialize the LLM client.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the OpenAI API
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Optional max tokens for generation
            timeout: Request timeout in seconds

        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ensure_nonempty = ensure_nonempty

    def generate(self, messages: list[OPENAI_MESSAGE], response_format: Any = None, think: bool = False) -> str:
        """Generate a completion from OpenAI-style messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_format: Optional response format specification
            think: Whether to enable the model's "thinking" capability

        Returns:
            str: Generated text from the model

        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": think,
                },
            },
            response_format=response_format,
        )

        raw_text = response.choices[0].message.content or ""
        stripped_text = self._strip_thinking(raw_text)

        if (log_to := os.environ.get('OPENAI_LOG_DIR', None)):
            Path(log_to).mkdir(exist_ok=True, parents=True)
            stem = datetime.now().strftime("%b %d, %Y %I:%M:%S %p")
            Path(f'{log_to}/{stem} in.txt').write_text(
                pretty_print_conversation_using_tab(messages), # type: ignore
            )
            Path(f'{log_to}/{stem} in.json').write_text(
                json.dumps(messages, indent=4, ensure_ascii=False),
            )
            Path(f'{log_to}/{stem} think.txt').write_text(raw_text)
            Path(f'{log_to}/{stem} out.txt').write_text(stripped_text)

        if (
            self.ensure_nonempty
            and (stripped_text is None or stripped_text.strip() == "") # pyright: ignore[reportUnnecessaryComparison]
        ):
            raise RuntimeError(
                "Empty model output detected. Stopping to avoid writing invalid predictions.\n"
                f"base_url={self.client._base_url}\n" # pyright: ignore[reportPrivateUsage]
                f"model={self.model}\n"
                "Common causes: vLLM server not running, wrong base_url, model name mismatch, auth mismatch.",
            )

        return stripped_text

    def _strip_thinking(self, text: str) -> str:
        """Удаляет блоки <think>...</think> и возвращает финальный ответ модели."""
        # Удаляем все think-блоки (на случай, если их несколько)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        return cleaned.strip()


if __name__ == "__main__":
    host = "http://localhost:8000/v1"
    api_key = "testkey"

    llm = LLM(
        api_key=api_key,
        base_url=host,
        model="Qwen/Qwen3-4B-FP8",
    )

    messages: list[OPENAI_MESSAGE] = [
        {"role": "system", "content": "You are RAGU, an AI assistant, that helps users."},
        {"role": "user", "content": "Who are you?"},
    ]
    output = llm.generate(messages)
    print("Output:", output)
