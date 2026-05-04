"""Agent for coreference resolution in conversations."""

from datetime import date
import json

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from src.shared.prompts import COREFERENCE_EXAMPLE, COREFERENCE_RESOLUTION
from src.client.openai_compatible import LLM
from src.shared.schemas import OPENAI_MESSAGE


class CoreferenceAgent:
    """Agent for coreference resolution."""

    def __init__(self, llm: LLM) -> None:
        """Initialize the CoreferenceAgent."""
        self.llm = llm

    def resolve(self, conversation: list[OPENAI_MESSAGE]) -> str:
        """Resolve coreferences in document messages within the conversation.

        Args:
            conversation: List of messages in the conversation.

        Returns:
            list[dict]: The resolved text.

        """
        today = date.today().isoformat()  # e.g. 2026-01-22

        prompt: list[OPENAI_MESSAGE] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=COREFERENCE_RESOLUTION.format(date=today),
            ),
            *COREFERENCE_EXAMPLE,
            ChatCompletionUserMessageParam(
                role="user",
                content=json.dumps(conversation, ensure_ascii=False),
            ),
        ]

        return self.llm.generate(prompt).strip()
