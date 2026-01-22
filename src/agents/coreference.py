"""Agent for coreference resolution in conversations."""

from datetime import date

from src.client.openai_compatible import LLM
from src.shared.prompts import COREFERENCE_RESOLUTION
from src.shared.schemas import ExtendedConversation, ExtendedMessage, OpenAIConversation, OpenAIMessage


class CoreferenceAgent:
    """Agent for coreference resolution."""

    def __init__(self, llm: LLM) -> None:
        """Initialize the CoreferenceAgent."""
        self.llm = llm

    def resolve(self, conversation: ExtendedConversation) -> ExtendedConversation:
        """Resolve coreferences in document messages within the conversation.

        Args:
            conversation: List of messages in the conversation.

        Returns:
            list[dict]: List of messages with coreferences resolved in document messages.

        """
        today = date.today().isoformat()  # e.g. 2026-01-22

        resolved_docs = []

        for message in conversation.messages:
            prompt = OpenAIConversation(
                [
                    OpenAIMessage("system", COREFERENCE_RESOLUTION.format(date=today)),
                    OpenAIMessage("user", message.content),
                ],
            )

            resolved_text = self.llm.generate(prompt)

            resolved_docs.append(
                ExtendedMessage(message.role, resolved_text.strip()),
            )

        return ExtendedConversation(resolved_docs)
