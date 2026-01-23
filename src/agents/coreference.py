"""Agent for coreference resolution in conversations."""

from datetime import date

from src.client.openai_compatible import LLM
from src.shared.prompts import COREFERENCE_RESOLUTION
from src.shared.schemas import (
    ExtendedConversation,
    ExtendedMessage,
    OpenAIConversation,
    OpenAIMessage,
    extended_to_openai,
)


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

        user_mess_id = max(i for i, msg in enumerate(conversation.messages) if msg.role == "user")

        until_last_query = conversation.messages[:user_mess_id]
        ready_to_llm = extended_to_openai(until_last_query)

        prompt = OpenAIConversation(
            ready_to_llm
            + [
                OpenAIMessage(
                    "system",
                    COREFERENCE_RESOLUTION.format(date=today, message=conversation.messages[user_mess_id].content),
                ),
            ],
        )

        resolved_text = self.llm.generate(prompt)
        conversation.messages[user_mess_id] = ExtendedMessage("user", resolved_text.strip())

        return conversation
