"""Agent for coreference resolution in conversations."""

from datetime import date

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from shared.prompts import COREFERENCE_EXAMPLE

from src.client.openai_compatible import LLM
from src.shared.conversations import extended_to_openai
from src.shared.prompts import COREFERENCE_RESOLUTION
from src.shared.schemas import ExtendedConversation, OpenAIDocument


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

        user_mess_id = max(
            i for i, msg in enumerate(conversation.messages)
            if not isinstance(msg, OpenAIDocument) and msg['role'] == "user"
        )

        until_last_query = conversation.messages[: user_mess_id + 1]

        user_message_contents = str(extended_to_openai(until_last_query))
        ready_to_llm = [
            ChatCompletionUserMessageParam(role="user", content=user_message_contents),
        ]

        prompt = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=COREFERENCE_RESOLUTION.format(date=today),
            ),
        ] + COREFERENCE_EXAMPLE + ready_to_llm

        resolved_text = self.llm.generate(prompt)
        conversation.messages[user_mess_id] = ChatCompletionUserMessageParam(role="user", content=resolved_text.strip())

        return conversation
