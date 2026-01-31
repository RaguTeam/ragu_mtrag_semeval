"""Agent for reviewing document relevance. Selects relevant documents based on user queries."""

from datetime import date

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.shared.schemas import OPENAI_MESSAGE
from src.client.openai_compatible import LLM
from src.shared.prompts import RELEVANCE_FILTERING
from src.shared.schemas import ExtendedConversation, OpenAIDocument


class RelevanceAgent:
    """Agent for reviewing document relevance."""

    def __init__(self, llm: LLM) -> None:
        """Initialize the RelevanceAgent."""
        self.llm = llm

    def filter(self, conversation: ExtendedConversation) -> ExtendedConversation:
        """Filter documents based on relevance to the user question.

        Args:
            conversation: List of messages in the conversation.

        Returns:
            str: List of messages with only relevant documents retained.

        """
        today = date.today().isoformat()
        user_question = next(
            m['content']
            for m in reversed(conversation.messages)
            if not isinstance(m, OpenAIDocument) and m['role'] == "user"
        )

        filtered_docs: list[OpenAIDocument] = []
        context = [m for m in conversation.messages if not isinstance(m, OpenAIDocument)]

        for doc in [m for m in conversation.messages if isinstance(m, OpenAIDocument)]:
            prompt: list[OPENAI_MESSAGE] = [
                ChatCompletionSystemMessageParam(role="system", content=RELEVANCE_FILTERING.format(date=today)),
                ChatCompletionUserMessageParam(role="user", content=f"Question: {user_question}"),
                ChatCompletionUserMessageParam(role="user", content=f"Document: {doc.text}"),
            ]
            verdict = self.llm.generate(prompt).strip().lower()

            if verdict == "yes":
                filtered_docs.append(doc)

        return ExtendedConversation(context + filtered_docs)
