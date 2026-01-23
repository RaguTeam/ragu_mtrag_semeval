"""Agent for reviewing document relevance. Selects relevant documents based on user queries."""

from datetime import date

from src.client.openai_compatible import LLM
from src.shared.prompts import RELEVANCE_FILTERING
from src.shared.schemas import ExtendedConversation, OpenAIConversation, OpenAIMessage


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
        today = date.today().isoformat()  # e.g. 2026-01-22
        user_question = next(m.content for m in reversed(conversation.messages) if m.role == "user")

        filtered_docs = []
        context = [m for m in conversation.messages if m.role != "document"]

        for doc in [m for m in conversation.messages if m.role == "document"]:
            prompt = OpenAIConversation(
                [
                    OpenAIMessage("system", RELEVANCE_FILTERING.format(date=today)),
                    OpenAIMessage("user", f"Question: {user_question}"),
                    OpenAIMessage("user", f"Document: {doc.content}"),
                ],
            )
            verdict = self.llm.generate(prompt).strip().lower()

            if verdict == "yes":
                filtered_docs.append(doc)

        return ExtendedConversation(context + filtered_docs)
