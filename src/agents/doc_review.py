"""Agent for reviewing document relevance. Selects relevant documents based on user queries."""

from datetime import date

from src.client.openai_compatible import LLM
from src.shared.prompts import RELEVANCE_FILTERING


class RelevanceAgent:
    """Agent for reviewing document relevance."""

    def __init__(self, llm: LLM) -> None:
        """Initialize the RelevanceAgent."""
        self.llm = llm

    def filter(self, conversation: list[dict]) -> list[dict]:
        """Filter documents based on relevance to the user question.

        Args:
            conversation: List of messages in the conversation.

        Returns:
            str: List of messages with only relevant documents retained.

        """
        today = date.today().isoformat()  # e.g. 2026-01-22
        user_question = next(m["content"] for m in reversed(conversation) if m["role"] == "user")

        filtered_docs = []
        context = [m for m in conversation if m["role"] != "document"]

        for doc in [m for m in conversation if m["role"] == "document"]:
            prompt = [
                {
                    "role": "system",
                    "content": RELEVANCE_FILTERING.format(date=today),
                },
                {"role": "user", "content": f"Question: {user_question}"},
                {"role": "user", "content": f"Document: {doc['content']}"},
            ]

            verdict = self.llm.generate(prompt).strip().lower()

            if verdict == "yes":
                filtered_docs.append(doc)

        return context + filtered_docs
