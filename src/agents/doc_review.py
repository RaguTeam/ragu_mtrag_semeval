"""Agent for reviewing document relevance. Selects relevant documents based on user queries."""

from datetime import date

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.shared.schemas import OPENAI_MESSAGE
from src.client.openai_compatible import LLM
from src.shared.prompts import RELEVANCE_FILTERING
from src.shared.schemas import OpenAIDocument


class RelevanceAgent:
    """Agent for reviewing document relevance."""

    def __init__(self, llm: LLM) -> None:
        """Initialize the RelevanceAgent."""
        self.llm = llm

    def filter(
        self,
        conversation: list[OPENAI_MESSAGE],
        docs: list[OpenAIDocument],
    ) -> list[int]:
        """Filter documents based on relevance to the user question.

        Args:
            conversation: List of messages in the conversation.

        Returns:
            str: Indices of relevant docs.

        """
        today = date.today().isoformat()

        assert conversation[-1]['role'] == 'user'
        user_question = conversation[-1]['content']
        assert isinstance(user_question, str)

        filtered_doc_idx: list[int] = []

        for doc_idx, doc in enumerate(docs):
            prompt: list[OPENAI_MESSAGE] = [
                ChatCompletionSystemMessageParam(role="system", content=RELEVANCE_FILTERING.format(date=today)),
                ChatCompletionUserMessageParam(role="user", content=f"Question: {user_question}"),
                ChatCompletionUserMessageParam(role="user", content=f"Document: {doc.text}"),
            ]
            verdict = self.llm.generate(prompt).strip().lower()

            if verdict == "yes":
                filtered_doc_idx.append(doc_idx)

        return filtered_doc_idx
