"""Assembly of multiple agents for question answering."""

import copy
import logging

from src.agents.answer import AnswerAgent
from src.agents.coreference import CoreferenceAgent
from src.agents.doc_review import RelevanceAgent
from src.shared.logging_config import setup_logging
from src.shared.schemas import OPENAI_MESSAGE, OpenAIDocument

setup_logging(
    log_level="INFO",
)

logger = logging.getLogger(__name__)


class MultiAgentQA:
    """Multi-agent question answering system."""

    def __init__(
        self,
        answer_agent: AnswerAgent,
        coref_agent: CoreferenceAgent | None = None,
        relevance_agent: RelevanceAgent | None = None,
    ) -> None:
        """Multi-agent question answering system."""

        self.coref_agent = coref_agent
        self.relevance_agent = relevance_agent
        self.answer_agent = answer_agent

    def run(
        self,
        conversation: list[OPENAI_MESSAGE],
        docs: list[OpenAIDocument],
    ) -> str:
        """Run the multi-agent QA pipeline."""

        conversation = copy.deepcopy(conversation)
        assert conversation[-1]['role'] == 'user'

        if self.coref_agent:
            conversation[-1]['content'] = self.coref_agent.resolve(conversation)

        if self.relevance_agent:
            doc_idxs = self.relevance_agent.filter(conversation, docs)
            docs = [doc for i, doc in enumerate(docs) if i in doc_idxs]

        return self.answer_agent.answer(conversation, docs)
