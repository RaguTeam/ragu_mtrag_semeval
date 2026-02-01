from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam 

from src.agents.answer import gemini_format_doc
from src.client.openai_compatible import LLM
from src.shared.conversations import pretty_print_conversation
from src.shared.schemas import OPENAI_MESSAGE, OpenAIDocument


class Analyser(ABC):
    @abstractmethod
    def analyse(
        self,
        conversation: list[OPENAI_MESSAGE],
        answer: str,
        docs: list[OpenAIDocument],
    ) -> tuple[str, str, str, str]:
        """Analyses the conversation, the answer and the docs via LLM.
        Returns a tuple (title, prompt, model_name, answer).
        """
        ...


DEFAULT_VSEGPT_SYSTEM_PROMPT = """\
You are a large language model.
Carefully heed the user's instructions.
Respond using Markdown."""


BEHAVIOUR_PROMPT = """\
I need to clone the behaviour of a specific LLM-agent in RAG scenario. \
I provide the dialog between the user and the agent, where each turn the \
agent received the question and the documents. Documents are shown only for \
the last question. You need to analyse the last question, documents and \
agent answer.

A dialogue ending with the user's question:

{conversation}

The agent's answer: {answer}

Documents that the agent uses at the current turn:

{documents}

I remind the agent's answer: {answer}

Write a list of the agent's behavioral characteristics. How does it \
handle wide, open, opinionated questions (cites the document? uses \
the world knowledge? reformulates the document?). Your list will help \
to build a similar agent.

Output the list ONLY (from 1 to 5 list items, each of 1-2 sentences).
"""


class AgentBehaviourAnalyser(Analyser):
    def __init__(self, llm: LLM):
        self.llm = llm

    def analyse(
        self,
        conversation: list[OPENAI_MESSAGE],
        answer: str,
        docs: list[OpenAIDocument],
    ) -> tuple[str, str, str, str]:
        prompt = BEHAVIOUR_PROMPT.format(
            conversation=pretty_print_conversation(conversation),
            answer=answer,
            documents='\n\n'.join(gemini_format_doc(doc.text) for doc in docs),
        )
        openai_messages: list[OPENAI_MESSAGE] = [
            ChatCompletionSystemMessageParam(role='system', content=DEFAULT_VSEGPT_SYSTEM_PROMPT),
            ChatCompletionUserMessageParam(role='user', content=prompt),
        ]
        response = self.llm.generate(openai_messages, think=True)
        return (
            'agent_behaviour',
            BEHAVIOUR_PROMPT,
            self.llm.model,
            response,
        )