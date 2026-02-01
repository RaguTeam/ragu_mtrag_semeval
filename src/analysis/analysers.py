from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam 

from src.agents.answer import gemini_format_doc
from src.client.openai_compatible import LLM
from src.data.utils import GenerationTaskMetrics
from src.shared.conversations import pretty_print_conversation
from src.shared.schemas import OPENAI_MESSAGE, OpenAIDocument


class Analyser(ABC):
    @abstractmethod
    def analyse(
        self,
        conversation: list[OPENAI_MESSAGE],
        reference_answer: str,
        docs: list[OpenAIDocument],
        predicted_answer: str | None = None,
        metrics: GenerationTaskMetrics | None = None,
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

The agent's answer: {reference_answer}

Documents that the agent uses at the current turn:

{documents}

I remind the agent's answer: {reference_answer}

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
        reference_answer: str,
        docs: list[OpenAIDocument],
        predicted_answer: str | None = None,
        metrics: GenerationTaskMetrics | None = None,
    ) -> tuple[str, str, str, str]:
        prompt = BEHAVIOUR_PROMPT.format(
            conversation=pretty_print_conversation(conversation),
            reference_answer=reference_answer,
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


SELF_IMPROVEMENT_PROMPT = """\
I need to maximize evaluation metric on a specific multi-turn RAG benchmark.

I provide your last dialog with uset that consists of:
- The whole conversation
- The documents supplemented for the last user question
- Your answer

Then I provide the reference (ground truth) answer and the evaluation metrics
and ask you to analyze it and suggest improvements to the system prompt.

A dialogue ending with the user's question:

{conversation}

Your answer: "{predicted_answer}"

Documents provided at the last turn:

{documents}

I remind your answer: "{predicted_answer}"

The reference answer was: "{reference_answer}"

Score is calculated as a harmonic mean of 3 scores:

- RB_alg (reference-based algorithmic): the harmonic mean of \
Bert-Recall (approximates completeness), Bert-K-Precision (compares \
model response to the passages) and Rouge-L (approximates faithfulness \
and completenes). Rouge-L and Bert-Rec are calculated using the \
response and the reference answer, while Bert-K-Prec uses the response \
and the passages).
- RB_llm (reference-based LLM): an LLM judge inspired by RAD-Bench. \
We compare the response and the reference answer, supplementing the prompt \
with the passages and anchor the evaluation on the metrics of faithfulness, \
appropriateness, and completeness. We take a median of several LLM judges.
- RL_F: the reference-less faithfulness LLM judge from RAGAS.

Your scores on this sample:

- RB_alg (algorithmic): {RB_agg}
    - Bert-Recall: {Recall}
    - Bert-K-Precision: {BertKPrec}
    - Rouge-L: {RougeL_stemFalse}
- RB_llm (LLM-as-judge): {RB_llm_idk}
- RL_F (reference-less faithfulness): {RL_F_idk}

Now analyze this. If the reference answer is wrong, suggest no improvements. \
Otherwise suggest the improvements to the **system prompt**: how to \
modify it to make metrics better? Do NOT rewrite the full system prompt,
Suggest modifications only as instructions for human to modify the prompt.

Output ONLY a bullet list with suggestions modifications, up to 2 items, \
each of 1-2 sentences in free form. If no suggestions, just answer "no suggestions".
"""


class SelfImprovementAnalyser(Analyser):
    def __init__(self, llm: LLM):
        self.llm = llm

    def analyse(
        self,
        conversation: list[OPENAI_MESSAGE],
        reference_answer: str,
        docs: list[OpenAIDocument],
        predicted_answer: str | None = None,
        metrics: GenerationTaskMetrics | None = None,
    ) -> tuple[str, str, str, str]:
        assert metrics
        prompt = SELF_IMPROVEMENT_PROMPT.format(
            conversation=pretty_print_conversation(conversation),
            reference_answer=reference_answer,
            documents='\n\n'.join(gemini_format_doc(doc.text) for doc in docs),
            predicted_answer=predicted_answer,
            RB_agg=f'{metrics.RB_agg_idk:.2f}',
            Recall=f'{metrics.Recall:.2f}',
            BertKPrec=f'{str(metrics.BertKPrec)}',
            RougeL_stemFalse=f'{metrics.RougeL_stemFalse:.2f}',
            RB_llm_idk=f'{metrics.RB_llm_idk:.2f}',
            RL_F_idk=f'{metrics.RL_F_idk:.2f}',
        )
        openai_messages: list[OPENAI_MESSAGE] = [
            ChatCompletionSystemMessageParam(role='system', content=DEFAULT_VSEGPT_SYSTEM_PROMPT),
            ChatCompletionUserMessageParam(role='user', content=prompt),
        ]
        response = self.llm.generate(openai_messages, think=True)
        return (
            'self_improvement',
            SELF_IMPROVEMENT_PROMPT,
            self.llm.model,
            response,
        )


REFERENCE_VALIDATOR_PROMPT = """\
You are provided with a conversation between user and RAG agent. \
Each turn agent answers based on the provided document fragments. \
The document are provided only for the last question. The agent \
reads these documents and tries to answer.

Analyze the agent's answer on the last question.

Conversation:

{conversation}

Documents that may be relevant to the last question:

{documents}

<end of documents section>

The agent's answer: "{reference_answer}"

Is the answer correct, or controversial, and why? \
Given the provided documents, could there be other or better correct answers? \
Can the answer be improved by using world knowledge beyond the provided documents?

Answer in no more than 1-2 paragraphs or less.
"""


class ReferenceValidatorAnalyser(Analyser):
    def __init__(self, llm: LLM):
        self.llm = llm

    def analyse(
        self,
        conversation: list[OPENAI_MESSAGE],
        reference_answer: str,
        docs: list[OpenAIDocument],
        predicted_answer: str | None = None,
        metrics: GenerationTaskMetrics | None = None,
    ) -> tuple[str, str, str, str]:
        prompt = REFERENCE_VALIDATOR_PROMPT.format(
            conversation=pretty_print_conversation(conversation),
            reference_answer=reference_answer,
            documents='\n\n'.join(gemini_format_doc(doc.text) for doc in docs),
        )
        openai_messages: list[OPENAI_MESSAGE] = [
            ChatCompletionSystemMessageParam(role='system', content=DEFAULT_VSEGPT_SYSTEM_PROMPT),
            ChatCompletionUserMessageParam(role='user', content=prompt),
        ]
        response = self.llm.generate(openai_messages, think=True)
        return (
            'reference_validator',
            REFERENCE_VALIDATOR_PROMPT,
            self.llm.model,
            response,
        )