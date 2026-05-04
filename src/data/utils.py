import copy
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
import re
from typing import Any, Literal, Self

from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from src import MTRAG_DATA
from src.shared.conversations import pretty_print_conversation


@dataclass
class MTRAG_Passage:
    id: str
    """A unique ID, typically formatted as {doc_id}-{start_char}-{end_char}."""

    _id: str
    """Equals .id for all passages (checked)."""

    url: str
    """A URL, sometimes a regexp (how?)."""

    title: str
    """Looks like a webpage title (?)."""

    text: str
    """Composed of .title + linebreak + page text."""

    def __post_init__(self):
        """Rules inferred from the MTRAG data."""
        assert self._id == self.id
        assert self.text.startswith(self.title + "\n")


@dataclass
class MTRAG_Document:
    document_id: str
    """A unique ID."""

    text: str
    """Composed of .title + linebreak + page text. May have some weird
    prefix symbols such as ï»¿\n\n.
    """

    title: str | None = None
    """Looks like a webpage title. For "cloud" corpus is ommited for all
    documents.
    """

    url: str | None = None
    """A URL, sometimes a regexp. May be omitted in some corpora."""

    domain: str | None = None
    """For "cloud" corpus may be N/A, /app-configuration or
    AnalyticsEngine.
    """

    passages: list[MTRAG_Passage] = field(default_factory=list[MTRAG_Passage])
    """Passages from this document."""

    @classmethod
    def from_json(cls, json_string: str) -> Self:
        """Performs field preprocessing to unify formats across corpora.
        Does not fill .passages
        """
        data = json.loads(json_string)
        if "_id" in data:
            # field varies: sometimes _id, sometimes document_id
            assert "id" not in data
            data["document_id"] = data.pop("_id")
        if "metadata" in data:
            # always seems to be empty, so skip it
            assert data["metadata"] == {}
            del data["metadata"]
        return cls(**data)


@dataclass
class RetrieverCollection:
    name: str  # such as 'mt-rag-clapnq-elser-512-100-20240503'
    size: str  # such as '183408'


@dataclass
class RetrieverParameters:
    max_count: int  # such as 3
    max_utterances: int  # such as -1
    query_syntax: str  # looks like a json
    project: str  # looks like a json with text, title, url
    collections: dict[str, Any] | None = None  # such as {'regex': 'mt-rag*'}


@dataclass
class RetrieverConfig:
    collection: RetrieverCollection
    parameters: RetrieverParameters


@dataclass
class GeneratorPrompt:
    template: str  # such as '[INST]\n${CONTEXT}\n${SYSTEM_INST}\n${INPUT}\n[/INST]\nanswer:'
    input: str  # such as '${SPEAKER}: ${TEXT}\n'
    context: str  # such as '[DOCUMENT]\n${TEXT}\n[END]\n'
    system_instruction: str  # such as 'You are an AI Assistant, tasked with...'


@dataclass
class GeneratorParameters:
    min_new_tokens: int  # such as 1
    max_new_tokens: int  # such as 512
    repetition_penalty: float  # such ass 1.05
    stop_sequences: list[str]  # such as ['<|endoftext|>']


@dataclass
class GeneratorConfig:
    id: str  # such as 'mistralai/mixtral-8x7b-instruct-v01'
    name: str  # such as 'mixtral-8x7b-instruct-v01'
    prompt: GeneratorPrompt
    parameters: GeneratorParameters


@dataclass
class StatusHistory:
    author: str  # id
    status: Literal["accepted", "acceptted", "edited", "created", "rejected", "rejectted"]  # wtf
    timestamp: int


@dataclass
class RelevanceJudgement:
    annotator: str  # id
    value: Literal["yes", "no"]
    timestamp: int


@dataclass
class UserMessageEnrichments:
    multi_turn: Literal["Clarification", "Follow-up", "N/A"]
    answerability: Literal["ANSWERABLE", "CONVERSATIONAL", "PARTIAL", "UNANSWERABLE"]
    question_type: list[
        Literal[
            "Comparative",
            "Composite",
            "Explanation",
            "Factoid",
            "How-To",
            "Keyword",
            "Non-Question",
            "Opinion",
            "Summarization",
            "Troubleshooting",
        ]
    ]


@dataclass
class UserMessage:
    speaker: Literal["user"]
    text: str
    timestamp: int
    enrichments: UserMessageEnrichments
    alternatives: dict[str, Any] | None = None  # a result of co-reference resolution? occurs very rarely (4 times only)


@dataclass
class AgentMessageContext:
    document_id: int  # actually a passage ID
    text: str  # is this always the same as passsage.text for passage ID?
    score: float  # what is this? a RAG score?
    query: dict[str, Any]  # a query json in a format similar to RetrieverParameters.query_syntax
    feedback: list[RelevanceJudgement] | None = None  # possible annotator feedbacks if the passage is relevant or not
    title: str | None = None  # is this always the same as passsage.title for passage ID?
    url: str | None = None  # possibly the document URL


@dataclass
class AgentMessage:
    speaker: Literal["agent"]
    text: str
    timestamp: int
    contexts: list[AgentMessageContext]  # retrieved docs
    original_text: str | None = None  # possibly a text before correction by annotator


@dataclass
class Conversation:
    messages: list[UserMessage | AgentMessage]
    """The messages in conversation.

    - `.messages[::2]` is always a list of UserMessage
    - `.messages[1::2]` is always a list of AgentMessage
    """

    author: str
    """Possibly an ID of the human annotator."""

    editor: str
    """Possibly an ID of the human annotator editor (who is?)."""

    reviewer: str
    """Possibly an ID of the human annotator on the reviewing stage."""

    domain: str
    """Some ID-like field (not clear)."""

    retriever: RetrieverConfig
    """Possibly a config for retriever when generating a dialog."""

    generator: GeneratorConfig
    """Possibly a config for generator when generating a dialog."""

    status: Literal["accepted"]
    """All conversations have status 'accepted'."""

    status_history: list[StatusHistory]
    """The history of changing status during annotation process."""


@dataclass
class GenerationTaskMessage:
    # similar to UserMessage and AgentMessage, but in GenerationTask format
    speaker: Literal["user", "agent"]
    text: str
    author_id: str
    created_at: int  # timestamp

    def to_openai(self) -> ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam:
        match self.speaker:
            case 'agent':
                return {'role': 'assistant', 'content': self.text}
            case 'user':
                return {'role': 'user', 'content': self.text}
            case _:
                raise ValueError('bad speaker field')


@dataclass
class AgentMessageContextForGenerationTask:
    # similar to AgentMessageContext, but in GenerationTask format
    document_id: int  # actually a passage ID
    text: str  # is this always the same as passsage.text for passage ID?
    score: float | None = None  # what is this? a RAG score?
    source: str | None = None  # some unclear value
    query: dict[str, Any] | None = (
        None  # a query json in a format similar to RetrieverParameters.query_syntax (if RAG-retrieved)
    )
    feedback: list[RelevanceJudgement] | None = None  # possible annotator feedbacks if the passage is relevant or not
    title: str | None = None  # is this always the same as passsage.title for passage ID?
    url: str | None = None  # possibly the document URL
    reference: bool = False  # is reference, or RAG-retrieved?


@dataclass
class GenerationTaskPrediction:
    text: str


@dataclass
class GenerationTaskMetrics:
    Recall: float | None = None
    RougeL_stemFalse: float | None = None
    BertscoreP: float | None = None
    BertscoreR: float | None = None
    Length: float | None = None
    RB_agg: float | None = None
    idk_eval: float | None = None
    RL_F: float | None = None
    RB_llm: float | None = None
    RL_F_idk: float | None = None
    RB_llm_idk: float | None = None
    RB_agg_idk: float | None = None
    BertKPrec: list[float] | None = None
    Extractiveness_RougeL: list[float] | None = None

    def __post_init__(self):
        # converts from 'idk_eval': [0.0] to 'idk_eval': 0.0
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, list) and field.name not in ('BertKPrec', 'Extractiveness_RougeL'):
                setattr(self, field.name, value[0])



@dataclass
class GenerationTask:
    conversation_id: str
    """The ID of the whole conversation of multiple turns."""

    task_id: str
    """The ID of the a turn in a conversation,
    formatted as {conversation_id}<::>{turn}.
    """

    task_type: Literal["rag"]
    """This field always equals "rag" for all the files
    human/generation_tasks/*.jsonl (is this a mistake?)
    """

    turn: int
    """A turn (the index of answer+response pair in the conversation)."""

    dataset: Literal["MT-RAG Authors (Internal)"]

    collection: Literal[
        "mt-rag-govt-elser-512-100-20240611",
        "mt-rag-ibmcloud-elser-512-100-20240502",
        "mt-rag-fiqa-beir-elser-512-100-20240501",
        "mt-rag-clapnq-elser-512-100-20240503",
        "fiqa",
        "ibmcloud",
        "clapnq",
        "govt"
    ]

    answerability: Literal["ANSWERABLE", "CONVERSATIONAL", "PARTIAL", "UNANSWERABLE"]

    multi_turn: Literal["Clarification", "Follow-up", "N/A"]

    question_type: Literal[
        "Comparative",
        "Composite",
        "Explanation",
        "Factoid",
        "How-To",
        "Keyword",
        "Non-Question",
        "Opinion",
        "Summarization",
        "Troubleshooting",
    ]

    input: list[GenerationTaskMessage]
    """Dialog history, .speaker == 'user' for even indices and 'agent'
    for odd indices.
    """

    contexts: list[AgentMessageContextForGenerationTask]
    """Context used to generate .target message."""

    target: GenerationTaskMessage | None = None
    """A message to evaluate."""

    rewritten_query: str | None = None
    """Unclear field, looke like paraphrasing, not a co-reference resolution."""

    standalone_type: Literal["Standalone", "Non-standalone"] | None = None
    """Non-standalone for 10 tasks, Standalone for 8 tasks and None for
    the remaining 2102 tasks.
    """

    validity: Literal["Adversarial"] | None = None
    """Adversarial for 3 tasks, None for all the remaining tasks."""

    ambiguity_type: list[ # type: ignore
        Literal[  # type: ignore
            "Needs Context from Chatbot Response",
            "Coreference",
            "Ellipsis",
        ]
    ] = field(default_factory=list)
    """Needs Context from Chatbot Response 3 times, Coreference 3 times,
    Ellipsis 2 times, empty for the remaining tasks.
    """

    n_references: int | None = None
    """Unclear field with the following statistics:
    Counter({None: 1278, 2: 266, 3: 243, 1: 105, 4: 96, 0: 65, 5: 36,
         6: 21, 7: 8, 9: 1, 8: 1})
    """

    predictions: list[GenerationTaskPrediction] | None = None

    metrics: GenerationTaskMetrics | None = None


@dataclass
class UnreferenceGenetionTask:
    task_id: str
    input: list[GenerationTaskMessage]
    contexts: list[AgentMessageContextForGenerationTask] = field(default_factory=list)
    prediction: str | None = None   


##### loading utils #####


def _agent_message_context_from_json(json_data: dict[str, Any]) -> AgentMessageContext:
    json_data = copy.deepcopy(json_data)
    if "feedback" in json_data:
        assert set(json_data["feedback"].keys()) == {"relevant"}  # type: ignore
        json_data["feedback"] = [
            RelevanceJudgement(annotator=annotator, **data)  # type: ignore
            for annotator, data in json_data["feedback"]["relevant"].items()  # type: ignore
        ]
    return AgentMessageContext(**json_data)  # type: ignore


def _agent_message_context_gen_from_json(json_data: dict[str, Any]) -> AgentMessageContextForGenerationTask:
    json_data = copy.deepcopy(json_data)
    if "feedback" in json_data:
        assert set(json_data["feedback"].keys()) == {"relevant"}  # type: ignore
        json_data["feedback"] = [
            RelevanceJudgement(annotator=annotator, **data)  # type: ignore
            for annotator, data in json_data["feedback"]["relevant"].items()  # type: ignore
        ]
    return AgentMessageContextForGenerationTask(**json_data)  # type: ignore


def _generation_task_message_from_json(json_data: dict[str, Any]) -> GenerationTaskMessage:
    return GenerationTaskMessage(
        speaker=json_data["speaker"],
        text=json_data["text"],
        author_id=json_data["metadata"]["author_id"],
        created_at=json_data["metadata"]["created_at"],
    )


def conversation_from_json(json_data: dict[str, Any]) -> Conversation:
    json_data = copy.deepcopy(json_data)

    json_data["retriever"]["collection"] = RetrieverCollection(**json_data["retriever"]["collection"])
    json_data["retriever"]["parameters"] = RetrieverParameters(**json_data["retriever"]["parameters"])
    json_data["retriever"] = RetrieverConfig(**json_data["retriever"])

    json_data["generator"]["prompt"] = GeneratorPrompt(**json_data["generator"]["prompt"])
    json_data["generator"]["parameters"] = GeneratorParameters(**json_data["generator"]["parameters"])
    json_data["generator"] = GeneratorConfig(**json_data["generator"])

    json_data["status_history"] = [StatusHistory(**x) for x in json_data["status_history"]]

    for message_idx, message in list(enumerate(json_data["messages"])):  # type: ignore
        if message_idx % 2 == 0:
            assert message["speaker"] == "user"
            message["enrichments"] = UserMessageEnrichments(
                multi_turn=message["enrichments"]["Multi-Turn"],  # type: ignore
                answerability=message["enrichments"]["Answerability"],  # type: ignore
                question_type=message["enrichments"]["Question Type"],  # type: ignore
            )
            json_data["messages"][message_idx] = UserMessage(**message)  # type: ignore
        else:
            assert message["speaker"] == "agent"
            message["contexts"] = [_agent_message_context_from_json(data) for data in message["contexts"]]  # type: ignore
            json_data["messages"][message_idx] = AgentMessage(**message)  # type: ignore

    return Conversation(**json_data)


def generation_task_from_json(json_data: dict[str, Any]) -> GenerationTask:
    json_data = copy.deepcopy(json_data)
    targets = json_data.pop("targets", None)
    if targets:        
        assert len(targets) == 1
        json_data["target"] = _generation_task_message_from_json(targets[0])

    json_data["input"] = [_generation_task_message_from_json(x) for x in json_data["input"]]
    json_data["contexts"] = [_agent_message_context_gen_from_json(x) for x in json_data["contexts"]]
    json_data["question_type"] = json_data.pop("Question Type", None)
    json_data["multi_turn"] = json_data.pop("Multi-Turn", None)
    json_data["answerability"] = json_data.pop("Answerability", None)
    json_data["collection"] = json_data.pop("Collection")
    if "Standalone Type" in json_data:
        value = json_data.pop("Standalone Type")
        if value is not None:
            assert len(value) == 1
            json_data["standalone_type"] = value[0]
        else:
            json_data["standalone_type"] = value
    if "Validity" in json_data:
        assert len(json_data["Validity"]) == 1
        json_data["validity"] = json_data.pop("Validity")[0]
    json_data["ambiguity_type"] = json_data.pop("Ambiguity Type", [])
    json_data["n_references"] = json_data.pop("No. References", None)
    if "predictions" in json_data:
        json_data["predictions"] = [GenerationTaskPrediction(**x) for x in json_data["predictions"]]
    if "metrics" in json_data:
        json_data["metrics"] = GenerationTaskMetrics(**json_data["metrics"])
    return GenerationTask(**json_data)


##### loading functions #####


def load_corpus_document_level(path: str | Path) -> list[MTRAG_Document]:
    """Loads MT-RAG corpus in document-level format. Examples:

    ```
    load_corpus_document_level('mt-rag-benchmark/corpora/document_level/clapnq.jsonl')
    load_corpus_document_level('mt-rag-benchmark/corpora/document_level/cloud.jsonl')
    load_corpus_document_level('mt-rag-benchmark/corpora/document_level/fiqa.jsonl')
    load_corpus_document_level('mt-rag-benchmark/corpora/document_level/gotv.jsonl')
    ```
    """
    return [MTRAG_Document.from_json(line) for line in Path(path).read_text().strip().split("\n")]


def load_corpus_passage_level(path: str | Path) -> list[MTRAG_Passage]:
    """Loads MT-RAG corpus in passage-level format. Examples:

    ```
    load_corpus_passage_level('mt-rag-benchmark/corpora/passage_level/clapnq.jsonl')
    load_corpus_passage_level('mt-rag-benchmark/corpora/passage_level/cloud.jsonl')
    load_corpus_passage_level('mt-rag-benchmark/corpora/passage_level/fiqa.jsonl')
    load_corpus_passage_level('mt-rag-benchmark/corpora/passage_level/gotv.jsonl')
    ```
    """
    return [MTRAG_Passage(**json.loads(line)) for line in Path(path).read_text().strip().split("\n")]


def load_conversations(
    path: str | Path = MTRAG_DATA / "human/conversations/conversations.json",
) -> list[Conversation]:
    """Loads MT-RAG conversations. Will use MTRAG_DATA if set (see readme)."""
    return [conversation_from_json(sample) for sample in json.loads(Path(path).read_text())]


def load_generation_tasks(
    path: str | Path = MTRAG_DATA / "human/generation_tasks/reference.json",
) -> list[GenerationTask]:
    """Loads MT-RAG generation tasks. Will use reference.json from MTRAG_DATA if set (see readme).

    Examples:
    ```
    load_conversations('mt-rag-benchmark/human/generation_tasks/RAG.jsonl')
    load_conversations('mt-rag-benchmark/human/generation_tasks/reference.jsonl')
    load_conversations('mt-rag-benchmark/human/generation_tasks/reference+RAG.jsonl')
    ```

    """
    return [generation_task_from_json(json.loads(line)) for line in Path(path).read_text().strip().split("\n")]



##### GenerationTask to GenerationTaskAnalysis #####

@dataclass
class GenerationTaskAnalysis:
    """A GenerationTask formatted for displaying and manual analysis."""

    task_id: str
    """A unique ID for the task."""

    dialog: str
    """A previous conversation in human-readable way."""

    documents: list[str]
    """Documents in a human-readable way, without empty lines."""

    reference: str
    """The reference answer."""

    answerability: Literal["ANSWERABLE", "CONVERSATIONAL", "PARTIAL", "UNANSWERABLE"]

    multi_turn: Literal["Clarification", "Follow-up", "N/A"]

    question_type: Literal[
        "Comparative",
        "Composite",
        "Explanation",
        "Factoid",
        "How-To",
        "Keyword",
        "Non-Question",
        "Opinion",
        "Summarization",
        "Troubleshooting",
    ]

    prediction: str | None = None
    """The predicted answer."""

    metrics: GenerationTaskMetrics | None = None

    analysis: list[tuple[str, str, str, str]] | None = None
    """Optional field with additonal questions to LLM as judge. Each element
    contains (title, prompt, model_name, answer), where prompt and answer can be
    multi-line, and title is a short description of the prompt.
    """

    @classmethod
    def from_task(cls, task: GenerationTask) -> Self:
        doc_texts_formatted = [
            re.sub(r'\n[\n\s]+', '\n', doc.text.replace('\r', '').strip())
            for doc in task.contexts
        ]
        return cls(
            task_id=task.task_id,
            dialog=pretty_print_conversation([x.to_openai() for x in task.input]),
            documents=doc_texts_formatted,
            reference=task.target.text,
            prediction=(
                task.predictions[0].text
                if task.predictions is not None
                else None
            ),
            metrics=task.metrics,
            answerability=task.answerability,
            multi_turn=task.multi_turn,
            question_type=task.question_type,
        )
