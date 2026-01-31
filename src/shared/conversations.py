"""Conversations list."""

from __future__ import annotations
from typing import TYPE_CHECKING
import textwrap

from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from src.shared.schemas import OPENAI_MESSAGE, ExtendedConversation, OpenAIDocument
if TYPE_CHECKING:
    from src.data.utils import GenerationTask


COREFERENCE_EXAMPLE: list[OPENAI_MESSAGE] = [
    ChatCompletionUserMessageParam(
        role="user",
        content="""[
    {"role": "user", "content": "Who is Albert Einstein?"},
    {"role": "assistant", "content": "Albert Einstein was a German-born theoretical physicist best known for developin"
    "g the theory of relativity."},
    {"role": "user", "content": "When was he born?"},
    """,
    ),
    ChatCompletionAssistantMessageParam(role="assistant", content="When was Albert Einstein born?"),
    ChatCompletionUserMessageParam(
        role="user",
        content="""[
    {"role": "user", "content": "What year was 'Attention is all you need' paper released?"},
    {"role": "assistant", "content": ""Attention Is All You Need" is a  research paper in machine learning authored by"
    " eight scientists working at Google. It was proposed in the year 2017."},
    {"role": "user", "content": "What was last year's most cited work?"},
    """,
    ),
    ChatCompletionAssistantMessageParam(role="assistant", content="What was year 2025 most cited work?"),
]

EXAMPLE_CONVERSATION = ExtendedConversation([
    ChatCompletionUserMessageParam(role="user", content="Привет!"),
    ChatCompletionAssistantMessageParam(role="assistant", content="Привет, как я могу помочь?"),
    ChatCompletionUserMessageParam(role="user", content="Что произошло в прошлом году?"),
    OpenAIDocument("Компания отчиталась о рекордной прибыли в 2023 году."),
    OpenAIDocument("В прошлом году был принят новый закон."),
    ChatCompletionAssistantMessageParam(role="assistant", content="В прошлом году был принят закон, хоть и не сказано, какой"),
    ChatCompletionUserMessageParam(role="user", content="Расскажи про Чарли Чаплина"),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content="Чарли Чаплин был одним из самых творческих и влиятельных людей в эпоху немого кино",
    ),
    ChatCompletionUserMessageParam(role="user", content="Когда он родился?"),
    OpenAIDocument("Чарли Чаплин родился 16 апреля 1889 года в Лондоне."),
])




def task_to_conversation(task: GenerationTask, doc_header: bool = True) -> ExtendedConversation:
    msgs: list[OPENAI_MESSAGE | OpenAIDocument] = []

    for m in task.input:
        if m.speaker == "user":
            msgs.append(ChatCompletionUserMessageParam(role='user', content=m.text))
        elif m.speaker == "agent":
            msgs.append(ChatCompletionAssistantMessageParam(role='assistant', content=m.text))
        else:
            raise ValueError(f"Unknown speaker {m.speaker!r} in task_id={task.task_id}")

    for i, ctx in enumerate(task.contexts, start=1):
        doc_id = getattr(ctx, "document_id", None)
        title = getattr(ctx, "title", "") or ""
        text = getattr(ctx, "text", "") or ""

        doc_id_str = "" if doc_id is None else str(doc_id)

        if doc_header:
            header_bits: list[str] = []
            if title:
                header_bits.append(f"title={title}")
            if doc_id_str:
                header_bits.append(f"id={doc_id_str}")
            header = f"Document {i}" + (f" ({', '.join(header_bits)})" if header_bits else "")
            content = f"{header}:\n{text}"
        else:
            content = text

        msgs.append(OpenAIDocument(content))

    return ExtendedConversation(msgs)


def pretty_print_turn_one_row(turn: OPENAI_MESSAGE) -> str:
    assert not 'refusal' in turn
    role = turn['role']
    match content := turn.get('content', None):
        case None:
            content_str = '<no content in response>'
        case str():
            content_str = content
        case _:
            raise ValueError('messages as list not supported')
    return f'{role.capitalize()}: {content_str.replace('\n', '\\n')}'
    


def pretty_print_conversation(conv: list[OPENAI_MESSAGE]) -> str:
    return '\n'.join(pretty_print_turn_one_row(turn) for turn in conv)



def pretty_print_turn_using_tab(turn: OPENAI_MESSAGE) -> str:
    assert not 'refusal' in turn
    role = turn['role']
    match content := turn.get('content', None):
        case None:
            content_str = '<no content in response>'
        case str():
            content_str = content
        case _:
            raise ValueError('messages as list not supported')
    return f'{role}\n{textwrap.indent(content_str, '\t')}'
    


def pretty_print_conversation_using_tab(conv: list[OPENAI_MESSAGE]) -> str:
    return '\n\n'.join(pretty_print_turn_using_tab(turn) for turn in conv)