"""Conversations list."""

from __future__ import annotations
from typing import TYPE_CHECKING
import textwrap

from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from src.shared.schemas import OPENAI_MESSAGE, OpenAIDocument
if TYPE_CHECKING:
    from src.data.utils import GenerationTask


def task_to_conversation(
    task: GenerationTask, doc_header: bool = True,
) -> tuple[list[OPENAI_MESSAGE], list[OpenAIDocument]]:

    msgs: list[OPENAI_MESSAGE] = []
    for m in task.input:
        if m.speaker == "user":
            msgs.append(ChatCompletionUserMessageParam(role='user', content=m.text))
        elif m.speaker == "agent":
            msgs.append(ChatCompletionAssistantMessageParam(role='assistant', content=m.text))
        else:
            raise ValueError(f"Unknown speaker {m.speaker!r} in task_id={task.task_id}")

    docs = [OpenAIDocument(text=x.text) for x in task.contexts]

    return msgs, docs


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
