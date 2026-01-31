"""Conversations list."""

from __future__ import annotations
from typing import TYPE_CHECKING, cast
import textwrap

from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from src.shared.prompts import DOCUMENT_TEMPLATE
from src.shared.schemas import OPENAI_MESSAGE, ExtendedConversation, OpenAIDocument
if TYPE_CHECKING:
    from src.data.utils import GenerationTask


def extended_to_openai(
    messages: list[OPENAI_MESSAGE | OpenAIDocument],
) -> list[OPENAI_MESSAGE]:
    """Convert ExtendedConversation messages to OpenAIConversation format.

    Args:
        messages: List of ExtendedMessage instances.

    Returns:
        list[OpenAIMessage]: Converted list of OpenAIMessage instances.

    """
    openai_messages: list[OPENAI_MESSAGE] = []

    pending_user: OPENAI_MESSAGE | None = None


    for msg in messages:
        is_doc = isinstance(msg, OpenAIDocument)

        if not is_doc and msg['role'] == "user":
            if pending_user is not None:
                openai_messages.append(pending_user)

            pending_user = ChatCompletionUserMessageParam(
                role="user",
                content=msg['content'],
            )

        elif is_doc:
            if pending_user is None:
                raise ValueError(
                    "Document message encountered without preceding user message",
                )

            pending_user = ChatCompletionUserMessageParam(
                role="user",
                content=(
                    cast(str, pending_user['content'])
                    + DOCUMENT_TEMPLATE.format(content=msg.text)
                ),
            )

        elif not is_doc and msg['role'] == "assistant":
            if pending_user is not None:
                openai_messages.append(pending_user)
                pending_user = None

            openai_messages.append(msg.copy())

        elif not is_doc and msg['role'] == "system":
            openai_messages.append(msg.copy())

        else:
            raise ValueError(f"Unsupported role: {msg['role']}")

    if pending_user is not None:
        openai_messages.append(pending_user)

    return openai_messages


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
