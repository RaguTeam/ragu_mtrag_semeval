"""Schemas for conversations and messages."""

from dataclasses import dataclass
from typing import cast

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from src.shared.prompts import DOCUMENT_TEMPLATE


OPENAI_MESSAGE = (
    ChatCompletionSystemMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionAssistantMessageParam
    # | ChatCompletionDeveloperMessageParam
    # | ChatCompletionToolMessageParam
    # | ChatCompletionFunctionMessageParam
)


@dataclass(frozen=True)
class OpenAIDocument:
    text: str


@dataclass
class ExtendedConversation:
    """Extended conversation with document messages."""

    messages: list[OPENAI_MESSAGE | OpenAIDocument]


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
