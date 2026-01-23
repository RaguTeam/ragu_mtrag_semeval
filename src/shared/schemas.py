"""Schemas for conversations and messages."""

from dataclasses import dataclass
from typing import Literal

from src.shared.prompts import DOCUMENT_TEMPLATE

ExtendedRole = Literal["system", "user", "assistant", "document"]


@dataclass(frozen=True)
class ExtendedMessage:
    """Extended message with document role."""

    role: ExtendedRole
    content: str


@dataclass
class ExtendedConversation:
    """Extended conversation with document messages."""

    messages: list[ExtendedMessage]


OpenAIRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class OpenAIMessage:
    """OpenAI-style message."""

    role: OpenAIRole
    content: str


@dataclass
class OpenAIConversation:
    """OpenAI-style conversation."""

    messages: list[OpenAIMessage]

    def to_openai(self) -> list[dict[str, str]]:
        """Convert to list of dicts for OpenAI API.

        Returns:
            list[dict]: List of messages as dicts with 'role' and 'content' keys.

        """
        return [{"role": m.role, "content": m.content} for m in self.messages]


def extended_to_openai(
    messages: list[ExtendedMessage],
) -> list[OpenAIMessage]:
    """Convert ExtendedConversation messages to OpenAIConversation format.

    Args:
        messages: List of ExtendedMessage instances.

    Returns:
        list[OpenAIMessage]: Converted list of OpenAIMessage instances.

    """
    openai_messages: list[OpenAIMessage] = []

    pending_user: OpenAIMessage | None = None

    for msg in messages:
        if msg.role == "user":
            if pending_user is not None:
                openai_messages.append(pending_user)

            pending_user = OpenAIMessage(
                role="user",
                content=msg.content,
            )

        elif msg.role == "document":
            if pending_user is None:
                raise ValueError(
                    "Document message encountered without preceding user message",
                )

            pending_user = OpenAIMessage(
                role="user",
                content=pending_user.content + DOCUMENT_TEMPLATE.format(content=msg.content),
            )

        elif msg.role == "assistant":
            if pending_user is not None:
                openai_messages.append(pending_user)
                pending_user = None

            openai_messages.append(
                OpenAIMessage(role="assistant", content=msg.content),
            )

        elif msg.role == "system":
            openai_messages.append(
                OpenAIMessage(role="system", content=msg.content),
            )

        else:
            raise ValueError(f"Unsupported role: {msg.role}")

    if pending_user is not None:
        openai_messages.append(pending_user)

    return openai_messages
