"""Schemas for conversations and messages."""

from dataclasses import dataclass
from typing import Literal

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
