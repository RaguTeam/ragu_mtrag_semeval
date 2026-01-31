"""Schemas for conversations and messages."""

from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)


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