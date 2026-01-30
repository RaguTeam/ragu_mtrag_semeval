"""Conversations list."""

import textwrap
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from src.shared.schemas import ExtendedConversation, ExtendedMessage, OpenAIMessage

COREFERENCE_EXAMPLE = [
    OpenAIMessage(
        "user",
        """[
    {"role": "user", "content": "Who is Albert Einstein?"},
    {"role": "assistant", "content": "Albert Einstein was a German-born theoretical physicist best known for developin"
    "g the theory of relativity."},
    {"role": "user", "content": "When was he born?"},
    """,
    ),
    OpenAIMessage("assistant", "When was Albert Einstein born?"),
    OpenAIMessage(
        "user",
        """[
    {"role": "user", "content": "What year was 'Attention is all you need' paper released?"},
    {"role": "assistant", "content": ""Attention Is All You Need" is a  research paper in machine learning authored by"
    " eight scientists working at Google. It was proposed in the year 2017."},
    {"role": "user", "content": "What was last year's most cited work?"},
    """,
    ),
    OpenAIMessage("assistant", "What was year 2025 most cited work?"),
]

EXAMPLE_CONVERSATION = ExtendedConversation(
    [
        ExtendedMessage("user", "Привет!"),
        ExtendedMessage("assistant", "Привет, как я могу помочь?"),
        ExtendedMessage("user", "Что произошло в прошлом году?"),
        ExtendedMessage("document", "Компания отчиталась о рекордной прибыли в 2023 году."),
        ExtendedMessage("document", "В прошлом году был принят новый закон."),
        ExtendedMessage("assistant", "В прошлом году был принят закон, хоть и не сказано, какой"),
        ExtendedMessage("user", "Расскажи про Чарли Чаплина"),
        ExtendedMessage(
            "assistant",
            "Чарли Чаплин был одним из самых творческих и влиятельных людей в эпоху немого кино",
        ),
        ExtendedMessage("user", "Когда он родился?"),
        ExtendedMessage("document", "Чарли Чаплин родился 16 апреля 1889 года в Лондоне."),
    ],
)


OPENAI_MESSAGE = (
    # does not support ChatCompletionToolMessageParam or ChatCompletionFunctionMessageParam yet
    ChatCompletionSystemMessageParam
    | ChatCompletionDeveloperMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionAssistantMessageParam
)


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