"""Conversations list."""

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
