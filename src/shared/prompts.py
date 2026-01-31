"""Prompts used for llm generation."""

from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam

from src.shared.schemas import OPENAI_MESSAGE, ExtendedConversation, OpenAIDocument


COREFERENCE_RESOLUTION = (
    "You are a coreference resolution agent.\n"
    "Today's date is {date}.\n\n"
    "Your task:\n"
    "Replace relative temporal expressions (e.g. "
    "'last year', 'this year', 'recently', 'now') with "
    "explicit calendar dates or years relative to today's date.\n"
    "Deduse mentions like 'he', 'she', 'they', 'the company', etc. from the context of the conversation.\n\n"
    "Rewrite the last user message with all coreferences and temporal expressions resolved.\n"
    "Do not answer the question, just rewrite it.\n"
)


RELEVANCE_FILTERING = (
    "Determine whether the document is relevant to the user question. "
    "Today's date is {date}.\n\n"
    "Answer ONLY 'yes' or 'no'."
)


ANSWER_QUESTION = (
    "Today's date is {date}.\n"
    "You are a question-answering assistant.\n"
    "Use only the provided documents.\n"
    "If the information is insufficient, explicitly say so."
)


DOCUMENT_TEMPLATE = "\n\n[Document]\n{content}"


COREFERENCE_EXAMPLE: list[OPENAI_MESSAGE] = [
    ChatCompletionUserMessageParam(
        role="user",
        content=str([
            {"role": "user", "content": "Who is Albert Einstein?"},
            {"role": "assistant", "content": "Albert Einstein was a German-born theoretical physicist best known for developin"
            "g the theory of relativity."},
            {"role": "user", "content": "When was he born?"},
        ]),
    ),
    ChatCompletionAssistantMessageParam(role="assistant", content="When was Albert Einstein born?"),
    ChatCompletionUserMessageParam(
        role="user",
        content=str([
            {"role": "user", "content": "What year was 'Attention is all you need' paper released?"},
            {"role": "assistant", "content": "\"Attention Is All You Need\" is a  research paper in machine learning authored by"
            " eight scientists working at Google. It was proposed in the year 2017."},
            {"role": "user", "content": "What was last year's most cited work?"},
        ]),
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
