"""Prompts used for llm generation."""

from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam

from src.shared.schemas import OPENAI_MESSAGE, OpenAIDocument


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


ANSWER_NO_CONTEXT = (
    "You are a retrieval-augmented question-answering assistant.\n\n"
    "IMPORTANT:\n"
    "No documents were provided for this question.\n"
    "You MUST NOT answer using general knowledge.\n\n"
    "Your task:\n"
    "- Politely say that you do not have enough context to answer.\n"
    "- Mention the topic by paraphrasing the user's question.\n"
    "- Do NOT guess, do NOT fabricate facts.\n"
    "- Do NOT say you searched the web or external sources.\n"
    "- Keep the answer short (1–2 sentences).\n"
)


DOCUMENT_TEMPLATE = "\n\n[DOCUMENT]\n{content}\n[END DOCUMENT]"


COREFERENCE_EXAMPLE: list[OPENAI_MESSAGE] = [
    # 1) Pronoun/ellipsis -> explicit subject
    ChatCompletionUserMessageParam(
        role="user",
        content=str([
            {"role": "user", "content": "What is ground water contamination?"},
            {
                "role": "assistant",
                "content": "Groundwater contamination refers to harmful substances getting into groundwater through improper disposal or seepage.",
            },
            {"role": "user", "content": "Can it be clean up?"},
        ]),
    ),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content="Can groundwater contamination be cleaned up?",
    ),

    # 2) "it" refers to the specific plan/product
    ChatCompletionUserMessageParam(
        role="user",
        content=str([
            {"role": "user", "content": "Are the Lite pricing plans available with all account types in IBM Cloud?"},
            {
                "role": "assistant",
                "content": "Yes, Lite pricing plans are available with all account types in IBM Cloud.",
            },
            {"role": "user", "content": "What is it?"},
            {
                "role": "assistant",
                "content": "The Lite plan is a free service plan with usage quotas.",
            },
            {"role": "user", "content": "Do I need to purchase it with other products?"},
        ]),
    ),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content="Do I need to purchase the IBM Cloud Lite plan with other products?",
    ),

    # 3) User assertion with implicit antecedent ("it" / "one") -> make it explicit
    ChatCompletionUserMessageParam(
        role="user",
        content=str([
            {"role": "user", "content": "What is the filing requirement for US-based companies under SEC regulations?"},
            {
                "role": "assistant",
                "content": "There are many different kinds of SEC filings (e.g., 10-K, 10-Q, Form 4) with different purposes.",
            },
            {"role": "user", "content": "Is there one that is more important than the others?"},
            {
                "role": "assistant",
                "content": "Often, the most crucial filings include Form 10-K and Form 10-Q, depending on what you mean by important.",
            },
            {"role": "user", "content": "I believe it is the 13F"},
        ]),
    ),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content="I believe the most important SEC filing is Form 13F.",
    ),

    # 4) "those" / "steps" refers to earlier procedure -> restate explicitly
    ChatCompletionUserMessageParam(
        role="user",
        content=str([
            {"role": "user", "content": "What are intents?"},
            {
                "role": "assistant",
                "content": "Intents are the purposes or goals expressed in a user's input that the assistant recognizes.",
            },
            {"role": "user", "content": "How is it created?"},
            {
                "role": "assistant",
                "content": "To create an intent, you define example user phrases and train the assistant to recognize them.",
            },
            {"role": "user", "content": "Are those the only steps?"},
        ]),
    ),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content="Are those the only steps to create an intent?",
    ),
]


EXAMPLE_CONVERSATION: list[OPENAI_MESSAGE] = [
    ChatCompletionUserMessageParam(role="user", content="Привет!"),
    ChatCompletionAssistantMessageParam(role="assistant", content="Привет, как я могу помочь?"),
    ChatCompletionUserMessageParam(role="user", content="Что произошло в прошлом году?"),
    ChatCompletionAssistantMessageParam(role="assistant", content="В прошлом году был принят закон, хоть и не сказано, какой"),
    ChatCompletionUserMessageParam(role="user", content="Расскажи про Чарли Чаплина"),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content="Чарли Чаплин был одним из самых творческих и влиятельных людей в эпоху немого кино",
    ),
    ChatCompletionUserMessageParam(role="user", content="Когда он родился?"),
]

EXAMPLE_CONVERSATION_DOCUMENTS = [
    OpenAIDocument("Чарли Чаплин родился 16 апреля 1889 года в Лондоне."),
]


ANSWER_NO_CONTEXT_FEW_SHOTS: list[OPENAI_MESSAGE] = [
    ChatCompletionUserMessageParam(
        role="user",
        content="Who was the last Roman Emperor?",
    ),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content=(
            "I have no information about who the last Emperor of the Western Roman Empire and Eastern Roman Empire was. "
        ),
    ),
    ChatCompletionUserMessageParam(
        role="user",
        content="Is Red Hat Virtualization the same as Red Hat OpenShift virtualization?",
    ),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content=(
            "The documents do not contain any information about Red Hat Virtualization to assist in deciding about potential differences."
        ),
    ),
    ChatCompletionUserMessageParam(
        role="user",
        content="How many fatalities each year are caused by collisions between motorcycles and other vehicles in New York?",
    ),
    ChatCompletionAssistantMessageParam(
        role="assistant",
        content=(
            "I do not have specific information on the number of fatalities each year caused by collisions between motorcycles and other vehicles in New York. "
        ),
    ),
]
