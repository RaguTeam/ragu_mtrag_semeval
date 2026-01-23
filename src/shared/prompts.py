"""Prompts used for llm generation."""

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
    "Message to rewrite:\n"
    "{message}\n\n"
    "Message with resolved coreferences:"
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

DOCUMENT_TEMPLATE = "\n\n[Документ]\n{content}"
