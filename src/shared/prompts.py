"""Prompts used for llm generation."""

COREFERENCE_RESOLUTION = (
    "You are a coreference resolution agent.\n"
    "Today's date is {date}.\n\n"
    "Your task:\n"
    "- Resolve all coreferences in the document.\n"
    "- Replace relative temporal expressions (e.g. "
    "'last year', 'this year', 'recently', 'now') with "
    "explicit calendar dates or years relative to today's date.\n"
    "- Resolve pronouns and vague references ('he', 'it', "
    "'this law', 'the company') using the dialogue context.\n"
    "- Output ONLY the rewritten document text, without explanations."
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
