"""Main script to run the multi-agent QA system."""

from src.client.openai_compatible import LLM
from src.generation.assembly import MultiAgentQA
from src.shared.schemas import ExtendedConversation, ExtendedMessage

if __name__ == "__main__":
    conversation = ExtendedConversation(
        [
            ExtendedMessage("user", "Что произошло в прошлом году?"),
            ExtendedMessage("document", "Компания отчиталась о рекордной прибыли в 2023 году."),
            ExtendedMessage("document", "В прошлом году был принят новый закон."),
        ],
    )

    host = "http://localhost:8000/v1"
    api_key = "testkey"

    llm = LLM(
        api_key=api_key,
        base_url=host,
        model="Qwen/Qwen3-4B-FP8",
    )
    qa_system = MultiAgentQA(llm)
    answer = qa_system.run(conversation)
    print("Final Answer:", answer)
