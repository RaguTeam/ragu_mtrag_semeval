"""Main script to run the multi-agent QA system."""

from src.client.openai_compatible import LLM
from src.generation.assembly import MultiAgentQA
from src.shared.conversations import EXAMPLE_CONVERSATION

if __name__ == "__main__":
    host = "http://localhost:8000/v1"
    api_key = "testkey"

    llm = LLM(
        api_key=api_key,
        base_url=host,
        model="Qwen/Qwen3-4B-FP8",
    )
    qa_system = MultiAgentQA(llm)
    answer = qa_system.run(EXAMPLE_CONVERSATION)
    print("Final Answer:", answer)
