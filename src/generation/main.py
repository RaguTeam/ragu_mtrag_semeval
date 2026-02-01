"""Main script to run the multi-agent QA system."""

from src.agents.answer import AnswerAgent
from src.client.openai_compatible import LLM
from src.agents.coreference import CoreferenceAgent
from src.generation.assembly import MultiAgentQA
from src.shared.prompts import EXAMPLE_CONVERSATION, EXAMPLE_CONVERSATION_DOCUMENTS

if __name__ == "__main__":
    host = "http://localhost:8000/v1"
    api_key = "testkey"

    llm = LLM(
        api_key=api_key,
        base_url=host,
        model="Qwen/Qwen3-4B-FP8",
    )
    qa_system = MultiAgentQA(llm, coref_agent=CoreferenceAgent(llm), answer_agent=AnswerAgent(llm))
    answer = qa_system.run(EXAMPLE_CONVERSATION, EXAMPLE_CONVERSATION_DOCUMENTS)
    print("Final Answer:", answer)
