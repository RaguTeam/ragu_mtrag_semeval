from src.data.utils import load_generation_tasks
from src.dash.generation_task import run_dashboard


generation_tasks = load_generation_tasks('../mt-rag-benchmark/human/generation_tasks/reference.jsonl')
run_dashboard(generation_tasks)
