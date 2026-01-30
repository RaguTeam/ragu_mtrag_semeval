import jsonlines

from src.data.utils import GenerationTaskAnalysis, generation_task_from_json
from src.dash.analysis import run_dashboard


tasks = [
    GenerationTaskAnalysis.from_task(generation_task_from_json(x)) # type: ignore
    for x in jsonlines.open('data/old_set/RAGU/answers_subsample_result_2.jsonl') # type: ignore
]
run_dashboard(tasks)


