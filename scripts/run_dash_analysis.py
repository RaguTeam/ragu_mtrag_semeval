from itertools import chain
import jsonlines

from src.data.utils import GenerationTaskAnalysis, generation_task_from_json
from src.dash.analysis import run_dashboard

def load_preds(path: str) -> list[GenerationTaskAnalysis]:
    return [
        GenerationTaskAnalysis.from_task(generation_task_from_json(x)) # type: ignore
        for x in jsonlines.open(path) # type: ignore
    ]

preds = {
    'meno_gte': load_preds(
        'data/old_set/RAGU/test_answers_subsample_meno_gte.jsonl'
    ),
    'qwen_gte': load_preds(
        'data/old_set/RAGU/test_answers_subsample_qwen_gte.jsonl'
    ),
    'meno_gte_new_prompt': load_preds(
        'data/old_set/RAGU/test_answers_subsample_meno_gte_new_prompt.jsonl'
    ),
    'qwen3_4b_eval_by_gemma': load_preds(
        'data/test_96/eval_qwen34b_coreference_test96.jsonl'
    ),
    'qwen3_8b_eval_by_gemma': load_preds(
        'data/test_96/eval_qwen38b_coreference_test96.jsonl'
    ),
    # 'answers_subsample_result_2': load_preds(
    #     'data/old_set/RAGU/answers_subsample_result_2.jsonl'
    # ),
}

for pred in chain(*preds.values()):
    pred.analysis = [('q1', 'prompt_q1', 'gemini', 'answer1'), ('q2', 'prompt_q2', 'gemini', 'answer2')]

run_dashboard(preds)
