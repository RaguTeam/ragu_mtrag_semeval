import json
from pathlib import Path
import jsonlines

from src.data.utils import GenerationTaskAnalysis, generation_task_from_json
from src.dash.analysis import run_dashboard

def load_preds(path: str) -> list[GenerationTaskAnalysis]:
    return [
        GenerationTaskAnalysis.from_task(generation_task_from_json(x)) # type: ignore
        for x in jsonlines.open(path) # type: ignore
    ]

def load_analysis(path: str) -> dict[str, tuple[str, str, str, str]]:
    return {
        row['task_id']: (
            row['analysis']['title'],
            row['analysis']['prompt'],
            row['analysis']['model_name'],
            row['analysis']['answer'],
        )
        for row in jsonlines.open(path) # pyright: ignore[reportUnknownMemberType]
    }

preds = {
    # gemini default prompt preds + metrics
    'gemini1': load_preds('data/old_set/metrics/test_gemini-3-pro-preview-high.jsonl'),
    # gemini new prompt preds + metrics
    'gemini2': load_preds('data/old_set/metrics/test_gemini-3-pro-preview-high_new_prompt.jsonl'),
    # gemini new prompt 2 preds + metrics
    'gemini3': load_preds('data/old_set/metrics/test_gemini-3-pro-preview-high_new_prompt_2_test_1.jsonl'),
    # qwen preds + metrics
    'qwen3-32b-coref': load_preds('data/old_set/metrics/test_qwen332b_coreference_test96.jsonl'),
    # ragu preds + metrics
    'RAGU new doc view': load_preds('data/old_set/metrics/ragu_new_doc_view_meno_gte.jsonl'),
    # lightRAG + gpt4.1mini preds + metrics
    # 'lightRAG + gpt4.1-mini': load_preds('data/old_set/metrics/test_eval_lightrag_and_gpt41mini_test96.jsonl'),
}

analysis = [
    load_analysis('data/old_set/analysis/agent_behaviour.json'),
    load_analysis('data/old_set/analysis/reference_validator.json'),
    load_analysis('data/old_set/analysis/reference_validator_gpt5.json'),
    load_analysis('data/old_set/analysis/reference_validator_claude.json'),
    load_analysis('data/old_set/analysis/self_improvement.json'),
]

task_ids = json.loads(Path('splits/test_1.json').read_text())

for name, lst in list(preds.items()):
    preds[name] = [task for task in lst if task.task_id in task_ids]

for name, lst in list(preds.items()):
    for task in lst:
        task.analysis = [
            analysis_dict[task.task_id]
            for analysis_dict in analysis
            if task.task_id in analysis_dict
        ]

run_dashboard(preds)
