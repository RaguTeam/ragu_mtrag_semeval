import json
from pathlib import Path
import textwrap
from typing import Any, cast

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import jsonlines

from src.analysis.analysers import Analyser
from src.shared.conversations import task_to_conversation
from src.data.utils import generation_task_from_json


@hydra.main(version_base=None, config_path="../../conf/analysis", config_name="template")
def main(cfg: DictConfig):
    tasks = [
        (data, generation_task_from_json(data))
        for data in jsonlines.open(cfg.input) # type: ignore
    ]

    if cfg.split_file:
        task_ids = set(cast(list[str], json.loads(Path(cfg.split_file).read_text())))
        tasks = [(data, task) for data, task in tasks if task.task_id in task_ids]

    done_ids = [
        data['task_id']
        for data in (jsonlines.open(cfg.output)) # type: ignore
    ] if Path(cfg.output).exists() else []

    analyser: Analyser = instantiate(cfg.analyser)

    with open(cfg.output, 'a', encoding="utf-8", newline="\n") as fout:
        for task_idx, (_task_json, task) in enumerate(tasks):
            print(f'{task_idx}/{len(tasks)}...')
            if task.task_id in done_ids:
                print('Already done')
                continue
            if cfg.max_examples is not None and task_idx >= cfg.max_examples:
                break

            conv, docs = task_to_conversation(task)
            assert task.target
            title, prompt, model_name, answer = analyser.analyse(
                conversation=conv,
                reference_answer=task.target.text,
                docs=docs,
                predicted_answer=(
                    task.predictions[0].text
                    if task.predictions
                    else None
                ),  # may be None
                metrics=task.metrics,  # may be None
            )

            output_dict: dict[str, Any] = {
                'task_id': task.task_id,
                'analysis': {
                    'title': title,
                    'prompt': prompt,
                    'model_name': model_name,
                    'answer': answer,
                },
            }

            fout.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

            print('---', title, model_name, '---')
            print(textwrap.shorten(prompt, width=80))
            print(textwrap.shorten(answer, width=80))


if __name__ == "__main__":
    main()