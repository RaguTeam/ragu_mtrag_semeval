import json
from pathlib import Path
import textwrap
from typing import cast

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import jsonlines

from src.shared.conversations import task_to_conversation
from src.data.utils import generation_task_from_json
from src.generation.assembly import MultiAgentQA


@hydra.main(version_base=None, config_path="../../conf/generate", config_name="default")
def main(cfg: DictConfig):
    tasks = [
        (data, generation_task_from_json(data))
        for data in jsonlines.open(cfg.input) # type: ignore
    ]

    if cfg.split_file:
        task_ids = set(cast(list[str], json.loads(Path(cfg.split_file).read_text())))
        tasks = [(data, task) for data, task in tasks if task.task_id in task_ids]

    done_ids = [
        generation_task_from_json(data).task_id
        for data in (jsonlines.open(cfg.output)) # type: ignore
    ] if Path(cfg.output).exists() else []

    qa: MultiAgentQA = instantiate(cfg.qa)

    with open(cfg.output, 'a', encoding="utf-8", newline="\n") as fout:
        for task_idx, (task_json, task) in enumerate(tasks):
            print(f'{task_idx}/{len(tasks)}...')
            if task.task_id in done_ids:
                print('Already done')
                continue
            if cfg.max_examples is not None and task_idx >= cfg.max_examples:
                break

            if (
                not task.contexts
                and cfg.no_context_placeholder is not None
            ):
                pred = cast(str, cfg.no_context_placeholder)
                print(f'Using placeholder "{pred}"')
            else:
                conv, docs = task_to_conversation(task)
                pred = qa.run(conv, docs)
            
            task_json["predictions"] = [{"text": pred}]
            fout.write(json.dumps(task_json, ensure_ascii=False) + "\n")

            print(textwrap.shorten(pred, width=80))


if __name__ == "__main__":
    main()