import argparse
import json
from pathlib import Path
import textwrap

import jsonlines

from src.shared.conversations import task_to_conversation
from src.client.openai_compatible import LLM
from src.generation.assembly import MultiAgentQA
from src.data.utils import generation_task_from_json


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--api_key", default="testkey")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--doc_header", action="store_true")
    args = ap.parse_args()

    tasks = [
        (data, generation_task_from_json(data))
        for data in jsonlines.open(args.input) # type: ignore
    ]

    done_ids = [
        generation_task_from_json(data).task_id
        for data in (jsonlines.open(args.output)) # type: ignore
    ] if Path(args.output).exists() else []

    llm = LLM(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        ensure_nonempty=True,
    )
    qa = MultiAgentQA(llm)

    with open(args.output, 'a', encoding="utf-8", newline="\n") as fout:
        for task_idx, (task_json, task) in enumerate(tasks):
            print(f'{task_idx}/{len(tasks)}...')
            if task.task_id in done_ids:
                continue
            if args.max_examples is not None and task_idx >= args.max_examples:
                break

            conv = task_to_conversation(task, doc_header=args.doc_header)
            pred = qa.run(conv)
            task_json["predictions"] = [{"text": pred}]
            fout.write(json.dumps(task_json, ensure_ascii=False) + "\n")

            print(textwrap.shorten(pred, width=80))
