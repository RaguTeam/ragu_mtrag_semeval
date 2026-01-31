import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple

from src.client.openai_compatible import LLM
from src.generation.assembly import MultiAgentQA
from src.data.utils import GenerationTask, generation_task_from_json
from src.shared.schemas import ExtendedConversation, ExtendedMessage


def iter_jsonl_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_rows_and_tasks(path: str) -> Iterable[Tuple[Dict[str, Any], GenerationTask]]:
    for row in iter_jsonl_rows(path):
        task = generation_task_from_json(dict(row))
        yield row, task


def load_existing_task_ids(output_path: str) -> Set[str]:
    p = Path(output_path)
    if not p.exists():
        return set()
    done: Set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                tid = row.get("task_id")
                if tid and "predictions" in row:
                    done.add(tid)
            except Exception:
                continue
    return done


def task_to_conversation(task: GenerationTask, doc_header: bool = True) -> ExtendedConversation:
    msgs: list[ExtendedMessage] = []

    for m in task.input:
        if m.speaker == "user":
            role = "user"
        elif m.speaker == "agent":
            role = "assistant"
        else:
            raise ValueError(f"Unknown speaker {m.speaker!r} in task_id={task.task_id}")
        msgs.append(ExtendedMessage(role=role, content=m.text))

    for i, ctx in enumerate(task.contexts, start=1):
        doc_id = getattr(ctx, "document_id", None)
        title = getattr(ctx, "title", "") or ""
        text = getattr(ctx, "text", "") or ""

        doc_id_str = "" if doc_id is None else str(doc_id)

        if doc_header:
            header_bits: list[str] = []
            if title:
                header_bits.append(f"title={title}")
            if doc_id_str:
                header_bits.append(f"id={doc_id_str}")
            header = f"Document {i}" + (f" ({', '.join(header_bits)})" if header_bits else "")
            content = f"{header}:\n{text}"
        else:
            content = text

        msgs.append(ExtendedMessage(role="document", content=content))

    return ExtendedConversation(msgs)


def ensure_nonempty(pred: str, task_id: str, base_url: str, model: str) -> str:
    if pred is None or pred.strip() == "": # pyright: ignore[reportUnnecessaryComparison]
        raise RuntimeError(
            "Empty model output detected. Stopping to avoid writing invalid predictions.\n"
            f"task_id={task_id}\n"
            f"base_url={base_url}\n"
            f"model={model}\n"
            "Common causes: vLLM server not running, wrong base_url, model name mismatch, auth mismatch."
        )
    return pred.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--api_key", default="testkey")

    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--start_idx", type=int, default=0)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--doc_header", action="store_true")
    ap.add_argument("--keep_error", action="store_true", help="If set, write prediction_error then stop (still stops).")
    args = ap.parse_args()

    done_ids: set[str] = load_existing_task_ids(args.output) if args.resume else set()

    llm = LLM(api_key=args.api_key, base_url=args.base_url, model=args.model)
    qa = MultiAgentQA(llm)

    written = 0
    mode = "a" if args.resume and Path(args.output).exists() else "w"

    with open(args.output, mode, encoding="utf-8", newline="\n") as fout:
        for idx, (row, task) in enumerate(iter_rows_and_tasks(args.input)):
            if idx < args.start_idx:
                continue
            if args.max_examples is not None and (idx - args.start_idx) >= args.max_examples:
                break

            task_id = row.get("task_id") or task.task_id
            if not task_id:
                raise ValueError(f"Missing task_id at idx={idx}")

            if args.resume and task_id in done_ids:
                continue

            conv = task_to_conversation(task, doc_header=args.doc_header)

            try:
                pred = qa.run(conv)
                pred = ensure_nonempty(pred, task_id, args.base_url, args.model)
                row["predictions"] = [{"text": pred}]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                if args.keep_error:
                    row["predictions"] = [{"text": ""}]
                    row["prediction_error"] = f"{type(e).__name__}: {e}"
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                raise  

    print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
