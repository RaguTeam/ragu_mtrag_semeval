import argparse
import json
from typing import Dict, Any, Iterable


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[ERROR] {path}:{lineno} invalid JSON: {e}") from e


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge missing metadata fields into predictions JSONL by task_id.",
    )
    ap.add_argument("--pred", required=True, help="Predictions jsonl (to be updated)")
    ap.add_argument("--ref", required=True, help="Reference jsonl (source of fields)")
    ap.add_argument("--out", required=True, help="Output jsonl")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite fields even if they already exist in pred file",
    )
    args = ap.parse_args()

    # Build mapping task_id -> metadata fields from reference
    ref_map: Dict[str, Dict[str, Any]] = {}
    wanted_keys = ["answerability", "Question Type", "Multi-Turn", "Collection"]

    for row in iter_jsonl(args.ref):
        tid = row.get("task_id")
        if not tid:
            continue
        payload = {}
        for k in wanted_keys:
            if k in row and row[k] is not None:
                payload[k] = row[k]
        if payload:
            ref_map[tid] = payload

    total = 0
    matched = 0
    filled_any = 0
    missing_in_ref = 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as fout:
        for row in iter_jsonl(args.pred):
            total += 1
            tid = row.get("task_id")
            if not tid or tid not in ref_map:
                if tid:
                    missing_in_ref += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            matched += 1
            before = dict(row)
            meta = ref_map[tid]

            for k, v in meta.items():
                if args.overwrite or (k not in row) or (row[k] is None):
                    row[k] = v

            if row != before:
                filled_any += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Total pred rows: {total}")
    print(f"Matched by task_id: {matched}")
    print(f"Rows updated (filled/overwritten): {filled_any}")
    print(f"Pred rows with task_id missing in ref: {missing_in_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
