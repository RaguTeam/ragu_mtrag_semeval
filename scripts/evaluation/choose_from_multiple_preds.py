import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from src.data.utils import generation_task_from_json  # type: ignore


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


def dataclass_to_dict(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x


def main() -> int:
    ap = argparse.ArgumentParser("Pick best sample across multiple eval JSONL files by Ragas RL_F.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files.")
    ap.add_argument("--output", required=True, help="Output JSONL (best-of).")
    ap.add_argument("--key", default="task_id", help="Key to match samples.")
    ap.add_argument("--tie-break", choices=["first", "longer_pred"], default="longer_pred")
    args = ap.parse_args()

    # key -> (best_score, best_task, source_file)
    best: Dict[str, Tuple[float, Any, str]] = {}

    # counts / diagnostics
    chosen_counts = Counter()
    keys_in_file = {p: set() for p in args.inputs}
    dup_in_file = {p: Counter() for p in args.inputs}

    # key -> set of (conversation_id, turn)
    identity = defaultdict(set)

    def make_key(task) -> str:
        if args.key == "task_id":
            return str(task.task_id)
        # conversation_turn
        return f"{task.conversation_id}<::>{task.turn}"

    def pred_len(task) -> int:
        try:
            if task.predictions and task.predictions[0] and task.predictions[0].text:
                return len(task.predictions[0].text)
        except Exception:
            pass
        return 0

    for path in args.inputs:
        if not os.path.exists(path):
            raise SystemExit(f"[ERROR] not found: {path}")

        seen = Counter()
        for row in iter_jsonl(path):
            task = generation_task_from_json(row)

            k = make_key(task)
            seen[k] += 1
            keys_in_file[path].add(k)

            identity[k].add((str(task.conversation_id), int(task.turn)))

            score: Optional[float] = None
            if task.metrics is not None:
                score = getattr(task.metrics, "RL_F", None)

            if score is None:
                # treat missing as very bad
                score_val = float("-inf")
            else:
                score_val = float(score)

            prev = best.get(k)
            if prev is None:
                best[k] = (score_val, task, path)
            else:
                prev_score, prev_task, prev_path = prev
                if score_val > prev_score:
                    best[k] = (score_val, task, path)
                elif score_val == prev_score and args.tie_break == "longer_pred":
                    if pred_len(task) > pred_len(prev_task):
                        best[k] = (score_val, task, path)

        for k, c in seen.items():
            if c > 1:
                dup_in_file[path][k] = c

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for k in sorted(best.keys()):
            score, task, src = best[k]
            chosen_counts[src] += 1
            out.write(json.dumps(dataclass_to_dict(task), ensure_ascii=False) + "\n")

    union_keys = set().union(*keys_in_file.values())

    print("\n=== Best-of selection summary ===")
    print(f"unique keys in union: {len(union_keys)}")
    print(f"written best rows: {len(best)} -> {args.output}")

    print("\nChosen rows per file:")
    for path in args.inputs:
        print(f"  {path}: {chosen_counts.get(path, 0)}")

    print("\n=== Coverage mismatches (missing keys per file) ===")
    any_cov = False
    for path in args.inputs:
        missing = union_keys - keys_in_file[path]
        if missing:
            any_cov = True
            print(f"  {path}: missing {len(missing)} keys")
    if not any_cov:
        print("  (none)")

    print("\n=== Duplicate keys within files ===")
    any_dups = False
    for path in args.inputs:
        if dup_in_file[path]:
            any_dups = True
            print(f"  {path}: {len(dup_in_file[path])} duplicated keys (showing up to 10)")
            for kk, cc in dup_in_file[path].most_common(10):
                print(f"    {kk}: {cc}")
    if not any_dups:
        print("  (none)")

    print("\n=== Cross-file identity mismatches (same key -> different (conversation_id, turn)) ===")
    mism = {k: v for k, v in identity.items() if len(v) > 1}
    if not mism:
        print("  (none)")
    else:
        print(f"  keys with mismatched identity: {len(mism)} (showing up to 20)")
        for i, (k, vals) in enumerate(list(mism.items())[:20], start=1):
            print(f"  {i}. {k}: {sorted(vals)}")

    # stats: mean + percentiles (more useful than median)
    scores = [s for (s, _, _) in best.values() if s != float("-inf")]
    if scores:
        scores_sorted = sorted(scores)
        mean = sum(scores_sorted) / len(scores_sorted)

        perfect = sum(1 for s in scores_sorted if s >= 0.999)

        print("\n=== RL_F stats over selected rows ===")
        print(
            f"  min={scores_sorted[0]:.6f} "
            f"mean={mean:.6f} "
            f"max={scores_sorted[-1]:.6f}"
        )
        print(f"  perfect (>=0.999): {perfect}/{len(scores_sorted)} ({perfect/len(scores_sorted)*100:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
