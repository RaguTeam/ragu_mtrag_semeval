import argparse
import json


def main():
    ap = argparse.ArgumentParser(description="Remove evaluation fields (metrics/analysis) from a JSONL submission.")
    ap.add_argument("-i", "--input", required=True, help="Path to input .jsonl")
    ap.add_argument("-o", "--output", required=True, help="Path to output .jsonl")
    args = ap.parse_args()

    total = 0
    removed_metrics = 0
    removed_analysis = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[ERROR] {args.input}:{lineno} invalid JSON: {e}") from e

            if "metrics" in row:
                removed_metrics += 1
                row.pop("metrics", None)

            if "analysis" in row:
                removed_analysis += 1
                row.pop("analysis", None)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Done. Wrote {total} rows to: {args.output}")
    print(f"Rows with removed 'metrics':  {removed_metrics}")
    print(f"Rows with removed 'analysis': {removed_analysis}")


if __name__ == "__main__":
    main()
