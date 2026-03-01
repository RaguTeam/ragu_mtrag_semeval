import argparse
import json


def main():
    ap = argparse.ArgumentParser(
        description="Remove rows with Answerability == ['UNDERSPECIFIED']",
    )
    ap.add_argument("--input", required=True, help="Input JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL")
    args = ap.parse_args()

    removed = 0
    kept = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8", newline="\n") as fout:

        for lineno, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            # Support both capitalized and lowercase variants
            ans = row.get("Answerability") or row.get("answerability")

            if ans == ["UNDERSPECIFIED"]:
                removed += 1
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Removed: {removed}")
    print(f"Kept: {kept}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
