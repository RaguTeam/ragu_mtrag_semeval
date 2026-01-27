import argparse
import json
import math
from collections import Counter, defaultdict

BASE_METRICS = ["RB_agg", "RB_llm", "RL_F"]

def to_scalar(v):
    """Accept scalar or [scalar]. Return float or None (strings -> None)."""
    if v is None:
        return None
    if isinstance(v, list):
        if not v:
            return None
        v = v[0]
    if isinstance(v, (int, float)):
        fv = float(v)
        return None if math.isnan(fv) else fv
    return None

def harmonic_mean(values):
    """Harmonic mean of positive values; returns None if invalid."""
    vals = [v for v in values if v is not None and v > 0.0]
    if len(vals) != len(values):  # require all three to be present and >0
        return None
    return len(vals) / sum(1.0 / v for v in vals)

def agg_file(path: str):
    sums = defaultdict(float)
    counts = defaultdict(int)

    idk_dist = Counter()
    total_rows = 0
    unknown = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            obj = json.loads(line)
            metrics = obj.get("metrics", {}) or {}

            # track idk_eval parsing failures
            raw = metrics.get("idk_eval")
            idk_val = raw[0] if isinstance(raw, list) and raw else raw
            if idk_val is None:
                idk_val = "unknown"
            idk_dist[str(idk_val)] += 1
            if str(idk_val) == "unknown":
                unknown += 1

            # aggregate both conditioned and unconditioned metrics
            for m in BASE_METRICS:
                # unconditioned
                v = to_scalar(metrics.get(m))
                if v is not None:
                    sums[m] += v
                    counts[m] += 1

                # conditioned (if exists)
                midk = f"{m}_idk"
                if midk in metrics:
                    v_idk = to_scalar(metrics.get(midk))
                    if v_idk is not None:
                        sums[midk] += v_idk
                        counts[midk] += 1

    means = {}
    for k in sorted(sums):
        means[k] = sums[k] / counts[k] if counts[k] else None

    return total_rows, means, counts, idk_dist, unknown

def print_block(title, means, counts, metric_keys):
    print(f"\n{title}")
    for k in metric_keys:
        if k in means and means[k] is not None:
            print(f"{k:12s} {means[k]:.6f}  (n={counts.get(k, 0)})")
        else:
            print(f"{k:12s} MISSING")

    # harmonic mean over the three reported means
    vals = [means.get(k) for k in metric_keys]
    hm = harmonic_mean(vals)
    if hm is None:
        print(f"{'HM(3)':12s} MISSING (needs all 3 > 0)")
    else:
        print(f"{'HM(3)':12s} {hm:.6f}")

def main():
    ap = argparse.ArgumentParser(description="Aggregate MT-RAG eval JSONL: conditioned + unconditioned + harmonic mean.")
    ap.add_argument("-i", "--input", required=True, help="Path to evaluated .jsonl file")
    args = ap.parse_args()

    total_rows, means, counts, idk_dist, unknown = agg_file(args.input)

    print(f"Rows: {total_rows}")

    # Unconditioned
    print_block(
        "Unconditioned means",
        means,
        counts,
        BASE_METRICS,
    )

    # Conditioned (only if present)
    conditioned_keys = [f"{m}_idk" for m in BASE_METRICS]
    any_conditioned = any(k in means for k in conditioned_keys)
    if any_conditioned:
        print_block(
            "Conditioned means (paper-style: *_idk)",
            means,
            counts,
            conditioned_keys,
        )
    else:
        print("\nConditioned means: MISSING (no *_idk fields found in file)")

    # idk_eval distribution
    print("\nidk_eval distribution:")
    for label, c in idk_dist.most_common():
        print(f"{label:10s} {c:5d} ({c/total_rows:.1%})")
    print(f"\nunknown_rate: {unknown/total_rows:.2%}")

if __name__ == "__main__":
    main()
