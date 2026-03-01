import argparse
import json
import math
from collections import Counter, defaultdict

BASE_METRICS = ["RB_agg", "RB_llm", "RL_F"]
PRECONDITIONED_SUFFIX = " (Pre-conditioned)"
EXCLUDE_ANSWERABILITY = {"UNDERSPECIFIED", "UNANSWERABLE"}


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


def get_unconditioned_value(metrics: dict, m: str):
    """
    Canonical unconditioned value for metric m.

    Conventions supported:
      - Swapped: '{m} (Pre-conditioned)' is unconditioned, plain '{m}' is conditioned
      - Old: plain '{m}' is unconditioned
    """
    key_pc = f"{m}{PRECONDITIONED_SUFFIX}"
    if key_pc in metrics:
        return to_scalar(metrics.get(key_pc))
    return to_scalar(metrics.get(m))


def get_conditioned_value(metrics: dict, m: str):
    """
    Canonical conditioned (idk) value for metric m.

    Priority:
      1) '{m}_idk' (paper-style)
      2) if '{m} (Pre-conditioned)' exists, then plain '{m}' is the conditioned value (swap convention)
      3) otherwise: no conditioned value
    """
    key_idk = f"{m}_idk"
    if key_idk in metrics:
        return to_scalar(metrics.get(key_idk))

    key_pc = f"{m}{PRECONDITIONED_SUFFIX}"
    if key_pc in metrics and m in metrics:
        return to_scalar(metrics.get(m))

    return None


def update_acc(acc, metrics: dict):
    """Update accumulator with metrics from one row."""
    sums, counts = acc["sums"], acc["counts"]

    for m in BASE_METRICS:
        v = get_unconditioned_value(metrics, m)
        if v is not None:
            sums[m] += v
            counts[m] += 1

        v_idk = get_conditioned_value(metrics, m)
        if v_idk is not None:
            mk = f"{m}_idk"
            sums[mk] += v_idk
            counts[mk] += 1


def finalize_means(sums, counts):
    return {k: (sums[k] / counts[k] if counts[k] else None) for k in sorted(sums)}


def print_block(title, means, counts, metric_keys):
    print(f"\n{title}")
    for k in metric_keys:
        if k in means and means[k] is not None:
            print(f"{k:12s} {means[k]:.6f}  (n={counts.get(k, 0)})")
        else:
            print(f"{k:12s} MISSING")

    vals = [means.get(k) for k in metric_keys]
    hm = harmonic_mean(vals)
    if hm is None:
        print(f"{'HM(3)':12s} MISSING (needs all 3 > 0)")
    else:
        print(f"{'HM(3)':12s} {hm:.6f}")


def print_group_report(group_name: str, total_rows: int, acc, idk_dist: Counter, unknown: int):
    means = finalize_means(acc["sums"], acc["counts"])
    counts = acc["counts"]

    print("\n" + "=" * 80)
    print(f"Group: {group_name}  |  Rows: {total_rows}")

    print_block("Unconditioned means (canonical: RB_*, RL_F)", means, counts, BASE_METRICS)

    conditioned_keys = [f"{m}_idk" for m in BASE_METRICS]
    any_conditioned = any(counts.get(k, 0) > 0 for k in conditioned_keys)
    if any_conditioned:
        print_block("Conditioned means (canonical: *_idk)", means, counts, conditioned_keys)
    else:
        print("\nConditioned means: MISSING (no conditioned fields found in this group)")

    print("\nidk_eval distribution:")
    if total_rows > 0:
        for label, c in idk_dist.most_common():
            print(f"{label:10s} {c:5d} ({c/total_rows:.1%})")
        print(f"\nunknown_rate: {unknown/total_rows:.2%}")
    else:
        print("(empty group)")


def agg_file_grouped(path: str):
    # Global (ALL)
    global_acc = {"sums": defaultdict(float), "counts": defaultdict(int)}
    global_idk_dist = Counter()
    global_total = 0
    global_unknown = 0

    # Global excluding UNDERSPECIFIED (works even when answerability is a list)
    global_excl_acc = {"sums": defaultdict(float), "counts": defaultdict(int)}
    global_excl_idk_dist = Counter()
    global_excl_total = 0
    global_excl_unknown = 0

    # Per-answerability (keyed by stringified list, e.g. "['UNDERSPECIFIED']")
    group_acc = defaultdict(lambda: {"sums": defaultdict(float), "counts": defaultdict(int)})
    group_idk_dist = defaultdict(Counter)
    group_total = defaultdict(int)
    group_unknown = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            metrics = obj.get("metrics", {}) or {}

            # answerability is ALWAYS a list in data
            labels = obj.get("answerability") or ["unknown"]
            labels = [str(x).strip() for x in labels if str(x).strip()] or ["unknown"]
            group_key = str(labels)

            # idk_eval
            raw = metrics.get("idk_eval")
            idk_val = raw[0] if isinstance(raw, list) and raw else raw
            if idk_val is None:
                idk_val = "unknown"
            idk_val_str = str(idk_val)

            # Update ALL
            global_total += 1
            global_idk_dist[idk_val_str] += 1
            if idk_val_str == "unknown":
                global_unknown += 1
            update_acc(global_acc, metrics)

            if not any(lab in EXCLUDE_ANSWERABILITY for lab in labels):
                global_excl_total += 1
                global_excl_idk_dist[idk_val_str] += 1
                if idk_val_str == "unknown":
                    global_excl_unknown += 1
                update_acc(global_excl_acc, metrics)

            # Update per-group
            group_total[group_key] += 1
            group_idk_dist[group_key][idk_val_str] += 1
            if idk_val_str == "unknown":
                group_unknown[group_key] += 1
            update_acc(group_acc[group_key], metrics)

    return (
        (global_total, global_acc, global_idk_dist, global_unknown),
        (global_excl_total, global_excl_acc, global_excl_idk_dist, global_excl_unknown),
        (group_total, group_acc, group_idk_dist, group_unknown),
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            f"Aggregate MT-RAG eval JSONL: ALL + ALL(excluding filed in the EXCLUDE_ANSWERABILITY set in code) + per-answerability.\n"
            "Supports naming swap:\n"
            "  - '{m} (Pre-conditioned)' = unconditioned; plain '{m}' = conditioned\n"
            "Also supports paper-style '{m}_idk' as conditioned."
        ),
    )
    ap.add_argument("-i", "--input", required=True, help="Path to evaluated .jsonl file")
    ap.add_argument(
        "--only-groups",
        action="store_true",
        help="If set, do not print global reports; print only per-answerability reports.",
    )
    args = ap.parse_args()

    (all_pack, excl_pack, groups_pack) = agg_file_grouped(args.input)

    (global_total, global_acc, global_idk_dist, global_unknown) = all_pack
    (excl_total, excl_acc, excl_idk_dist, excl_unknown) = excl_pack
    (group_total, group_acc, group_idk_dist, group_unknown) = groups_pack

    if not args.only_groups:
        print_group_report("ALL", global_total, global_acc, global_idk_dist, global_unknown)
        print_group_report(
            f"ALL (excluding answerability {EXCLUDE_ANSWERABILITY})",
            excl_total,
            excl_acc,
            excl_idk_dist,
            excl_unknown,
        )

    for g, n in sorted(group_total.items(), key=lambda x: x[1], reverse=True):
        print_group_report(
            f"answerability={g}",
            n,
            group_acc[g],
            group_idk_dist[g],
            group_unknown[g],
        )


if __name__ == "__main__":
    main()
