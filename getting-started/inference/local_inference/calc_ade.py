#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Iterable, List, Tuple


def l2(p1: Iterable[float], p2: Iterable[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def average_l2(gt: List[List[float]], pred: List[List[float]]) -> Tuple[float, int, int]:
    n = min(len(gt), len(pred))
    if n == 0:
        return float("nan"), len(gt), len(pred)
    total = 0.0
    for i in range(n):
        total += l2(gt[i], pred[i])
    return total / n, len(gt), len(pred)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute ADE between ground_truth and llm_answer.")
    parser.add_argument(
        "input",
        nargs="?",
        default="vec_emb_output_results.json",
        help="Path to JSONL results file (default: vec_emb_output_results.json)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a plot of ground_truth vs llm_answer for each entry.",
    )
    args = parser.parse_args()

    per_entry = []
    total_weighted = 0.0
    total_points = 0
    mismatched = 0

    plot_dir = None
    if args.plot:
        import matplotlib.pyplot as plt

        plot_dir = os.path.dirname(os.path.abspath(args.input)) or "."

    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            gt = data.get("ground_truth", [])
            pred = data.get("llm_answer", [])
            ade, gt_len, pred_len = average_l2(gt, pred)
            per_entry.append((data.get("sid"), ade, gt_len, pred_len))
            n = min(gt_len, pred_len)
            if gt_len != pred_len:
                mismatched += 1
            if n > 0 and not math.isnan(ade):
                total_weighted += ade * n
                total_points += n

            if args.plot and plot_dir is not None:
                sid = data.get("sid")
                sid_str = sid if sid is not None else f"entry_{line_no}"
                safe_sid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in sid_str)
                out_path = os.path.join(plot_dir, f"vec_emb_{safe_sid}.png")

                gt_x = [p[0] for p in gt]
                gt_y = [p[1] for p in gt]
                pred_x = [p[0] for p in pred]
                pred_y = [p[1] for p in pred]

                plt.figure(figsize=(6, 6))
                plt.plot(gt_x, gt_y, label="ground_truth", linewidth=2)
                plt.plot(pred_x, pred_y, label="llm_answer", linewidth=2)
                plt.title(sid_str)
                plt.axis("equal")
                plt.grid(True, linestyle="--", alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()

    if not per_entry:
        print("No entries found.")
        return 1

    mean_unweighted = sum(x[1] for x in per_entry if not math.isnan(x[1])) / len(
        [x for x in per_entry if not math.isnan(x[1])]
    )
    mean_weighted = total_weighted / total_points if total_points else float("nan")

    print(f"Entries: {len(per_entry)}")
    print(f"Unweighted ADE mean: {mean_unweighted:.6f}")
    print(f"Weighted ADE mean:   {mean_weighted:.6f}")
    if mismatched:
        print(f"Length mismatches:  {mismatched}")

    for sid, ade, gt_len, pred_len in per_entry:
        sid_str = sid if sid is not None else "<no sid>"
        print(f"{sid_str}: ADE={ade:.6f} (gt={gt_len}, pred={pred_len})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
