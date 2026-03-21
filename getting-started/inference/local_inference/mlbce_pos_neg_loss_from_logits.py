#!/usr/bin/env python3
import argparse
import json
from typing import List

import torch
import torch.nn.functional as F


def _parse_float_list(raw: str) -> List[float]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [float(x.strip()) for x in raw.split(",")]


def _parse_mark_list(raw: str) -> List[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    marks = [x.strip().lower() for x in raw.split(",")]
    for m in marks:
        if m not in {"pos", "neg", "ignore"}:
            raise ValueError(f"Unsupported mark '{m}'. Allowed: pos, neg, ignore.")
    return marks


def _parse_int_list(raw: str) -> List[int]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    vals = [int(x.strip()) for x in raw.split(",")]
    for v in vals:
        if v not in {0, 1}:
            raise ValueError("Sampling keep mask must contain only 0 or 1.")
    return vals


def _build_inputs(args):
    if args.logit is not None or args.mark is not None:
        if args.logit is None or args.mark is None:
            raise ValueError("Single-token mode requires both --logit and --mark.")
        mark = args.mark.strip().lower()
        if mark not in {"pos", "neg", "ignore"}:
            raise ValueError("Single-token --mark must be one of: pos, neg, ignore.")
        logits = [float(args.logit)]
        marks = [mark]
        weights = [float(args.weight)]
        keep_mask = [int(args.neg_keep)]
    else:
        logits = _parse_float_list(args.logits)
        marks = _parse_mark_list(args.marks)
        if len(logits) == 0:
            raise ValueError("Provide logits via --logits (or single-token --logit).")
        if len(marks) != len(logits):
            raise ValueError("Length mismatch: --marks must have same length as --logits.")

        if args.weights is None:
            weights = [1.0] * len(logits)
        else:
            weights = _parse_float_list(args.weights)
            if len(weights) != len(logits):
                raise ValueError("Length mismatch: --weights must have same length as --logits.")

        keep_mask = None
        if args.neg_keep_mask is not None:
            keep_mask = _parse_int_list(args.neg_keep_mask)
            if len(keep_mask) != len(logits):
                raise ValueError("Length mismatch: --neg_keep_mask must match --logits length.")

    return logits, marks, weights, keep_mask


def compute_pos_neg_multilabel_bce(
    logits: List[float],
    marks: List[str],
    weights: List[float],
    pos_weight: float = 100.0,
    apply_half_negative_sampling: bool = False,
    neg_keep_mask: List[int] = None,
    seed: int = 42,
    gt_is_vec: bool = True,
):
    # This script assumes all provided tokens are VEC tokens.
    # It reproduces the positive/negative parts used in multi_label_bce_loss,
    # with type loss intentionally excluded.
    logits_t = torch.tensor(logits, dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)

    pos_mask = torch.tensor([m == "pos" for m in marks], dtype=torch.bool)
    neg_candidate_mask = torch.tensor([m == "neg" for m in marks], dtype=torch.bool)

    if neg_keep_mask is not None:
        neg_keep = torch.tensor([bool(x) for x in neg_keep_mask], dtype=torch.bool)
        neg_mask = neg_candidate_mask & neg_keep
    elif apply_half_negative_sampling:
        torch.manual_seed(seed)
        sampled = torch.rand_like(logits_t) < 0.5
        neg_mask = neg_candidate_mask & sampled
    else:
        # Deterministic path: all marked negatives are used.
        neg_mask = neg_candidate_mask

    step_gate = 1.0 if gt_is_vec else 0.0

    pos_logits = logits_t[pos_mask]
    pos_weights = weights_t[pos_mask]
    pos_logsig = F.logsigmoid(pos_logits)
    pos_denom = max(1, int(pos_mask.sum().item()))

    if pos_logits.numel() > 0:
        pos_loss_ungated = (-(pos_weights * pos_logsig).sum() / float(pos_denom)) * float(pos_weight)
    else:
        pos_loss_ungated = torch.tensor(0.0, dtype=torch.float32)
    pos_loss = pos_loss_ungated * step_gate

    neg_logits = logits_t[neg_mask]
    neg_logsig = -F.logsigmoid(-neg_logits)
    neg_denom = max(1, int(neg_mask.sum().item()))
    if neg_logits.numel() > 0:
        neg_loss_ungated = neg_logsig.sum() / float(neg_denom)
    else:
        neg_loss_ungated = torch.tensor(0.0, dtype=torch.float32)
    neg_loss = neg_loss_ungated * step_gate

    pos_per_token = []
    neg_per_token = []
    for idx, (logit_val, mark, w) in enumerate(zip(logits, marks, weights)):
        if mark == "pos":
            raw_bce = float((-F.logsigmoid(torch.tensor(logit_val))).item())
            weighted_raw = raw_bce * float(w)
            after_pos_denom = weighted_raw / float(pos_denom)
            after_pos_weight = after_pos_denom * float(pos_weight)
            final_contribution = after_pos_weight * step_gate
            pos_per_token.append(
                {
                    "index": idx,
                    "mark": mark,
                    "logit": float(logit_val),
                    "weight": float(w),
                    "raw_bce_term": raw_bce,
                    "weighted_raw_bce_term": weighted_raw,
                    "after_pos_denom": after_pos_denom,
                    "after_pos_weight": after_pos_weight,
                    "final_loss_contribution": final_contribution,
                }
            )
        elif mark == "neg":
            kept = bool(neg_mask[idx].item())
            raw_bce = float((-F.logsigmoid(torch.tensor(-logit_val))).item())
            after_neg_denom = (raw_bce / float(neg_denom)) if kept else 0.0
            final_contribution = after_neg_denom * step_gate
            neg_per_token.append(
                {
                    "index": idx,
                    "mark": mark,
                    "logit": float(logit_val),
                    "kept_for_neg_loss": kept,
                    "raw_bce_term": raw_bce,
                    "after_neg_denom": after_neg_denom,
                    "final_loss_contribution": final_contribution,
                }
            )
        else:
            # ignore
            pass

    return {
        "assumptions": {
            "type_loss_included": False,
            "all_tokens_assumed_vec": True,
            "gt_is_vec": bool(gt_is_vec),
        },
        "config": {
            "pos_weight": float(pos_weight),
            "apply_half_negative_sampling": bool(apply_half_negative_sampling),
            "seed": int(seed),
        },
        "counts": {
            "num_tokens": len(logits),
            "num_pos": int(pos_mask.sum().item()),
            "num_neg_candidates": int(neg_candidate_mask.sum().item()),
            "num_neg_kept": int(neg_mask.sum().item()),
            "pos_denom": int(pos_denom),
            "neg_denom": int(neg_denom),
        },
        "loss": {
            "pos_loss": float(pos_loss.item()),
            "neg_loss": float(neg_loss.item()),
            "pos_neg_total": float((pos_loss + neg_loss).item()),
            "pos_loss_before_vec_gate": float(pos_loss_ungated.item()),
            "neg_loss_before_vec_gate": float(neg_loss_ungated.item()),
        },
        "positive_terms": pos_per_token,
        "negative_terms": neg_per_token,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute positive/negative multi-label BCE terms (type loss excluded) "
            "from logits + mark labels."
        )
    )

    single_group = parser.add_argument_group("Single-token mode")
    single_group.add_argument("--logit", type=float, default=None, help="Single logit value.")
    single_group.add_argument(
        "--mark",
        type=str,
        default=None,
        help="Single mark: pos, neg, ignore.",
    )
    single_group.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Single-token positive weight (used only when --mark=pos).",
    )
    single_group.add_argument(
        "--neg_keep",
        type=int,
        default=1,
        choices=[0, 1],
        help="Single-token negative keep flag when --mark=neg and --neg_keep_mask is not used.",
    )

    list_group = parser.add_argument_group("Multi-token mode")
    list_group.add_argument(
        "--logits",
        type=str,
        default=None,
        help="Comma-separated logits, e.g. '2.0,-1.0,0.5'.",
    )
    list_group.add_argument(
        "--marks",
        type=str,
        default=None,
        help="Comma-separated marks aligned to logits: pos/neg/ignore.",
    )
    list_group.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated positive weights aligned to logits. Defaults to 1.0 for all.",
    )
    list_group.add_argument(
        "--neg_keep_mask",
        type=str,
        default=None,
        help="Optional comma-separated 0/1 keep mask aligned to logits for negatives.",
    )

    parser.add_argument("--pos_weight", type=float, default=100.0)
    parser.add_argument(
        "--apply_half_negative_sampling",
        action="store_true",
        help="Apply Bernoulli(0.5) keep on negative candidates, like training code.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gt_is_vec",
        action="store_true",
        default=True,
        help="Assume GT token is VEC (default true).",
    )
    parser.add_argument(
        "--gt_is_not_vec",
        action="store_true",
        help="Override and force GT token as non-VEC (will zero pos/neg terms).",
    )
    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()
    logits, marks, weights, keep_mask = _build_inputs(args)

    gt_is_vec = True
    if args.gt_is_not_vec:
        gt_is_vec = False

    result = compute_pos_neg_multilabel_bce(
        logits=logits,
        marks=marks,
        weights=weights,
        pos_weight=args.pos_weight,
        apply_half_negative_sampling=args.apply_half_negative_sampling,
        neg_keep_mask=keep_mask,
        seed=args.seed,
        gt_is_vec=gt_is_vec,
    )

    text = json.dumps(result, indent=2)
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(text + "\n")
        print(f"saved_json: {args.output_json}")
    else:
        print(text)


if __name__ == "__main__":
    main()
