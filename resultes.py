import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple


def _extract_top1(result_file: Path, dataset: str) -> Optional[float]:
    if not result_file.is_file():
        return None

    target_key = f"{dataset}:top1"
    fallback_value = None

    with result_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if target_key in row:
                return float(row[target_key])

            for key, value in row.items():
                if key.endswith(":top1"):
                    fallback_value = float(value)

    return fallback_value


def _resolve_results_root(user_path: Optional[str]) -> Path:
    if user_path:
        return Path(user_path).expanduser()

    local_default = Path.home() / "results"
    if local_default.exists():
        return local_default

    return Path("/data/139-1/users/selkarrat/results")


def _legacy_tag_aliases(tag: str) -> List[str]:
    aliases = [tag]
    legacy_map = {
        "adapter_zeroshot": "mlp",
        "merged_adapter_zeroshot": "mlp_merging",
        "adapter_adapted_finetuned": "adapted_finetuned",
    }
    if tag in legacy_map:
        aliases.append(legacy_map[tag])
    # Accept both naming variants for the baseline image encoder.
    if tag.endswith("_normal"):
        aliases.append(tag.replace("_normal", "_zeroshot"))
    elif tag.endswith("_zeroshot"):
        aliases.append(tag.replace("_zeroshot", "_normal"))
    return aliases


def _collect_rows(
    eval_dir: Path, model_dir: Path, run_tag: str, baseline_tag: str
) -> List[Tuple[str, float, float, float]]:
    rows = []
    seen = set()

    run_tags = _legacy_tag_aliases(run_tag)
    baseline_tags = _legacy_tag_aliases(baseline_tag)

    # legacy baseline files live in model root (e.g., results_zeroshot_<dataset>)
    if baseline_tag == "normal_zeroshot":
        baseline_tags.append("zeroshot")

    for current_run_tag in run_tags:
        prefix = f"results_{current_run_tag}_"
        for result_file in sorted(eval_dir.glob(f"{prefix}*")):
            dataset = result_file.name.replace(prefix, "", 1)
            if dataset in seen:
                continue

            baseline_top1 = None
            for current_baseline_tag in baseline_tags:
                baseline_file_eval_dir = eval_dir / f"results_{current_baseline_tag}_{dataset}"
                baseline_file_model_dir = model_dir / f"results_{current_baseline_tag}_{dataset}"

                baseline_top1 = _extract_top1(baseline_file_eval_dir, dataset)
                if baseline_top1 is None:
                    baseline_top1 = _extract_top1(baseline_file_model_dir, dataset)
                if baseline_top1 is not None:
                    break

            run_top1 = _extract_top1(result_file, dataset)
            if run_top1 is None or baseline_top1 is None:
                continue

            delta = run_top1 - baseline_top1
            rows.append((dataset, baseline_top1, run_top1, delta))
            seen.add(dataset)

    return rows


def _print_table(rows: List[Tuple[str, float, float, float]], run_label: str, baseline_label: str) -> None:
    if not rows:
        print("No comparable results found.")
        return

    rows = sorted(rows, key=lambda x: x[0])
    header = f"{'Dataset':<12} {baseline_label:>12} {run_label:>14} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    for dataset, baseline, run_value, delta in rows:
        print(f"{dataset:<12} {baseline:>12.4f} {run_value:>14.4f} {delta:>+10.4f}")

    avg_base = sum(r[1] for r in rows) / len(rows)
    avg_run = sum(r[2] for r in rows) / len(rows)
    avg_delta = sum(r[3] for r in rows) / len(rows)
    print("-" * len(header))
    print(f"{'AVG':<12} {avg_base:>12.4f} {avg_run:>14.4f} {avg_delta:>+10.4f}")


def _write_csv(
    rows: List[Tuple[str, float, float, float]],
    output: Path,
    baseline_tag: str,
    run_tag: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", f"{baseline_tag}_top1", f"{run_tag}_top1", "delta"])
        for dataset, baseline, run_value, delta in sorted(rows, key=lambda x: x[0]):
            writer.writerow([dataset, baseline, run_value, delta])
    print(f"\nSaved comparison CSV to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare evaluate.py runs using results_{head}_{encoder}_{dataset} files."
    )
    parser.add_argument("--model", default="ViT-B-32", help="Model name.")
    parser.add_argument(
        "--head",
        choices=["normal", "prompt_head", "adapter", "merged_adapter", "drift_per_task", "drift_per_class"],
        required=True,
    )
    parser.add_argument(
        "--encoder",
        choices=[
            "normal",
            "zeroshot",
            "finetuned",
            "adapted_finetuned",
            "adapted_finetuned_drift_per_task",
            "adapted_finetuned_drift_per_class",
            "merged_adapted_finetuned",
        ],
        required=True,
    )
    parser.add_argument(
        "--baseline-head",
        choices=["normal", "prompt_head", "adapter", "merged_adapter", "drift_per_task", "drift_per_class"],
        default="normal",
        help="Baseline run head (default: normal).",
    )
    parser.add_argument(
        "--baseline-encoder",
        choices=[
            "normal",
            "zeroshot",
            "finetuned",
            "adapted_finetuned",
            "adapted_finetuned_drift_per_task",
            "adapted_finetuned_drift_per_class",
            "merged_adapted_finetuned",
        ],
        default="zeroshot",
        help="Baseline run encoder (default: zeroshot).",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Root folder containing model result folders (default: ~/results).",
    )
    parser.add_argument("--csv", default=None, help="Optional path to save CSV output.")
    args = parser.parse_args()

    run_tag = f"{args.head}_{args.encoder}"
    baseline_tag = f"{args.baseline_head}_{args.baseline_encoder}"

    results_root = _resolve_results_root(args.results_root)
    model_dir = results_root / args.model
    eval_dir = model_dir / "drift_eval"

    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation results folder not found: {eval_dir}")

    rows = _collect_rows(eval_dir, model_dir, run_tag, baseline_tag)
    _print_table(rows, run_tag, baseline_tag)

    if args.csv:
        _write_csv(rows, Path(args.csv).expanduser(), baseline_tag, run_tag)


if __name__ == "__main__":
    main()
