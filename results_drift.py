import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple


def _extract_top1(result_file: Path, dataset: str) -> Optional[float]:
    """
    Read a result file (JSON lines) and extract '<dataset>:top1'.
    Falls back to the first '*:top1' key if the exact key is absent.
    """
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

    # Typical local path in this workspace.
    local_default = Path.home() / "results"
    if local_default.exists():
        return local_default

    # Fallback used in training/eval scripts.
    return Path("/data/139-1/users/selkarrat/results")


def _collect_rows(model_dir: Path) -> List[Tuple[str, float, float, float]]:
    drift_dir = model_dir / "drift_eval"
    rows = []

    for drift_file in sorted(drift_dir.glob("results_drift_*")):
        dataset = drift_file.name.replace("results_drift_", "", 1)
        zeroshot_file = model_dir / f"results_zeroshot_{dataset}"

        drift_top1 = _extract_top1(drift_file, dataset)
        zeroshot_top1 = _extract_top1(zeroshot_file, dataset)

        if drift_top1 is None or zeroshot_top1 is None:
            continue

        delta = drift_top1 - zeroshot_top1
        rows.append((dataset, zeroshot_top1, drift_top1, delta))

    return rows


def _print_table(rows: List[Tuple[str, float, float, float]]) -> None:
    if not rows:
        print("No comparable drift/zeroshot results found.")
        return

    rows = sorted(rows, key=lambda x: x[0])

    header = f"{'Dataset':<12} {'Zeroshot':>10} {'Drift':>10} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    for dataset, zeroshot, drift, delta in rows:
        print(f"{dataset:<12} {zeroshot:>10.4f} {drift:>10.4f} {delta:>+10.4f}")

    avg_zero = sum(r[1] for r in rows) / len(rows)
    avg_drift = sum(r[2] for r in rows) / len(rows)
    avg_delta = sum(r[3] for r in rows) / len(rows)
    print("-" * len(header))
    print(f"{'AVG':<12} {avg_zero:>10.4f} {avg_drift:>10.4f} {avg_delta:>+10.4f}")


def _write_csv(rows: List[Tuple[str, float, float, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "zeroshot_top1", "drift_top1", "delta"])
        for dataset, zeroshot, drift, delta in sorted(rows, key=lambda x: x[0]):
            writer.writerow([dataset, zeroshot, drift, delta])
    print(f"\nSaved comparison CSV to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare drift vs zeroshot top1 results for a model."
    )
    parser.add_argument("--model", default="ViT-B-32", help="Model name.")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Root folder containing model result folders (default: ~/results).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save CSV output.",
    )
    args = parser.parse_args()

    results_root = _resolve_results_root(args.results_root)
    model_dir = results_root / args.model

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model results folder not found: {model_dir}")

    rows = _collect_rows(model_dir)
    _print_table(rows)

    if args.csv:
        _write_csv(rows, Path(args.csv).expanduser())


if __name__ == "__main__":
    main()
