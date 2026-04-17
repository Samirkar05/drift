import json
import glob
import os
import re
from statistics import mean


DEFAULT_RESULTS_DIR = "/data/139-1/users/selkarrat/results"
DEFAULT_ACCURACIES_JSON = os.path.join(DEFAULT_RESULTS_DIR, "all_accuracies.json")


def run_name_from_path(path: str) -> str:
    # "results_finetuned_Cars" -> "finetuned_Cars"
    filename = os.path.basename(path)
    if filename.startswith("results_"):
        return filename[len("results_") :]
    return filename

def extract_top1_scores(d: dict) -> dict:
    # {"MNIST:top1": 0.51, ...} -> {"MNIST": 0.51, ...}
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and k.endswith(":top1") and isinstance(v, (int, float)):
            out[k.split(":", 1)[0]] = float(v)
    return out


def save_accuracies_json(accuracies: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(accuracies, f, indent=2, sort_keys=True)


def read_accuracies_json(input_path: str = DEFAULT_ACCURACIES_JSON) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results(results_dir: str = DEFAULT_RESULTS_DIR) -> dict:
    # results[run][dataset] = score
    results = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "results_*"))):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        results[run_name_from_path(path)] = extract_top1_scores(d)
    return results


def finetuned_dataset_from_run(run_name: str) -> str | None:
    if not run_name.startswith("finetuned_"):
        return None
    return run_name[len("finetuned_") :]


def _to_percent_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 4)


def normalize_run_key(run_name: str) -> str:
    # Convert run names like "Iso-C" / "task-arithmetic" into stable JSON keys.
    key = re.sub(r"[^a-zA-Z0-9]+", "_", run_name.strip().lower())
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def print_results_table(results: dict) -> None:
    # ---- simple table (stdout) ----
    runs = sorted(results.keys())
    datasets = sorted({ds for r in runs for ds in results[r].keys()})

    col0 = max(12, max((len(r) for r in runs), default=12))
    colw = max(7, max((len(ds) for ds in datasets), default=7))

    # header
    print(f"{'run':<{col0}} " + " ".join(f"{ds:<{colw}}" for ds in datasets))
    print(f"{'-'*col0} " + " ".join("-" * colw for _ in datasets))

    # rows
    for run in runs:
        row = [f"{run:<{col0}}"]
        for ds in datasets:
            v = results[run].get(ds, None)
            row.append(f"{'' if v is None else f'{v:.4f}':<{colw}}")
        print(" ".join(row))


def finetuned_own_score(run_name: str, scores: dict) -> float | None:
    # run_name like "finetuned_Cars" -> returns score on "Cars"
    dataset = finetuned_dataset_from_run(run_name)
    if dataset is None:
        return None
    return scores.get(dataset)


def build_summary(results: dict) -> dict:
    # collect finetuned self-scores and average them
    finetuned_self = {}
    for run_name, scores in results.items():
        v = finetuned_own_score(run_name, scores)
        if v is not None:
            finetuned_self[run_name] = v

    summary = {}
    if finetuned_self:
        summary["finetuned_self_scores"] = finetuned_self
        summary["average_absolute_accuracy_finetuned"] = mean(finetuned_self.values())
    else:
        summary["finetuned_self_scores"] = {}
        summary["average_absolute_accuracy_finetuned"] = None

    base_datasets = ["MNIST", "SVHN", "Cars", "SUN397", "RESISC45", "GTSRB", "EuroSAT", "DTD"]
    average_by_run = {}
    for key in results.keys():
        if key not in base_datasets and results[key]:
            average_by_run[key] = mean(results[key].values())
    summary["average_absolute_accuracy_by_run"] = average_by_run
    return summary


def build_accuracies_payload(results: dict) -> dict:
    finetuned_self_scores = []
    diagonal_by_dataset = {}
    for run_name, scores in results.items():
        dataset = finetuned_dataset_from_run(run_name)
        v = finetuned_own_score(run_name, scores)
        if v is not None:
            finetuned_self_scores.append(v)
            if dataset is not None:
                diagonal_by_dataset[dataset] = _to_percent_float(v)

    payload = {
        "database_finetuned": _to_percent_float(mean(finetuned_self_scores)) if finetuned_self_scores else None,
    }

    # For every non-finetuned run in the folder:
    # 1) store its average accuracy under a normalized run key
    # 2) store per-dataset entries as "<normalized_run_key>_<dataset>"
    non_finetuned_runs = {
        run_name: scores
        for run_name, scores in results.items()
        if finetuned_dataset_from_run(run_name) is None
    }
    for run_name, scores in non_finetuned_runs.items():
        run_key = normalize_run_key(run_name)
        payload[run_key] = _to_percent_float(mean(scores.values())) if scores else None
        for dataset, score in scores.items():
            payload[f"{run_key}_{dataset}"] = _to_percent_float(score)

    payload.update(diagonal_by_dataset)
    return payload


def print_summary(results: dict) -> None:
    payload = build_accuracies_payload(results)
    print("Simple accuracies:", payload)


def main() -> None:
    results = collect_results(DEFAULT_RESULTS_DIR)
    payload = build_accuracies_payload(results)
    save_accuracies_json(payload, DEFAULT_ACCURACIES_JSON)
    print(f"Saved all accuracies JSON to: {DEFAULT_ACCURACIES_JSON}")
    print_results_table(results)
    print_summary(results)


if __name__ == "__main__":
    main()
