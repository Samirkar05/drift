import argparse
import os

import torch
import torch.nn.functional as F

from src.datasets.registry import get_dataset
from src.heads import get_classification_head


def _load_trained_head(head_path):
    if not os.path.isfile(head_path):
        raise FileNotFoundError(f"Trained head checkpoint not found: {head_path}")
    return torch.load(head_path, map_location="cpu")


def _effective_class_embeddings(head):
    class_embeds = head.weight.detach().cpu()
    if hasattr(head, "drift"):
        class_embeds = class_embeds + head.drift.detach().cpu().unsqueeze(0)
    return class_embeds


def _summary(name, norms):
    print(
        f"{name}: mean={norms.mean().item():.6f}, std={norms.std(unbiased=False).item():.6f}, "
        f"min={norms.min().item():.6f}, max={norms.max().item():.6f}"
    )


def _top_changes(classnames, normal_norms, trained_norms, top_k):
    delta = trained_norms - normal_norms
    abs_delta = delta.abs()
    k = min(top_k, abs_delta.numel())
    values, idx = torch.topk(abs_delta, k=k, largest=True)
    print(f"\nTop-{k} classes by |norm change|:")
    for rank, (class_idx, magnitude) in enumerate(zip(idx.tolist(), values.tolist()), start=1):
        label = classnames[class_idx] if class_idx < len(classnames) else f"class_{class_idx}"
        print(
            f"{rank:2d}. {label}: "
            f"normal={normal_norms[class_idx].item():.6f}, "
            f"trained={trained_norms[class_idx].item():.6f}, "
            f"delta={delta[class_idx].item():+.6f}, "
            f"|delta|={magnitude:.6f}"
        )


def _compare_dataset(dataset_name, save_root, trained_head_path, args):
    normal_head = get_classification_head(args, dataset_name, drift=False).cpu()
    trained_head = _load_trained_head(trained_head_path).cpu()

    normal_embeds = _effective_class_embeddings(normal_head)
    trained_embeds = _effective_class_embeddings(trained_head)

    if normal_embeds.shape != trained_embeds.shape:
        raise ValueError(
            f"Shape mismatch for dataset {dataset_name}: normal={tuple(normal_embeds.shape)}, "
            f"trained={tuple(trained_embeds.shape)}"
        )

    normal_norms = torch.linalg.norm(normal_embeds, dim=1)
    trained_norms = torch.linalg.norm(trained_embeds, dim=1)
    delta = trained_norms - normal_norms
    cosine = F.cosine_similarity(normal_embeds, trained_embeds, dim=1)

    return {
        "dataset": dataset_name,
        "normal_head_path": os.path.join(save_root, f"head_{dataset_name}.pt"),
        "trained_head_path": trained_head_path,
        "normal_norms": normal_norms,
        "trained_norms": trained_norms,
        "delta_norms": delta,
        "cosine": cosine,
    }


def _print_all_datasets_table(rows):
    headers = [
        "Dataset",
        "NormalMean",
        "DriftMean",
        "DeltaMean",
        "AbsDeltaMean",
        "CosineMean",
    ]
    table = []
    for row in rows:
        table.append(
            [
                row["dataset"],
                f"{row['normal_norms'].mean().item():.6f}",
                f"{row['trained_norms'].mean().item():.6f}",
                f"{row['delta_norms'].mean().item():+.6f}",
                f"{row['delta_norms'].abs().mean().item():.6f}",
                f"{row['cosine'].mean().item():.6f}",
            ]
        )

    widths = [len(h) for h in headers]
    for r in table:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt_line(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_line(headers))
    print("-+-".join("-" * w for w in widths))
    for r in table:
        print(fmt_line(r))


def _discover_datasets_from_checkpoints(save_root):
    discovered = []
    if not os.path.isdir(save_root):
        return discovered
    for entry in sorted(os.listdir(save_root)):
        head_path = os.path.join(save_root, entry, "trained_drift_head.pt")
        if os.path.isfile(head_path):
            discovered.append(entry)
    return discovered


def main():
    parser = argparse.ArgumentParser(description="Compare class embedding norms between normal and trained heads.")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name, e.g. MNISTVal")
    parser.add_argument(
        "--datasets",
        type=lambda x: x.split(","),
        default=None,
        help="Comma-separated datasets for table mode, e.g. MNISTVal,EuroSATVal",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Report mean-norm comparison table across datasets.",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Checkpoint root for this model (contains head_<dataset>.pt and dataset folder).",
    )
    parser.add_argument(
        "--trained-head-path",
        type=str,
        default=None,
        help="Path to trained head checkpoint. Defaults to {save}/{dataset}/trained_drift_head.pt",
    )
    parser.add_argument("--data-location", type=str, default=os.path.expanduser("~/data"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.all_datasets:
        datasets = args.datasets if args.datasets is not None else _discover_datasets_from_checkpoints(args.save)
        if not datasets:
            raise ValueError(
                "No datasets found for --all-datasets. "
                "Pass --datasets or ensure {save}/{dataset}/trained_drift_head.pt exists."
            )

        rows = []
        for dataset_name in datasets:
            trained_head_path = os.path.join(args.save, dataset_name, "trained_drift_head.pt")
            rows.append(_compare_dataset(dataset_name, args.save, trained_head_path, args))

        print(f"Model: {args.model}")
        print(f"Checkpoint root: {args.save}")
        print()
        _print_all_datasets_table(rows)
        return

    if args.dataset is None:
        raise ValueError("Please provide --dataset for single-dataset mode, or use --all-datasets.")

    if args.trained_head_path is None:
        args.trained_head_path = os.path.join(args.save, args.dataset, "trained_drift_head.pt")

    row = _compare_dataset(args.dataset, args.save, args.trained_head_path, args)
    normal_norms = row["normal_norms"]
    trained_norms = row["trained_norms"]
    delta = row["delta_norms"]

    print(f"Dataset: {args.dataset}")
    print(f"Normal head path: {row['normal_head_path']}")
    print(f"Trained head path: {row['trained_head_path']}")
    _summary("Normal norms", normal_norms)
    _summary("Trained norms", trained_norms)
    _summary("Delta (trained - normal)", delta)

    _summary("Cosine(normal, trained)", row["cosine"])

    classnames = get_dataset(args.dataset, None, location=args.data_location).classnames
    _top_changes(classnames, normal_norms, trained_norms, top_k=args.top_k)


if __name__ == "__main__":
    main()
