import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset

DEFAULT_DATASETS = ["GTSRB", "MNIST", "DTD", "SVHN", "EuroSAT", "RESISC45", "Cars", "SUN397"]
ENCODER_CHOICES = [
    "finetuned",
    "adapted_finetuned",
    "adapted_finetuned_drift_per_task",
    "adapted_finetuned_drift_per_class",
    "adapted_finetuned_prompt_head",
]


def _ensure_train_dataset_name(dataset_name: str) -> str:
    return dataset_name if dataset_name.endswith("Val") else f"{dataset_name}Val"


def _base_dataset_name(dataset_name: str) -> str:
    return dataset_name[:-3] if dataset_name.endswith("Val") else dataset_name


def _encoder_checkpoint_filename(encoder_mode: str) -> str:
    mapping = {
        "finetuned": "finetuned.pt",
        "adapted_finetuned": "adapted_finetuned.pt",
        "adapted_finetuned_drift_per_task": "adapted_finetuned_drift_per_task.pt",
        "adapted_finetuned_drift_per_class": "adapted_finetuned_drift_per_class.pt",
        "adapted_finetuned_prompt_head": "adapted_finetuned_prompt_head.pt",
    }
    if encoder_mode not in mapping:
        raise ValueError(f"Unsupported encoder mode: {encoder_mode}")
    return mapping[encoder_mode]


def _load_encoder_checkpoint(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def _mean_feature_cosine_on_dataset(
    pretrained_encoder,
    adapted_encoder,
    dataset_name,
    data_location,
    batch_size,
    device,
    show_progress=False,
    progress_desc="",
    max_batches=-1,
) -> float:
    # Use the pretrained encoder preprocessing so both models see identical inputs.
    dataset = get_dataset(
        dataset_name,
        pretrained_encoder.val_preprocess,
        location=data_location,
        batch_size=batch_size,
    )
    loader_args = argparse.Namespace(batch_size=batch_size, device=device)
    dataloader = get_dataloader(dataset, is_train=False, args=loader_args, image_encoder=None)

    pretrained_encoder = pretrained_encoder.to(device).eval()
    adapted_encoder = adapted_encoder.to(device).eval()

    sim_sum = 0.0
    n = 0
    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc=progress_desc, leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(device)

            z_pre = pretrained_encoder(x)
            z_adapt = adapted_encoder(x)

            z_pre = F.normalize(z_pre, dim=-1)
            z_adapt = F.normalize(z_adapt, dim=-1)
            batch_sim = (z_pre * z_adapt).sum(dim=-1)

            sim_sum += float(batch_sim.sum().item())
            n += int(batch_sim.numel())

    if n == 0:
        raise RuntimeError(f"No samples found for dataset split: {dataset_name}")
    return sim_sum / n


def _compute_dataset_cosines(
    model: str,
    checkpoint_root: str,
    data_location: str,
    dataset: str,
    batch_size: int,
    device: str,
    target_encoder: str,
    split: str,
    show_progress: bool,
    max_batches: int,
):
    train_dataset = _ensure_train_dataset_name(dataset)
    eval_dataset = train_dataset if split == "validation" else _base_dataset_name(dataset)
    ckpt_dir = os.path.join(checkpoint_root, model, train_dataset)

    zs_path = os.path.join(ckpt_dir, "zeroshot.pt")
    ft_path = os.path.join(ckpt_dir, "finetuned.pt")
    target_path = os.path.join(ckpt_dir, _encoder_checkpoint_filename(target_encoder))

    zeroshot_encoder = _load_encoder_checkpoint(zs_path)
    finetuned_encoder = _load_encoder_checkpoint(ft_path)
    target_encoder_model = _load_encoder_checkpoint(target_path)
    print(f"[{dataset}] Loaded encoders. Evaluating split={eval_dataset} ...", flush=True)

    sim_ft = _mean_feature_cosine_on_dataset(
        zeroshot_encoder,
        finetuned_encoder,
        dataset_name=eval_dataset,
        data_location=data_location,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
        progress_desc=f"{dataset}: finetuned",
        max_batches=max_batches,
    )
    print(f"[{dataset}] finetuned similarity ready: {sim_ft:.6f}", flush=True)
    sim_target = _mean_feature_cosine_on_dataset(
        zeroshot_encoder,
        target_encoder_model,
        dataset_name=eval_dataset,
        data_location=data_location,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
        progress_desc=f"{dataset}: {target_encoder}",
        max_batches=max_batches,
    )
    print(f"[{dataset}] {target_encoder} similarity ready: {sim_target:.6f}", flush=True)
    return sim_ft, sim_target


def _plot_dumbbell(rows: List[Tuple[str, float, float]], out_path: str, title: str, right_label: str):
    labels = [r[0] for r in rows] + ["AVG"]
    left_vals = [r[1] for r in rows]
    right_vals = [r[2] for r in rows]
    left_vals.append(sum(left_vals) / len(left_vals))
    right_vals.append(sum(right_vals) / len(right_vals))

    y_pos = list(range(len(labels)))
    avg_idx = len(labels) - 1

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.set_facecolor("#efefef")
    fig.patch.set_facecolor("#efefef")

    for i, (l, r) in enumerate(zip(left_vals, right_vals)):
        if i == avg_idx:
            ax.plot([l, r], [i, i], color="#333333", linewidth=2.0, zorder=1)
        else:
            ax.plot([l, r], [i, i], color="#b5b5b5", linestyle="--", linewidth=2.0, zorder=1)

        delta_pct = ((r - l) / max(abs(l), 1e-12)) * 100.0
        xm = (l + r) / 2.0
        ax.text(
            xm,
            i - 0.16,
            f"{delta_pct:+.0f}%",
            color="#3f8c3f",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            zorder=4,
        )

    ax.scatter(left_vals, y_pos, s=120, color="#4f7eb9", edgecolor="#1d1d1d", zorder=3, label="FT")
    ax.scatter(
        right_vals,
        y_pos,
        s=95,
        marker="D",
        color="#e08a36",
        edgecolor="#1d1d1d",
        zorder=3,
        label=right_label,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=14)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.invert_yaxis()

    all_vals = left_vals + right_vals
    x_min = min(all_vals)
    x_max = max(all_vals)
    pad = max(0.01, 0.08 * (x_max - x_min if x_max > x_min else 1.0))
    ax.set_xlim(max(-1.0, x_min - pad), min(1.0, x_max + pad))

    ax.set_xlabel("Cosine similarity", fontsize=16)
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(axis="x", alpha=0.25)

    legend = ax.legend(loc="lower left", frameon=True, facecolor="#efefef", edgecolor="#aaaaaa", fontsize=10)
    for text in legend.get_texts():
        text.set_fontsize(15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _print_points(rows: List[Tuple[str, float, float]], right_label: str):
    print("\nPoint values (cosine similarity):")
    header = f"{'Dataset':<10} {'Finetuned':>12} {right_label:>34}"
    print(header)
    print("-" * len(header))
    for ds, left, right in rows:
        print(f"{ds:<10} {left:>12.6f} {right:>34.6f}")
    avg_left = sum(r[1] for r in rows) / len(rows)
    avg_right = sum(r[2] for r in rows) / len(rows)
    print("-" * len(header))
    print(f"{'AVG':<10} {avg_left:>12.6f} {avg_right:>34.6f}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a dumbbell chart of feature-level cosine similarity averaged over images: "
            "E_x[cos(z_pre(x), z_i(x))], with blue=finetuned and orange=target encoder."
        )
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument(
        "--datasets",
        type=lambda x: [part.strip() for part in x.split(",") if part.strip()],
        default=None,
        help="Comma-separated base datasets.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="/data/139-1/users/selkarrat/checkpoints")
    parser.add_argument("--data-location", type=str, default="/data/139-1/datasets/merging")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--split", choices=["validation", "eval"], default="eval")
    parser.add_argument(
        "--target-encoder",
        choices=ENCODER_CHOICES,
        default="adapted_finetuned",
        help="Right-side encoder; blue circles remain zeroshot-vs-finetuned.",
    )
    parser.add_argument("--output", type=str, default="./plots/cosine_similarity_encoder_dumbbell.png")
    parser.add_argument("--title", type=str, default=None, help="Optional custom title.")
    parser.add_argument("--show-progress", action="store_true", help="Show progress bars.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=-1,
        help="Debug option: if > 0, process only this many batches per similarity pass.",
    )
    args = parser.parse_args()

    method_label_map = {
        "finetuned": "FT",
        "adapted_finetuned": "Adapted FT",
        "adapted_finetuned_drift_per_task": "DT",
        "adapted_finetuned_drift_per_class": "DC",
        "adapted_finetuned_prompt_head": "Prompt Head",
    }
    right_label = method_label_map.get(args.target_encoder, args.target_encoder)
    plot_title = args.title if args.title else f"FT vs {right_label}"

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    rows: List[Tuple[str, float, float]] = []
    dataset_iter = datasets
    if args.show_progress:
        dataset_iter = tqdm(datasets, desc="Datasets", leave=True)
    for dataset in dataset_iter:
        base_dataset = _base_dataset_name(dataset)
        try:
            sim_ft, sim_target = _compute_dataset_cosines(
                model=args.model,
                checkpoint_root=args.checkpoint_root,
                data_location=args.data_location,
                dataset=base_dataset,
                batch_size=args.batch_size,
                device=args.device,
                target_encoder=args.target_encoder,
                split=args.split,
                show_progress=args.show_progress,
                max_batches=args.max_batches,
            )
            rows.append((base_dataset, sim_ft, sim_target))
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Skipping {base_dataset}: {exc}")

    if len(rows) == 0:
        raise RuntimeError("No dataset rows available to plot.")

    _print_points(rows, right_label=right_label)
    _plot_dumbbell(rows, out_path=args.output, title=plot_title, right_label=right_label)


if __name__ == "__main__":
    main()
