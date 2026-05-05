import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder
from compute_drift_distance_metrics import _load_prompt_head

DEFAULT_DATASETS = ["Cars", "SVHN", "MNIST", "SUN397", "RESISC45", "GTSRB", "EuroSAT", "DTD"]


def _ensure_train_dataset_name(dataset_name: str) -> str:
    return dataset_name if dataset_name.endswith("Val") else f"{dataset_name}Val"


def _base_dataset_name(dataset_name: str) -> str:
    return dataset_name[:-3] if dataset_name.endswith("Val") else dataset_name


def _effective_text_embeddings(head):
    embeds = head.weight.detach().cpu()
    if hasattr(head, "drift"):
        drift = head.drift.detach().cpu()
        if drift.ndim == 1:
            embeds = embeds + drift.unsqueeze(0)
        else:
            embeds = embeds + drift
    return F.normalize(embeds, dim=-1)


def _load_normal_head(model_name, checkpoint_root, data_location, device, train_dataset_name):
    model_ckpt_dir = os.path.join(checkpoint_root, model_name)
    explicit_path = os.path.join(model_ckpt_dir, f"head_{train_dataset_name}.pt")
    if os.path.isfile(explicit_path):
        return torch.load(explicit_path, map_location="cpu")

    head_args = SimpleNamespace(
        save=model_ckpt_dir,
        model=model_name,
        data_location=data_location,
        device=device,
    )
    return get_classification_head(head_args, train_dataset_name, drift=False).cpu()


def _load_trained_drift_head(model_name, checkpoint_root, train_dataset_name, mode):
    dataset_ckpt_dir = os.path.join(checkpoint_root, model_name, train_dataset_name)
    if mode == "per_task":
        filename = "trained_drift_head.pt"
    elif mode == "per_class":
        filename = "trained_drift_head_per_class.pt"
    else:
        raise ValueError(f"Unsupported drift mode: {mode}")

    path = os.path.join(dataset_ckpt_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {mode} drift head checkpoint: {path}\n"
            f"Train it first with train_drift.py --head-type {mode}"
        )
    return torch.load(path, map_location="cpu")


def _load_image_encoder(model_name, checkpoint_root, train_dataset_name):
    dataset_ckpt_dir = os.path.join(checkpoint_root, model_name, train_dataset_name)
    zs_path = os.path.join(dataset_ckpt_dir, "zeroshot.pt")
    if os.path.isfile(zs_path):
        return torch.load(zs_path, map_location="cpu")
    encoder_args = SimpleNamespace(model=model_name, cache_dir=None)
    return ImageEncoder(encoder_args, keep_lang=False)


def _collect_visual_class_centroids(
    image_encoder,
    dataset_name,
    data_location,
    device,
    batch_size,
):
    dataset = get_dataset(
        dataset_name,
        image_encoder.val_preprocess,
        location=data_location,
        batch_size=batch_size,
    )
    loader_args = SimpleNamespace(batch_size=batch_size, device=device)
    dataloader = get_dataloader(dataset, is_train=False, args=loader_args, image_encoder=None)

    image_encoder = image_encoder.to(device)
    image_encoder.eval()

    class_sums = None
    class_counts = None
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(device)
            y = batch["labels"].cpu()

            feats = image_encoder(x)
            feats = F.normalize(feats, dim=-1).cpu()

            if class_sums is None:
                num_classes = len(dataset.classnames)
                dim = feats.shape[1]
                class_sums = torch.zeros(num_classes, dim)
                class_counts = torch.zeros(num_classes, dtype=torch.long)

            for i in range(feats.shape[0]):
                cls = int(y[i].item())
                class_sums[cls] += feats[i]
                class_counts[cls] += 1

    if class_sums is None:
        raise RuntimeError("No image embeddings collected for visual centroids.")

    valid_mask = class_counts > 0
    if not valid_mask.all():
        missing = [dataset.classnames[i] for i in (~valid_mask).nonzero(as_tuple=False).flatten().tolist()]
        raise RuntimeError(
            "Some classes have zero validation samples; cannot align class-pair matrices exactly. "
            f"Missing classes: {', '.join(missing)}"
        )

    centroids = class_sums / class_counts.unsqueeze(1)
    centroids = F.normalize(centroids, dim=-1)
    return centroids


def _angles_from_similarity(sim, units):
    sim = sim.clamp(min=-1.0, max=1.0)
    angles = torch.acos(sim)
    if units == "degrees":
        angles = torch.rad2deg(angles)
    return angles


def _pairwise_angle_matrix(embeds_a, embeds_b, units):
    sim = embeds_a @ embeds_b.t()
    return _angles_from_similarity(sim, units)


def _summary_stats(values):
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _within_head_stats(embeds, units):
    matrix = _pairwise_angle_matrix(embeds, embeds, units=units)
    num_classes = matrix.shape[0]
    eye_mask = torch.eye(num_classes, dtype=torch.bool)

    diag = matrix[eye_mask]
    if num_classes > 1:
        offdiag = matrix[~eye_mask]
        offdiag_stats = _summary_stats(offdiag)
    else:
        offdiag_stats = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}

    return {
        "diag": _summary_stats(diag),
        "offdiag": offdiag_stats,
    }


def _offdiag_mean(matrix: torch.Tensor) -> float:
    n = matrix.shape[0]
    if n <= 1:
        return float("nan")
    mask = ~torch.eye(n, dtype=torch.bool)
    return float(matrix[mask].mean().item())


def _save_diff_heatmaps(
    out_path,
    dataset_name,
    units,
    diff_normal,
    diff_class,
    diff_prompt,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for heatmap output.") from exc

    mats = [diff_normal, diff_class, diff_prompt]
    titles = ["NormalAngle - VisualAngle", "ClassAngle - VisualAngle", "PromptAngle - VisualAngle"]
    vmax = max(float(torch.max(torch.abs(m)).item()) for m in mats)
    vmax = max(vmax, 1e-8)
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat.cpu(), cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Class index")
        ax.set_ylabel("Class index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    unit_label = "deg" if units == "degrees" else "rad"
    fig.suptitle(f"{dataset_name}: angle diff heatmaps ({unit_label})", fontsize=12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


def _run_pair_diff_heatmaps(args, dataset_name):
    base_dataset_name = _base_dataset_name(dataset_name)
    train_dataset_name = _ensure_train_dataset_name(base_dataset_name)

    normal_head = _load_normal_head(
        model_name=args.model,
        checkpoint_root=args.checkpoint_root,
        data_location=args.data_location,
        device=args.device,
        train_dataset_name=train_dataset_name,
    )
    per_class_head = _load_trained_drift_head(
        model_name=args.model,
        checkpoint_root=args.checkpoint_root,
        train_dataset_name=train_dataset_name,
        mode="per_class",
    )
    prompt_head = _load_prompt_head(
        model_name=args.model,
        checkpoint_root=args.checkpoint_root,
        data_location=args.data_location,
        device=args.device,
        train_dataset_name=train_dataset_name,
    )

    image_encoder = _load_image_encoder(args.model, args.checkpoint_root, train_dataset_name)
    visual_centroids = _collect_visual_class_centroids(
        image_encoder=image_encoder,
        dataset_name=train_dataset_name,
        data_location=args.data_location,
        device=args.device,
        batch_size=args.batch_size,
    )

    txt_normal = _effective_text_embeddings(normal_head)
    txt_class = _effective_text_embeddings(per_class_head)
    txt_prompt = _effective_text_embeddings(prompt_head)

    if (
        txt_normal.shape != txt_class.shape
        or txt_class.shape != txt_prompt.shape
        or txt_class.shape[0] != visual_centroids.shape[0]
    ):
        raise RuntimeError(
            "Class count mismatch among normal/class/prompt text embeddings and visual centroids: "
            f"normal={tuple(txt_normal.shape)}, class={tuple(txt_class.shape)}, "
            f"prompt={tuple(txt_prompt.shape)}, visual={tuple(visual_centroids.shape)}"
        )

    angle_normal = _pairwise_angle_matrix(txt_normal, txt_normal, units=args.units)
    angle_class = _pairwise_angle_matrix(txt_class, txt_class, units=args.units)
    angle_prompt = _pairwise_angle_matrix(txt_prompt, txt_prompt, units=args.units)
    angle_visual = _pairwise_angle_matrix(visual_centroids, visual_centroids, units=args.units)

    diff_normal = angle_normal - angle_visual
    diff_class = angle_class - angle_visual
    diff_prompt = angle_prompt - angle_visual

    out_path = os.path.join(args.heatmap_output_dir, args.model, f"{train_dataset_name}_angle_diff_heatmaps.png")
    _save_diff_heatmaps(
        out_path=out_path,
        dataset_name=train_dataset_name,
        units=args.units,
        diff_normal=diff_normal,
        diff_class=diff_class,
        diff_prompt=diff_prompt,
    )

    print("\nAverage angle difference over all class pairs (off-diagonal):")
    print(f"NormalAngle - VisualAngle: {_offdiag_mean(diff_normal):+.6f}")
    print(f"ClassAngle  - VisualAngle: {_offdiag_mean(diff_class):+.6f}")
    print(f"PromptAngle - VisualAngle: {_offdiag_mean(diff_prompt):+.6f}")


def _run_single_dataset(args, dataset_name):
    base_dataset_name = _base_dataset_name(dataset_name)
    train_dataset_name = _ensure_train_dataset_name(base_dataset_name)

    normal_head = _load_normal_head(
        model_name=args.model,
        checkpoint_root=args.checkpoint_root,
        data_location=args.data_location,
        device=args.device,
        train_dataset_name=train_dataset_name,
    )
    per_task_head = _load_trained_drift_head(
        model_name=args.model,
        checkpoint_root=args.checkpoint_root,
        train_dataset_name=train_dataset_name,
        mode="per_task",
    )
    per_class_head = _load_trained_drift_head(
        model_name=args.model,
        checkpoint_root=args.checkpoint_root,
        train_dataset_name=train_dataset_name,
        mode="per_class",
    )
    prompt_head = _load_prompt_head(
        model_name=args.model,
        checkpoint_root=args.checkpoint_root,
        data_location=args.data_location,
        device=args.device,
        train_dataset_name=train_dataset_name,
    )

    txt_normal = _effective_text_embeddings(normal_head)
    txt_task = _effective_text_embeddings(per_task_head)
    txt_class = _effective_text_embeddings(per_class_head)
    txt_prompt = _effective_text_embeddings(prompt_head)

    if txt_normal.shape != txt_task.shape or txt_normal.shape != txt_class.shape or txt_normal.shape != txt_prompt.shape:
        raise RuntimeError(
            "Text embedding shape mismatch: "
            f"normal={tuple(txt_normal.shape)}, "
            f"task={tuple(txt_task.shape)}, "
            f"class={tuple(txt_class.shape)}, "
            f"prompt={tuple(txt_prompt.shape)}"
        )

    within_normal = _within_head_stats(txt_normal, units=args.units)
    within_task = _within_head_stats(txt_task, units=args.units)
    within_class = _within_head_stats(txt_class, units=args.units)
    within_prompt = _within_head_stats(txt_prompt, units=args.units)

    return {
        "dataset": train_dataset_name,
        "normal": within_normal["offdiag"],
        "task": within_task["offdiag"],
        "class": within_class["offdiag"],
        "prompt": within_prompt["offdiag"],
    }


def _print_joint_summary(rows, units):
    unit_label = "º" if units == "degrees" else "rad"
    headers = [
        "Dataset",
        f"NormalMean({unit_label})",
        f"NormalStd({unit_label})",
        f"TaskMean({unit_label})",
        f"TaskStd({unit_label})",
        f"ClassMean({unit_label})",
        f"ClassStd({unit_label})",
        f"PromptMean({unit_label})",
        f"PromptStd({unit_label})",
    ]

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["dataset"],
                f"{row['normal']['mean']:.6f}",
                f"{row['normal']['std']:.6f}",
                f"{row['task']['mean']:.6f}",
                f"{row['task']['std']:.6f}",
                f"{row['class']['mean']:.6f}",
                f"{row['class']['std']:.6f}",
                f"{row['prompt']['mean']:.6f}",
                f"{row['prompt']['std']:.6f}",
            ]
        )

    avg_normal_mean = sum(r["normal"]["mean"] for r in rows) / len(rows)
    avg_normal_std = sum(r["normal"]["std"] for r in rows) / len(rows)
    avg_task_mean = sum(r["task"]["mean"] for r in rows) / len(rows)
    avg_task_std = sum(r["task"]["std"] for r in rows) / len(rows)
    avg_class_mean = sum(r["class"]["mean"] for r in rows) / len(rows)
    avg_class_std = sum(r["class"]["std"] for r in rows) / len(rows)
    avg_prompt_mean = sum(r["prompt"]["mean"] for r in rows) / len(rows)
    avg_prompt_std = sum(r["prompt"]["std"] for r in rows) / len(rows)
    table_rows.append(
        [
            "AVG",
            f"{avg_normal_mean:.6f}",
            f"{avg_normal_std:.6f}",
            f"{avg_task_mean:.6f}",
            f"{avg_task_std:.6f}",
            f"{avg_class_mean:.6f}",
            f"{avg_class_std:.6f}",
            f"{avg_prompt_mean:.6f}",
            f"{avg_prompt_std:.6f}",
        ]
    )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("\nnormal, task, class and prompt_head (º)")
    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for i, row in enumerate(table_rows):
        if i == len(table_rows) - 1:
            print("-+-".join("-" * w for w in widths))
        print(_fmt(row))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute angle metrics on text embeddings for normal, drift per-task, "
            "and drift per-class heads."
        )
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single base dataset name, e.g. MNIST.",
    )
    parser.add_argument(
        "--datasets",
        type=lambda x: [part.strip() for part in x.split(",") if part.strip()],
        default=None,
        help="Comma-separated base dataset names, e.g. MNIST,EuroSAT,DTD.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="/data/139-1/users/selkarrat/checkpoints")
    parser.add_argument("--data-location", type=str, default="/data/139-1/datasets/merging")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--pair-diff-heatmaps",
        action="store_true",
        help=(
            "For one dataset, compute all class-pair angles and save heatmaps of "
            "(Normal/Class/Prompt angle - Visual angle). Also prints average off-diagonal differences."
        ),
    )
    parser.add_argument(
        "--heatmap-output-dir",
        type=str,
        default="./plots/angle_diff_heatmaps",
        help="Output directory for pair-difference heatmaps.",
    )
    parser.add_argument(
        "--units",
        choices=["degrees", "radians"],
        default="degrees",
        help="Units for reported angles.",
    )
    args = parser.parse_args()

    if args.dataset and args.datasets:
        raise ValueError("Use either --dataset or --datasets, not both.")
    if args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = DEFAULT_DATASETS

    if args.pair_diff_heatmaps:
        if args.dataset is None:
            raise ValueError("--pair-diff-heatmaps requires --dataset <name>.")
        _run_pair_diff_heatmaps(args, args.dataset)

    joint_rows = []

    for dataset in datasets:
        try:
            row = _run_single_dataset(args, dataset)
            joint_rows.append(row)
        except FileNotFoundError as exc:
            print(f"\nSkipping {dataset}: {exc}")
        except RuntimeError as exc:
            print(f"\nSkipping {dataset}: {exc}")

    if not joint_rows:
        raise RuntimeError("No dataset angle metrics were produced.")

    _print_joint_summary(joint_rows, units=args.units)


if __name__ == "__main__":
    main()
