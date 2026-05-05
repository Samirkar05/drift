import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset

from compute_drift_distance_metrics import (
    DEFAULT_DATASETS,
    _base_dataset_name,
    _ensure_train_dataset_name,
    _effective_text_embeddings,
    _load_image_encoder,
    _load_trained_drift_head,
    _load_prompt_head,
)


def _extract_top1(result_file: Path, dataset: str):
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


def _tag_aliases(tag: str):
    aliases = [tag]
    legacy_map = {
        "normal_zeroshot": "zeroshot",
        "adapter_zeroshot": "mlp",
        "merged_adapter_zeroshot": "mlp_merging",
        "adapter_adapted_finetuned": "adapted_finetuned",
    }
    if tag in legacy_map:
        aliases.append(legacy_map[tag])
    if tag.endswith("_normal"):
        aliases.append(tag.replace("_normal", "_zeroshot"))
    elif tag.endswith("_zeroshot"):
        aliases.append(tag.replace("_zeroshot", "_normal"))
    if tag == "normal_zeroshot":
        aliases.append("normal_normal")
    return list(dict.fromkeys(aliases))


def _find_result_top1(results_root, model, run_tag, dataset):
    model_dir = Path(results_root) / model
    eval_dir = model_dir / "drift_eval"
    for tag in _tag_aliases(run_tag):
        eval_file = eval_dir / f"results_{tag}_{dataset}"
        top1 = _extract_top1(eval_file, dataset)
        if top1 is not None:
            return top1

        model_file = model_dir / f"results_{tag}_{dataset}"
        top1 = _extract_top1(model_file, dataset)
        if top1 is not None:
            return top1
    return None


def _load_text_embeddings(mode, args, train_dataset_name):
    if mode == "class":
        head = _load_trained_drift_head(
            model_name=args.model,
            checkpoint_root=args.checkpoint_root,
            train_dataset_name=train_dataset_name,
            mode="per_class",
        )
    elif mode == "task":
        head = _load_trained_drift_head(
            model_name=args.model,
            checkpoint_root=args.checkpoint_root,
            train_dataset_name=train_dataset_name,
            mode="per_task",
        )
    elif mode == "prompt":
        head = _load_prompt_head(
            model_name=args.model,
            checkpoint_root=args.checkpoint_root,
            data_location=args.data_location,
            device=args.device,
            train_dataset_name=train_dataset_name,
        )
    else:
        raise ValueError(f"Unsupported text mode: {mode}")
    return _effective_text_embeddings(head)


def _compute_hard_margin_for_dataset(args, dataset_name, text_mode: str) -> float:
    base_dataset_name = _base_dataset_name(dataset_name)
    train_dataset_name = _ensure_train_dataset_name(base_dataset_name)
    eval_dataset_name = train_dataset_name if args.split == "validation" else base_dataset_name

    image_encoder = _load_image_encoder(args.model, args.checkpoint_root, train_dataset_name)
    txt = _load_text_embeddings(text_mode, args, train_dataset_name).to(args.device)

    dataset = get_dataset(
        eval_dataset_name,
        image_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    loader_args = argparse.Namespace(batch_size=args.batch_size, device=args.device)
    dataloader = get_dataloader(dataset, is_train=False, args=loader_args, image_encoder=None)

    image_encoder = image_encoder.to(args.device)
    image_encoder.eval()

    margins: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(args.device)
            y = batch["labels"].to(args.device)

            feats = image_encoder(x)
            feats = F.normalize(feats, dim=-1)
            sims = feats @ txt.t()  # [B, C]

            row_ids = torch.arange(sims.shape[0], device=sims.device)
            correct = sims[row_ids, y]
            masked = sims.clone()
            masked[row_ids, y] = float("-inf")
            hardest_wrong = masked.max(dim=1).values
            margins.append((correct - hardest_wrong).cpu())

    if len(margins) == 0:
        raise RuntimeError(f"No samples found for dataset: {eval_dataset_name}")

    return float(torch.cat(margins).mean().item())


def _pearson_r(x_vals, y_vals):
    x = torch.tensor(x_vals, dtype=torch.float32)
    y = torch.tensor(y_vals, dtype=torch.float32)
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).sum()) * (y.pow(2).sum()))
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((x * y).sum().item() / denom.item())


def _plot_hard_margin(rows, out_path, y_label, r_class, r_prompt, include_task=False, r_task=float("nan")):
    x_class = [r["class_hard_margin"] for r in rows]
    y_class = [r["class_target_acc"] for r in rows]
    x_prompt = [r["prompt_hard_margin"] for r in rows]
    y_prompt = [r["prompt_target_acc"] for r in rows]
    if include_task:
        x_task = [r["task_hard_margin"] for r in rows]
        y_task = [r["task_target_acc"] for r in rows]
    labels = [r["dataset"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_facecolor("#f4f4f4")
    fig.patch.set_facecolor("#f4f4f4")

    ax.scatter(
        x_class,
        y_class,
        s=95,
        color="#4f7eb9",
        edgecolor="#1d1d1d",
        linewidths=0.8,
        alpha=0.9,
        label=f"Class Drift (r={r_class:.3f})",
        zorder=3,
    )
    ax.scatter(
        x_prompt,
        y_prompt,
        s=90,
        marker="D",
        color="#e08a36",
        edgecolor="#1d1d1d",
        linewidths=0.8,
        alpha=0.9,
        label=f"Prompt Head (r={r_prompt:.3f})",
        zorder=3,
    )
    if include_task:
        ax.scatter(
            x_task,
            y_task,
            s=95,
            marker="^",
            color="#4c9f70",
            edgecolor="#1d1d1d",
            linewidths=0.8,
            alpha=0.9,
            label=f"Task Drift (r={r_task:.3f})",
            zorder=3,
        )

    for i, ds in enumerate(labels):
        ax.annotate(ds, (x_class[i], y_class[i]), fontsize=8, xytext=(4, 4), textcoords="offset points", color="#365b86")
        ax.annotate(ds, (x_prompt[i], y_prompt[i]), fontsize=8, xytext=(4, -10), textcoords="offset points", color="#9a5c1f")
        if include_task:
            ax.annotate(ds, (x_task[i], y_task[i]), fontsize=8, xytext=(4, 10), textcoords="offset points", color="#2f6d4c")

    ax.grid(alpha=0.25)
    ax.set_xlabel("Hard Margin  E[x·t_y - max_{j≠y} x·t_j]", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if include_task:
        ax.set_title("Hard Margin vs Accuracy: Class Drift, Prompt Head, and Task Drift", fontsize=13)
    else:
        ax.set_title("Hard Margin vs Accuracy: Class Drift and Prompt Head", fontsize=13)

    leg = ax.legend(loc="best", frameon=True, facecolor="#f4f4f4", edgecolor="#aaaaaa")
    for txt in leg.get_texts():
        txt.set_fontsize(10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Correlate hard margin A-C with accuracy for two methods separately: "
            "Class Drift and Prompt Head, in one styled plot."
        )
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument(
        "--datasets",
        type=lambda x: [part.strip() for part in x.split(",") if part.strip()],
        default=None,
        help="Comma-separated datasets. Default uses the standard 8 datasets.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="/data/139-1/users/selkarrat/checkpoints")
    parser.add_argument("--data-location", type=str, default="/data/139-1/datasets/merging")
    parser.add_argument("--results-root", type=str, default="/data/139-1/users/selkarrat/results")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", choices=["validation", "eval"], default="validation")

    parser.add_argument("--class-run-tag", type=str, default="drift_per_class_normal")
    parser.add_argument("--prompt-run-tag", type=str, default="prompt_head_zeroshot")
    parser.add_argument("--task-run-tag", type=str, default="drift_per_task_normal")
    parser.add_argument(
        "--include-task-drift",
        action="store_true",
        help="Include Task Drift as a third series with separate correlation.",
    )
    parser.add_argument("--baseline-run-tag", type=str, default=None)
    parser.add_argument(
        "--use-delta-accuracy",
        action="store_true",
        help="Use delta accuracy: acc(method)-acc(baseline_run_tag).",
    )
    parser.add_argument("--output-dir", type=str, default="./plots/margin_correlation")
    args = parser.parse_args()

    if args.use_delta_accuracy and not args.baseline_run_tag:
        raise ValueError("--use-delta-accuracy requires --baseline-run-tag")

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for dataset in datasets:
        base_dataset = _base_dataset_name(dataset)
        print(f"\nProcessing {base_dataset}...")

        try:
            class_margin = _compute_hard_margin_for_dataset(args, base_dataset, text_mode="class")
            prompt_margin = _compute_hard_margin_for_dataset(args, base_dataset, text_mode="prompt")
            if args.include_task_drift:
                task_margin = _compute_hard_margin_for_dataset(args, base_dataset, text_mode="task")
        except Exception as exc:
            print(f"Skipping {base_dataset} (margin computation failed): {exc}")
            continue

        class_acc = _find_result_top1(args.results_root, args.model, args.class_run_tag, base_dataset)
        prompt_acc = _find_result_top1(args.results_root, args.model, args.prompt_run_tag, base_dataset)
        task_acc = None
        if args.include_task_drift:
            task_acc = _find_result_top1(args.results_root, args.model, args.task_run_tag, base_dataset)
        if class_acc is None or prompt_acc is None or (args.include_task_drift and task_acc is None):
            print(
                f"Skipping {base_dataset} (missing accuracy files): "
                f"class={class_acc is not None}, prompt={prompt_acc is not None}, "
                f"task={task_acc is not None if args.include_task_drift else True}"
            )
            continue

        if args.use_delta_accuracy:
            baseline_acc = _find_result_top1(args.results_root, args.model, args.baseline_run_tag, base_dataset)
            if baseline_acc is None:
                print(f"Skipping {base_dataset} (missing baseline accuracy: {args.baseline_run_tag})")
                continue
            class_target = class_acc - baseline_acc
            prompt_target = prompt_acc - baseline_acc
            if args.include_task_drift:
                task_target = task_acc - baseline_acc
            y_label = "Delta Accuracy (vs baseline)"
        else:
            class_target = class_acc
            prompt_target = prompt_acc
            if args.include_task_drift:
                task_target = task_acc
            y_label = "Top-1 Accuracy"

        row = {
            "dataset": base_dataset,
            "class_hard_margin": class_margin,
            "prompt_hard_margin": prompt_margin,
            "class_target_acc": class_target,
            "prompt_target_acc": prompt_target,
        }
        if args.include_task_drift:
            row["task_hard_margin"] = task_margin
            row["task_target_acc"] = task_target
        rows.append(row)

    if len(rows) < 2:
        raise RuntimeError("Need at least 2 datasets with hard-margin and accuracy values.")

    r_class = _pearson_r(
        [r["class_hard_margin"] for r in rows],
        [r["class_target_acc"] for r in rows],
    )
    r_prompt = _pearson_r(
        [r["prompt_hard_margin"] for r in rows],
        [r["prompt_target_acc"] for r in rows],
    )
    r_task = float("nan")
    if args.include_task_drift:
        r_task = _pearson_r(
            [r["task_hard_margin"] for r in rows],
            [r["task_target_acc"] for r in rows],
        )

    _plot_hard_margin(
        rows,
        out_path=os.path.join(args.output_dir, "hard_margin_class_vs_prompt.png"),
        y_label=y_label,
        r_class=r_class,
        r_prompt=r_prompt,
        include_task=args.include_task_drift,
        r_task=r_task,
    )

    print("\nPearson correlations (computed separately):")
    print(f"Class Drift:  r(M_hard, target_acc) = {r_class:.4f}")
    print(f"Prompt Head:  r(M_hard, target_acc) = {r_prompt:.4f}")
    if args.include_task_drift:
        print(f"Task Drift:   r(M_hard, target_acc) = {r_task:.4f}")


if __name__ == "__main__":
    main()
