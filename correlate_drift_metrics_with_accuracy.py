import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch

from compute_drift_distance_metrics import (
    DEFAULT_DATASETS,
    _base_dataset_name,
    _collect_class_centroids,
    _collect_class_embeddings,
    _effective_text_embeddings,
    _ensure_train_dataset_name,
    _load_image_encoder,
    _load_normal_head,
    _load_prompt_head,
    _load_trained_drift_head,
    _pairwise_avg_distance_all_embeddings,
    _pairwise_cosine_distance,
    _summarize_against_normal,
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
    # Common compatibility fallback for baseline heads.
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


def _compute_dataset_increments(args, dataset):
    base_dataset_name = _base_dataset_name(dataset)
    train_dataset_name = _ensure_train_dataset_name(base_dataset_name)
    embed_dataset_name = train_dataset_name if args.split == "validation" else base_dataset_name

    image_encoder = _load_image_encoder(args.model, args.checkpoint_root, train_dataset_name)
    if args.distance_target == "centroid":
        visual_targets, _, _, valid_indices, _ = _collect_class_centroids(
            image_encoder=image_encoder,
            dataset_name=embed_dataset_name,
            data_location=args.data_location,
            device=args.device,
            batch_size=args.batch_size,
            max_images_per_class=args.max_images_per_class,
        )
    else:
        visual_targets, _, _, valid_indices, _ = _collect_class_embeddings(
            image_encoder=image_encoder,
            dataset_name=embed_dataset_name,
            data_location=args.data_location,
            device=args.device,
            batch_size=args.batch_size,
            max_images_per_class=args.max_images_per_class,
            show_progress=args.show_progress,
        )

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

    txt_normal = _effective_text_embeddings(normal_head)[valid_indices]
    txt_task = _effective_text_embeddings(per_task_head)[valid_indices]
    txt_class = _effective_text_embeddings(per_class_head)[valid_indices]
    txt_prompt = _effective_text_embeddings(prompt_head)[valid_indices]

    if args.distance_target == "centroid":
        dist_normal = _pairwise_cosine_distance(txt_normal, visual_targets)
        dist_task = _pairwise_cosine_distance(txt_task, visual_targets)
        dist_class = _pairwise_cosine_distance(txt_class, visual_targets)
        dist_prompt = _pairwise_cosine_distance(txt_prompt, visual_targets)
    else:
        dist_normal = _pairwise_avg_distance_all_embeddings(
            txt_normal, visual_targets, show_progress=args.show_progress
        )
        dist_task = _pairwise_avg_distance_all_embeddings(
            txt_task, visual_targets, show_progress=args.show_progress
        )
        dist_class = _pairwise_avg_distance_all_embeddings(
            txt_class, visual_targets, show_progress=args.show_progress
        )
        dist_prompt = _pairwise_avg_distance_all_embeddings(
            txt_prompt, visual_targets, show_progress=args.show_progress
        )

    summary_task = _summarize_against_normal(dist_normal, dist_task)
    summary_class = _summarize_against_normal(dist_normal, dist_class)
    summary_prompt = _summarize_against_normal(dist_normal, dist_prompt)
    return {
        "task_match_inc": summary_task["match_distance_increment_mean"],
        "task_other_inc": summary_task["nonmatch_distance_increment_mean"],
        "class_match_inc": summary_class["match_distance_increment_mean"],
        "class_other_inc": summary_class["nonmatch_distance_increment_mean"],
        "prompt_match_inc": summary_prompt["match_distance_increment_mean"],
        "prompt_other_inc": summary_prompt["nonmatch_distance_increment_mean"],
    }


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


def _plot_scatter(rows, x_key, y_key, out_path, title_prefix):
    x = [r[x_key] for r in rows]
    y = [r[y_key] for r in rows]
    labels = [r["dataset"] for r in rows]
    r = _pearson_r(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=60, alpha=0.85)
    for i, ds in enumerate(labels):
        ax.annotate(ds, (x[i], y[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.axhline(0.0, color="gray", linewidth=1, alpha=0.5)
    ax.axvline(0.0, color="gray", linewidth=1, alpha=0.5)
    ax.grid(alpha=0.2)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{title_prefix} (Pearson r={r:.4f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return r


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Correlate dataset-level delta accuracy with drift distance increments and save plots."
        )
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument(
        "--datasets",
        type=lambda x: x.split(","),
        default=None,
        help="Comma-separated datasets. Default uses the standard 8 datasets.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="/data/139-1/users/selkarrat/checkpoints")
    parser.add_argument("--data-location", type=str, default="/data/139-1/datasets/merging")
    parser.add_argument("--results-root", type=str, default="/data/139-1/users/selkarrat/results")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-images-per-class", type=int, default=-1)
    parser.add_argument("--split", choices=["validation", "eval"], default="validation")
    parser.add_argument("--distance-target", choices=["centroid", "all_embeddings"], default="centroid")
    parser.add_argument("--show-progress", action="store_true")

    parser.add_argument("--task-run-tag", type=str, default="drift_per_task_normal")
    parser.add_argument("--class-run-tag", type=str, default="drift_per_class_normal")
    parser.add_argument("--prompt-run-tag", type=str, default="prompt_head_zeroshot")
    parser.add_argument("--baseline-run-tag", type=str, default="normal_zeroshot")
    parser.add_argument("--output-dir", type=str, default="./plots/correlation")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for dataset in datasets:
        base_dataset = _base_dataset_name(dataset)
        print(f"\nProcessing {base_dataset}...")
        try:
            increments = _compute_dataset_increments(args, base_dataset)
        except Exception as exc:
            print(f"Skipping {base_dataset} (increment computation failed): {exc}")
            continue

        acc_base = _find_result_top1(args.results_root, args.model, args.baseline_run_tag, base_dataset)
        acc_task = _find_result_top1(args.results_root, args.model, args.task_run_tag, base_dataset)
        acc_class = _find_result_top1(args.results_root, args.model, args.class_run_tag, base_dataset)
        acc_prompt = _find_result_top1(args.results_root, args.model, args.prompt_run_tag, base_dataset)
        if acc_base is None or acc_task is None or acc_class is None or acc_prompt is None:
            print(
                f"Skipping {base_dataset} (missing accuracy file): "
                f"baseline={acc_base is not None}, task={acc_task is not None}, "
                f"class={acc_class is not None}, prompt={acc_prompt is not None}"
            )
            continue

        rows.append(
            {
                "dataset": base_dataset,
                "delta_acc_task": acc_task - acc_base,
                "delta_acc_class": acc_class - acc_base,
                "delta_acc_prompt": acc_prompt - acc_base,
                **increments,
            }
        )
        rows[-1]["task_sum_inc"] = rows[-1]["task_match_inc"] + rows[-1]["task_other_inc"]
        rows[-1]["class_sum_inc"] = rows[-1]["class_match_inc"] + rows[-1]["class_other_inc"]
        rows[-1]["prompt_sum_inc"] = rows[-1]["prompt_match_inc"] + rows[-1]["prompt_other_inc"]

    if len(rows) < 2:
        raise RuntimeError("Need at least 2 datasets with both metrics and accuracies to compute correlation.")

    r_task_other = _plot_scatter(
        rows,
        x_key="task_other_inc",
        y_key="delta_acc_task",
        out_path=os.path.join(args.output_dir, "corr_task_other_vs_delta_acc.png"),
        title_prefix="Per-task: other_increment vs delta_acc",
    )
    r_task_match = _plot_scatter(
        rows,
        x_key="task_match_inc",
        y_key="delta_acc_task",
        out_path=os.path.join(args.output_dir, "corr_task_match_vs_delta_acc.png"),
        title_prefix="Per-task: match_increment vs delta_acc",
    )
    r_class_other = _plot_scatter(
        rows,
        x_key="class_other_inc",
        y_key="delta_acc_class",
        out_path=os.path.join(args.output_dir, "corr_class_other_vs_delta_acc.png"),
        title_prefix="Per-class: other_increment vs delta_acc",
    )
    r_class_match = _plot_scatter(
        rows,
        x_key="class_match_inc",
        y_key="delta_acc_class",
        out_path=os.path.join(args.output_dir, "corr_class_match_vs_delta_acc.png"),
        title_prefix="Per-class: match_increment vs delta_acc",
    )
    r_prompt_other = _plot_scatter(
        rows,
        x_key="prompt_other_inc",
        y_key="delta_acc_prompt",
        out_path=os.path.join(args.output_dir, "corr_prompt_other_vs_delta_acc.png"),
        title_prefix="Prompt-head: other_increment vs delta_acc",
    )
    r_prompt_match = _plot_scatter(
        rows,
        x_key="prompt_match_inc",
        y_key="delta_acc_prompt",
        out_path=os.path.join(args.output_dir, "corr_prompt_match_vs_delta_acc.png"),
        title_prefix="Prompt-head: match_increment vs delta_acc",
    )
    r_task_sum = _plot_scatter(
        rows,
        x_key="task_sum_inc",
        y_key="delta_acc_task",
        out_path=os.path.join(args.output_dir, "corr_task_sum_vs_delta_acc.png"),
        title_prefix="Per-task: (match+other)_increment vs delta_acc",
    )
    r_class_sum = _plot_scatter(
        rows,
        x_key="class_sum_inc",
        y_key="delta_acc_class",
        out_path=os.path.join(args.output_dir, "corr_class_sum_vs_delta_acc.png"),
        title_prefix="Per-class: (match+other)_increment vs delta_acc",
    )
    r_prompt_sum = _plot_scatter(
        rows,
        x_key="prompt_sum_inc",
        y_key="delta_acc_prompt",
        out_path=os.path.join(args.output_dir, "corr_prompt_sum_vs_delta_acc.png"),
        title_prefix="Prompt-head: (match+other)_increment vs delta_acc",
    )

    print("\nPearson correlations:")
    print(f"per-task other vs delta_acc: {r_task_other:.4f}")
    print(f"per-task match vs delta_acc: {r_task_match:.4f}")
    print(f"per-class other vs delta_acc: {r_class_other:.4f}")
    print(f"per-class match vs delta_acc: {r_class_match:.4f}")
    print(f"prompt-head other vs delta_acc: {r_prompt_other:.4f}")
    print(f"prompt-head match vs delta_acc: {r_prompt_match:.4f}")
    print(f"per-task (match+other) vs delta_acc: {r_task_sum:.4f}")
    print(f"per-class (match+other) vs delta_acc: {r_class_sum:.4f}")
    print(f"prompt-head (match+other) vs delta_acc: {r_prompt_sum:.4f}")
    print(f"\nUsed {len(rows)} datasets.")


if __name__ == "__main__":
    main()
