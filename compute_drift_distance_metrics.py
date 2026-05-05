import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageEncoder

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


def _load_image_encoder(model_name, checkpoint_root, train_dataset_name):
    dataset_ckpt_dir = os.path.join(checkpoint_root, model_name, train_dataset_name)
    zs_path = os.path.join(dataset_ckpt_dir, "zeroshot.pt")
    if os.path.isfile(zs_path):
        print(f"Loading image encoder from {zs_path}")
        return torch.load(zs_path, map_location="cpu")

    print("No zeroshot checkpoint found. Building image encoder from model weights.")
    encoder_args = SimpleNamespace(model=model_name, cache_dir=None)
    return ImageEncoder(encoder_args, keep_lang=False)


def _load_normal_head(model_name, checkpoint_root, data_location, device, train_dataset_name):
    model_ckpt_dir = os.path.join(checkpoint_root, model_name)
    explicit_path = os.path.join(model_ckpt_dir, f"head_{train_dataset_name}.pt")
    if os.path.isfile(explicit_path):
        print(f"Loading normal head from {explicit_path}")
        return torch.load(explicit_path, map_location="cpu")

    print("No cached normal head found. Building one with get_classification_head(...).")
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
    print(f"Loading {mode} drift head from {path}")
    return torch.load(path, map_location="cpu")


def _load_prompt_head(model_name, checkpoint_root, data_location, device, train_dataset_name):
    dataset_ckpt_dir = os.path.join(checkpoint_root, model_name, train_dataset_name)
    base_dataset = _base_dataset_name(train_dataset_name)
    prompt_path = os.path.join(dataset_ckpt_dir, f"prompt_csc_{base_dataset}.pt")
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Could not find prompt head checkpoint: {prompt_path}")

    prompt_obj = torch.load(prompt_path, map_location="cpu")
    normal_head = _load_normal_head(model_name, checkpoint_root, data_location, device, train_dataset_name)

    if isinstance(prompt_obj, torch.Tensor):
        prompt_weights = prompt_obj
        prompt_bias = None
    elif isinstance(prompt_obj, dict):
        if "weight" not in prompt_obj:
            raise ValueError(f'Unsupported prompt head format at "{prompt_path}": missing "weight".')
        prompt_weights = prompt_obj["weight"]
        prompt_bias = prompt_obj.get("bias")
    else:
        raise ValueError(f'Unsupported prompt head checkpoint type at "{prompt_path}": {type(prompt_obj)}')

    if normal_head.weight.shape != prompt_weights.shape:
        raise ValueError(
            "Prompt head shape mismatch for "
            f"{train_dataset_name}: expected {tuple(normal_head.weight.shape)}, "
            f"got {tuple(prompt_weights.shape)}"
        )
    normal_head.weight.data.copy_(prompt_weights.to(normal_head.weight.device))

    if prompt_bias is not None and getattr(normal_head, "bias", None) is not None:
        if normal_head.bias.shape != prompt_bias.shape:
            raise ValueError(
                "Prompt head bias shape mismatch for "
                f"{train_dataset_name}: expected {tuple(normal_head.bias.shape)}, "
                f"got {tuple(prompt_bias.shape)}"
            )
        normal_head.bias.data.copy_(prompt_bias.to(normal_head.bias.device))

    return normal_head.cpu()


def _collect_class_centroids(
    image_encoder,
    dataset_name,
    data_location,
    device,
    batch_size,
    max_images_per_class,
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
                if max_images_per_class > 0 and class_counts[cls] >= max_images_per_class:
                    continue
                class_sums[cls] += feats[i]
                class_counts[cls] += 1

    if class_sums is None:
        raise RuntimeError("No image embeddings collected; cannot compute class centroids.")

    valid_mask = class_counts > 0
    valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
    missing_indices = (~valid_mask).nonzero(as_tuple=False).flatten()

    if valid_indices.numel() == 0:
        raise RuntimeError("No classes with sampled images were found in this split.")

    centroids = class_sums[valid_indices] / class_counts[valid_indices].unsqueeze(1)
    centroids = F.normalize(centroids, dim=-1)
    valid_classnames = [dataset.classnames[int(i)] for i in valid_indices.tolist()]
    missing_classnames = [dataset.classnames[int(i)] for i in missing_indices.tolist()]

    return (
        centroids,
        class_counts[valid_indices],
        valid_classnames,
        valid_indices,
        missing_classnames,
    )


def _collect_class_embeddings(
    image_encoder,
    dataset_name,
    data_location,
    device,
    batch_size,
    max_images_per_class,
    show_progress=False,
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

    num_classes = len(dataset.classnames)
    per_class_feats = [[] for _ in range(num_classes)]
    per_class_count = torch.zeros(num_classes, dtype=torch.long)

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc=f"Embedding {dataset_name}", leave=False)

    with torch.no_grad():
        for batch in iterator:
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(device)
            y = batch["labels"].cpu()

            feats = image_encoder(x)
            feats = F.normalize(feats, dim=-1).cpu()

            for i in range(feats.shape[0]):
                cls = int(y[i].item())
                if max_images_per_class > 0 and per_class_count[cls] >= max_images_per_class:
                    continue
                per_class_feats[cls].append(feats[i])
                per_class_count[cls] += 1

    valid_indices = (per_class_count > 0).nonzero(as_tuple=False).flatten()
    missing_indices = (per_class_count == 0).nonzero(as_tuple=False).flatten()

    if valid_indices.numel() == 0:
        raise RuntimeError("No classes with sampled images were found in this split.")

    grouped_embeddings = [torch.stack(per_class_feats[int(i)], dim=0) for i in valid_indices.tolist()]
    valid_counts = per_class_count[valid_indices]
    valid_classnames = [dataset.classnames[int(i)] for i in valid_indices.tolist()]
    missing_classnames = [dataset.classnames[int(i)] for i in missing_indices.tolist()]

    return grouped_embeddings, valid_counts, valid_classnames, valid_indices, missing_classnames


def _pairwise_cosine_distance(text_embeds, visual_centroids):
    # Inputs are normalized so cosine distance = 1 - cosine similarity.
    sim = text_embeds @ visual_centroids.t()
    return 1.0 - sim


def _pairwise_cosine_similarity(text_embeds, visual_centroids):
    # Inputs are normalized; returns cosine similarities in [-1, 1].
    return text_embeds @ visual_centroids.t()


def _pairwise_avg_distance_all_embeddings(text_embeds, grouped_visual_embeds, show_progress=False):
    num_classes = len(grouped_visual_embeds)
    out = torch.empty(num_classes, num_classes, dtype=text_embeds.dtype)

    iterator = range(num_classes)
    if show_progress:
        iterator = tqdm(iterator, desc="Averaging class distances", leave=False)

    for j in iterator:
        visual_j = grouped_visual_embeds[j]  # [Nj, D]
        sims = text_embeds @ visual_j.t()  # [C, Nj]
        out[:, j] = 1.0 - sims.mean(dim=1)
    return out


def _centroids_from_grouped_embeddings(grouped_visual_embeds):
    centroids = torch.stack([embeds.mean(dim=0) for embeds in grouped_visual_embeds], dim=0)
    return F.normalize(centroids, dim=-1)


def _save_dual_similarity_plot(
    out_path,
    dataset_name,
    model_name,
    class_names,
    sim_normal,
    sim_right,
    right_title,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot-dual-sim-matrices but is not installed."
        ) from exc

    # Plot with rows=visual classes and cols=text classes.
    left = sim_normal.t().cpu()
    right = sim_right.t().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    im_left = axes[0].imshow(left, cmap="plasma", vmin=0.0, vmax=1.0, aspect="auto")
    im_right = axes[1].imshow(right, cmap="plasma", vmin=0.0, vmax=1.0, aspect="auto")

    axes[0].set_title("Normal")
    axes[1].set_title(right_title)
    for ax in axes:
        ax.set_xlabel("Text class (x)")
        ax.set_ylabel("Visual centroid class (y)")

        if len(class_names) <= 30:
            tick_ids = list(range(len(class_names)))
            ax.set_xticks(tick_ids)
            ax.set_yticks(tick_ids)
            ax.set_xticklabels(class_names, rotation=90, fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(im_left, ax=axes[0], fraction=0.046, pad=0.04, label="cosine similarity")
    fig.colorbar(im_right, ax=axes[1], fraction=0.046, pad=0.04, label="cosine similarity")
    fig.suptitle(f"{model_name} - {dataset_name}: cosine similarity matrices", fontsize=12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved dual similarity plot: {out_path}")


def _bootstrap_ci(values, confidence=0.95):
    alpha = 1.0 - confidence
    lower = float(torch.quantile(values, alpha / 2.0).item())
    upper = float(torch.quantile(values, 1.0 - alpha / 2.0).item())
    mean = float(values.mean().item())
    return {"mean": mean, "ci_lower": lower, "ci_upper": upper, "excludes_zero": not (lower <= 0.0 <= upper)}


def _bootstrap_from_per_class(summary_task, summary_class, summary_prompt, iters, seed, confidence=0.95):
    generator = torch.Generator().manual_seed(seed)
    task_match = torch.tensor(summary_task["per_class_match_increment"], dtype=torch.float32)
    class_match = torch.tensor(summary_class["per_class_match_increment"], dtype=torch.float32)
    task_other = torch.tensor(summary_task["per_class_nonmatch_avg_increment"], dtype=torch.float32)
    class_other = torch.tensor(summary_class["per_class_nonmatch_avg_increment"], dtype=torch.float32)
    prompt_match = torch.tensor(summary_prompt["per_class_match_increment"], dtype=torch.float32)
    prompt_other = torch.tensor(summary_prompt["per_class_nonmatch_avg_increment"], dtype=torch.float32)

    num_classes = task_match.numel()
    if num_classes == 0:
        raise RuntimeError("Cannot bootstrap with zero classes.")

    boot_task_match = torch.empty(iters, dtype=torch.float32)
    boot_class_match = torch.empty(iters, dtype=torch.float32)
    boot_task_other = torch.empty(iters, dtype=torch.float32)
    boot_class_other = torch.empty(iters, dtype=torch.float32)
    boot_prompt_match = torch.empty(iters, dtype=torch.float32)
    boot_prompt_other = torch.empty(iters, dtype=torch.float32)

    for b in range(iters):
        idx = torch.randint(num_classes, (num_classes,), generator=generator)
        boot_task_match[b] = task_match[idx].mean()
        boot_class_match[b] = class_match[idx].mean()
        boot_task_other[b] = task_other[idx].mean()
        boot_class_other[b] = class_other[idx].mean()
        boot_prompt_match[b] = prompt_match[idx].mean()
        boot_prompt_other[b] = prompt_other[idx].mean()

    return {
        "task_match_increment": _bootstrap_ci(boot_task_match, confidence=confidence),
        "task_other_increment": _bootstrap_ci(boot_task_other, confidence=confidence),
        "class_match_increment": _bootstrap_ci(boot_class_match, confidence=confidence),
        "class_other_increment": _bootstrap_ci(boot_class_other, confidence=confidence),
        "prompt_match_increment": _bootstrap_ci(boot_prompt_match, confidence=confidence),
        "prompt_other_increment": _bootstrap_ci(boot_prompt_other, confidence=confidence),
    }


def _bootstrap_from_all_embeddings(
    txt_normal,
    txt_task,
    txt_class,
    txt_prompt,
    grouped_visual_embeds,
    iters,
    seed,
    confidence=0.95,
    show_progress=False,
):
    generator = torch.Generator().manual_seed(seed)
    num_classes = len(grouped_visual_embeds)
    if num_classes == 0:
        raise RuntimeError("Cannot bootstrap with zero classes.")

    boot_task_match = torch.empty(iters, dtype=torch.float32)
    boot_class_match = torch.empty(iters, dtype=torch.float32)
    boot_task_other = torch.empty(iters, dtype=torch.float32)
    boot_class_other = torch.empty(iters, dtype=torch.float32)
    boot_prompt_match = torch.empty(iters, dtype=torch.float32)
    boot_prompt_other = torch.empty(iters, dtype=torch.float32)

    iterator = range(iters)
    if show_progress:
        iterator = tqdm(iterator, desc="Bootstrap CI", leave=False)

    for b in iterator:
        sampled_groups = []
        for embeds in grouped_visual_embeds:
            n = embeds.shape[0]
            idx = torch.randint(n, (n,), generator=generator)
            sampled_groups.append(embeds[idx])

        dist_normal = _pairwise_avg_distance_all_embeddings(txt_normal, sampled_groups, show_progress=False)
        dist_task = _pairwise_avg_distance_all_embeddings(txt_task, sampled_groups, show_progress=False)
        dist_class = _pairwise_avg_distance_all_embeddings(txt_class, sampled_groups, show_progress=False)
        dist_prompt = _pairwise_avg_distance_all_embeddings(txt_prompt, sampled_groups, show_progress=False)

        summary_task = _summarize_against_normal(dist_normal, dist_task)
        summary_class = _summarize_against_normal(dist_normal, dist_class)
        summary_prompt = _summarize_against_normal(dist_normal, dist_prompt)

        boot_task_match[b] = float(summary_task["match_distance_increment_mean"])
        boot_class_match[b] = float(summary_class["match_distance_increment_mean"])
        boot_task_other[b] = float(summary_task["nonmatch_distance_increment_mean"])
        boot_class_other[b] = float(summary_class["nonmatch_distance_increment_mean"])
        boot_prompt_match[b] = float(summary_prompt["match_distance_increment_mean"])
        boot_prompt_other[b] = float(summary_prompt["nonmatch_distance_increment_mean"])

    return {
        "task_match_increment": _bootstrap_ci(boot_task_match, confidence=confidence),
        "task_other_increment": _bootstrap_ci(boot_task_other, confidence=confidence),
        "class_match_increment": _bootstrap_ci(boot_class_match, confidence=confidence),
        "class_other_increment": _bootstrap_ci(boot_class_other, confidence=confidence),
        "prompt_match_increment": _bootstrap_ci(boot_prompt_match, confidence=confidence),
        "prompt_other_increment": _bootstrap_ci(boot_prompt_other, confidence=confidence),
    }


def _summarize_against_normal(normal_dist, candidate_dist):
    num_classes = normal_dist.shape[0]
    eye_mask = torch.eye(num_classes, dtype=torch.bool)

    normal_match = normal_dist[eye_mask]
    cand_match = candidate_dist[eye_mask]

    delta_match = cand_match - normal_match

    if num_classes > 1:
        normal_nonmatch = normal_dist[~eye_mask].view(num_classes, num_classes - 1)
        cand_nonmatch = candidate_dist[~eye_mask].view(num_classes, num_classes - 1)
        delta_nonmatch = cand_nonmatch.mean(dim=1) - normal_nonmatch.mean(dim=1)
        nonmatch_distance_normal_mean = float(normal_nonmatch.mean().item())
        nonmatch_distance_candidate_mean = float(cand_nonmatch.mean().item())
        nonmatch_distance_increment_mean = float(delta_nonmatch.mean().item())
        nonmatch_distance_increment_std = float(delta_nonmatch.std(unbiased=False).item())
        per_class_nonmatch_avg_increment = delta_nonmatch.tolist()
    else:
        nonmatch_distance_normal_mean = float("nan")
        nonmatch_distance_candidate_mean = float("nan")
        nonmatch_distance_increment_mean = float("nan")
        nonmatch_distance_increment_std = float("nan")
        per_class_nonmatch_avg_increment = [float("nan")]

    out = {
        "match_distance_normal_mean": float(normal_match.mean().item()),
        "match_distance_candidate_mean": float(cand_match.mean().item()),
        "match_distance_increment_mean": float(delta_match.mean().item()),
        "match_distance_increment_std": float(delta_match.std(unbiased=False).item()),
        "nonmatch_distance_normal_mean": nonmatch_distance_normal_mean,
        "nonmatch_distance_candidate_mean": nonmatch_distance_candidate_mean,
        "nonmatch_distance_increment_mean": nonmatch_distance_increment_mean,
        "nonmatch_distance_increment_std": nonmatch_distance_increment_std,
        "per_class_match_increment": delta_match.tolist(),
        "per_class_nonmatch_avg_increment": per_class_nonmatch_avg_increment,
    }
    return out


def _print_summary_table(summary_per_task, summary_per_class, summary_prompt):
    headers = [
        "Method",
        "MatchDist(Normal)",
        "MatchDist(Method)",
        "MatchIncrement",
        "OtherDist(Normal)",
        "OtherDist(Method)",
        "OtherAvgIncrement",
    ]

    rows = [
        [
            "drift_per_task",
            f"{summary_per_task['match_distance_normal_mean']:.6f}",
            f"{summary_per_task['match_distance_candidate_mean']:.6f}",
            f"{summary_per_task['match_distance_increment_mean']:+.6f}",
            f"{summary_per_task['nonmatch_distance_normal_mean']:.6f}",
            f"{summary_per_task['nonmatch_distance_candidate_mean']:.6f}",
            f"{summary_per_task['nonmatch_distance_increment_mean']:+.6f}",
        ],
        [
            "drift_per_class",
            f"{summary_per_class['match_distance_normal_mean']:.6f}",
            f"{summary_per_class['match_distance_candidate_mean']:.6f}",
            f"{summary_per_class['match_distance_increment_mean']:+.6f}",
            f"{summary_per_class['nonmatch_distance_normal_mean']:.6f}",
            f"{summary_per_class['nonmatch_distance_candidate_mean']:.6f}",
            f"{summary_per_class['nonmatch_distance_increment_mean']:+.6f}",
        ],
        [
            "prompt_head",
            f"{summary_prompt['match_distance_normal_mean']:.6f}",
            f"{summary_prompt['match_distance_candidate_mean']:.6f}",
            f"{summary_prompt['match_distance_increment_mean']:+.6f}",
            f"{summary_prompt['nonmatch_distance_normal_mean']:.6f}",
            f"{summary_prompt['nonmatch_distance_candidate_mean']:.6f}",
            f"{summary_prompt['nonmatch_distance_increment_mean']:+.6f}",
        ],
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(_fmt(row))


def _print_joint_summary(rows):
    headers = [
        "Dataset",
        "TaskMatchInc",
        "TaskOtherInc",
        "ClassMatchInc",
        "ClassOtherInc",
        "PromptMatchInc",
        "PromptOtherInc",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["dataset"],
                f"{row['task_match_inc']:+.6f}",
                f"{row['task_other_inc']:+.6f}",
                f"{row['class_match_inc']:+.6f}",
                f"{row['class_other_inc']:+.6f}",
                f"{row['prompt_match_inc']:+.6f}",
                f"{row['prompt_other_inc']:+.6f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("\nJoint Summary Across Datasets")
    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
        print(_fmt(row))

    avg_task_match = sum(r["task_match_inc"] for r in rows) / len(rows)
    avg_task_other = sum(r["task_other_inc"] for r in rows) / len(rows)
    avg_class_match = sum(r["class_match_inc"] for r in rows) / len(rows)
    avg_class_other = sum(r["class_other_inc"] for r in rows) / len(rows)
    avg_prompt_match = sum(r["prompt_match_inc"] for r in rows) / len(rows)
    avg_prompt_other = sum(r["prompt_other_inc"] for r in rows) / len(rows)
    print("-+-".join("-" * w for w in widths))
    print(
        _fmt(
            [
                "AVG",
                f"{avg_task_match:+.6f}",
                f"{avg_task_other:+.6f}",
                f"{avg_class_match:+.6f}",
                f"{avg_class_other:+.6f}",
                f"{avg_prompt_match:+.6f}",
                f"{avg_prompt_other:+.6f}",
            ]
        )
    )


def _print_bootstrap_summary(bootstrap_out, confidence=0.95):
    pct = int(round(confidence * 100))
    print(f"\nBootstrap {pct}% CI (mean increment):")
    print("Metric | Mean | CI Lower | CI Upper | Excludes 0")
    print("-----------------------------------------------")
    for key, label in [
        ("task_match_increment", "task_match_increment"),
        ("task_other_increment", "task_other_increment"),
        ("class_match_increment", "class_match_increment"),
        ("class_other_increment", "class_other_increment"),
        ("prompt_match_increment", "prompt_match_increment"),
        ("prompt_other_increment", "prompt_other_increment"),
    ]:
        stats = bootstrap_out[key]
        print(
            f"{label} | {stats['mean']:+.6f} | {stats['ci_lower']:+.6f} | "
            f"{stats['ci_upper']:+.6f} | {stats['excludes_zero']}"
        )


def _print_per_class(class_names, summary, title):
    print(f"\n{title}")
    print("Class | MatchIncrement | OtherAvgIncrement")
    print("------------------------------------------")
    for i, name in enumerate(class_names):
        dm = summary["per_class_match_increment"][i]
        dn = summary["per_class_nonmatch_avg_increment"][i]
        print(f"{name} | {dm:+.6f} | {dn:+.6f}")


def _run_single_dataset(args, dataset_name):
    base_dataset_name = _base_dataset_name(dataset_name)
    train_dataset_name = _ensure_train_dataset_name(base_dataset_name)
    embed_dataset_name = train_dataset_name if args.split == "validation" else base_dataset_name
    print("\n" + "=" * 100)
    print(f"Dataset: {train_dataset_name} | embedding split: {args.split} ({embed_dataset_name})")
    print("=" * 100)

    image_encoder = _load_image_encoder(args.model, args.checkpoint_root, train_dataset_name)
    if args.distance_target == "centroid":
        visual_targets, class_counts, class_names, valid_indices, missing_class_names = _collect_class_centroids(
            image_encoder=image_encoder,
            dataset_name=embed_dataset_name,
            data_location=args.data_location,
            device=args.device,
            batch_size=args.batch_size,
            max_images_per_class=args.max_images_per_class,
        )
        print(
            f"Collected class centroids for {len(class_names)} classes. "
            f"Min/Max samples per class: {int(class_counts.min().item())}/{int(class_counts.max().item())}"
        )
        visual_centroids_for_plot = visual_targets
    else:
        visual_targets, class_counts, class_names, valid_indices, missing_class_names = _collect_class_embeddings(
            image_encoder=image_encoder,
            dataset_name=embed_dataset_name,
            data_location=args.data_location,
            device=args.device,
            batch_size=args.batch_size,
            max_images_per_class=args.max_images_per_class,
            show_progress=args.show_progress,
        )
        print(
            f"Collected all class embeddings for {len(class_names)} classes. "
            f"Min/Max samples per class: {int(class_counts.min().item())}/{int(class_counts.max().item())}"
        )
        visual_centroids_for_plot = _centroids_from_grouped_embeddings(visual_targets)
    if missing_class_names:
        print(
            f"Warning: {len(missing_class_names)} classes have no validation samples "
            f"in split {embed_dataset_name} and were excluded from metrics."
        )
        print("Missing classes:", ", ".join(missing_class_names))

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

    # Align text classes to the classes that are present in the sampled validation split.
    txt_normal = txt_normal[valid_indices]
    txt_task = txt_task[valid_indices]
    txt_class = txt_class[valid_indices]
    txt_prompt = txt_prompt[valid_indices]

    if args.plot_dual_sim_matrices:
        sim_normal = _pairwise_cosine_similarity(txt_normal, visual_centroids_for_plot)
        sim_task = _pairwise_cosine_similarity(txt_task, visual_centroids_for_plot)
        sim_class = _pairwise_cosine_similarity(txt_class, visual_centroids_for_plot)
        sim_prompt = _pairwise_cosine_similarity(txt_prompt, visual_centroids_for_plot)

        dataset_plot_dir = os.path.join(args.plot_output_dir, args.model, train_dataset_name)
        ext = args.plot_format
        out_task = os.path.join(dataset_plot_dir, f"dual_similarity_normal_vs_drift_per_task.{ext}")
        out_class = os.path.join(dataset_plot_dir, f"dual_similarity_normal_vs_drift_per_class.{ext}")
        out_prompt = os.path.join(dataset_plot_dir, f"dual_similarity_normal_vs_prompt_head.{ext}")
        _save_dual_similarity_plot(
            out_path=out_task,
            dataset_name=train_dataset_name,
            model_name=args.model,
            class_names=class_names,
            sim_normal=sim_normal,
            sim_right=sim_task,
            right_title="Drift per-task",
        )
        _save_dual_similarity_plot(
            out_path=out_class,
            dataset_name=train_dataset_name,
            model_name=args.model,
            class_names=class_names,
            sim_normal=sim_normal,
            sim_right=sim_class,
            right_title="Drift per-class",
        )
        _save_dual_similarity_plot(
            out_path=out_prompt,
            dataset_name=train_dataset_name,
            model_name=args.model,
            class_names=class_names,
            sim_normal=sim_normal,
            sim_right=sim_prompt,
            right_title="Prompt head",
        )

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

    print("\nDistance metric: cosine distance = 1 - cosine_similarity (lower is closer).")
    print("Increment = method_distance - normal_distance.")
    print("Negative increment means method is closer than normal.")
    print()
    _print_summary_table(summary_task, summary_class, summary_prompt)

    if args.bootstrap_iters > 0:
        if args.distance_target == "centroid":
            bootstrap_out = _bootstrap_from_per_class(
                summary_task,
                summary_class,
                summary_prompt,
                iters=args.bootstrap_iters,
                seed=args.bootstrap_seed,
                confidence=args.bootstrap_confidence,
            )
        else:
            bootstrap_out = _bootstrap_from_all_embeddings(
                txt_normal,
                txt_task,
                txt_class,
                txt_prompt,
                visual_targets,
                iters=args.bootstrap_iters,
                seed=args.bootstrap_seed,
                confidence=args.bootstrap_confidence,
                show_progress=args.show_progress,
            )
        _print_bootstrap_summary(bootstrap_out, confidence=args.bootstrap_confidence)

    if args.print_per_class:
        _print_per_class(class_names, summary_task, "Per-class increments: drift_per_task vs normal")
        _print_per_class(class_names, summary_class, "Per-class increments: drift_per_class vs normal")
        _print_per_class(class_names, summary_prompt, "Per-class increments: prompt_head vs normal")

    return {
        "dataset": train_dataset_name,
        "task_match_inc": summary_task["match_distance_increment_mean"],
        "task_other_inc": summary_task["nonmatch_distance_increment_mean"],
        "class_match_inc": summary_class["match_distance_increment_mean"],
        "class_other_inc": summary_class["nonmatch_distance_increment_mean"],
        "prompt_match_inc": summary_prompt["match_distance_increment_mean"],
        "prompt_other_inc": summary_prompt["nonmatch_distance_increment_mean"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute drift-vs-normal distance metrics using class visual centroids. "
            "Distances are cosine distances on normalized embeddings."
        )
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Base dataset name, e.g. MNIST. If omitted, runs all default datasets and prints joint summary.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="/data/139-1/users/selkarrat/checkpoints")
    parser.add_argument("--data-location", type=str, default="/data/139-1/datasets/merging")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=-1,
        help="Limit val images per class used for centroids. Default uses all images.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "eval"],
        default="validation",
        help=(
            "Which split to use for visual embeddings: validation uses <dataset>Val val split; "
            "eval uses <dataset> test split."
        ),
    )
    parser.add_argument(
        "--distance-target",
        choices=["centroid", "all_embeddings"],
        default="centroid",
        help="Distance target per class: class centroid or average over all class embeddings.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars for embedding collection and all-embeddings distance aggregation.",
    )
    parser.add_argument(
        "--plot-dual-sim-matrices",
        action="store_true",
        help=(
            "Save dual cosine-similarity heatmaps per dataset: left=normal, "
            "right=drift_per_task, a second plot with right=drift_per_class, "
            "and a third plot with right=prompt_head."
        ),
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default="./plots/similarity_matrices",
        help="Output directory for dual similarity matrix plots.",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf"],
        default="png",
        help="File format for saved similarity matrix plots.",
    )
    parser.add_argument(
        "--print-per-class",
        action="store_true",
        help="Print per-class increments for both methods.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="If > 0, run bootstrap to estimate CI for increment means.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed used for bootstrap sampling.",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CI (e.g., 0.95).",
    )
    args = parser.parse_args()
    if args.bootstrap_iters < 0:
        raise ValueError("--bootstrap-iters must be >= 0.")
    if not (0.0 < args.bootstrap_confidence < 1.0):
        raise ValueError("--bootstrap-confidence must be in (0, 1).")
    datasets = [args.dataset] if args.dataset else DEFAULT_DATASETS
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
        raise RuntimeError("No dataset metrics were produced.")

    if len(joint_rows) > 1:
        _print_joint_summary(joint_rows)


if __name__ == "__main__":
    main()
