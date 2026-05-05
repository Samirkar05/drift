import argparse
import os
from typing import Dict, List, Tuple

import torch
from merge_adapters import merge_adapted_finetuned_visual_encoders

DEFAULT_DATASETS = ["Cars", "SVHN", "MNIST", "SUN397", "RESISC45", "GTSRB", "EuroSAT", "DTD"]


def _ensure_train_dataset_name(dataset_name: str) -> str:
    return dataset_name if dataset_name.endswith("Val") else f"{dataset_name}Val"


def _load_checkpoint(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def _to_state_dict(obj) -> Dict[str, torch.Tensor]:
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        # If this is already a state_dict-like mapping.
        tensor_values = [v for v in obj.values() if torch.is_tensor(v)]
        if len(tensor_values) > 0:
            return obj
    raise TypeError(f"Unsupported checkpoint payload type: {type(obj)}")


def _is_weight_tensor(name: str, tensor: torch.Tensor, only_2d: bool) -> bool:
    if not torch.is_tensor(tensor):
        return False
    if not name.endswith("weight"):
        return False
    if tensor.dtype in (torch.int64, torch.uint8, torch.bool):
        return False
    if only_2d:
        return tensor.ndim == 2
    return tensor.ndim >= 2


def _collect_frobenius_norms(
    pretrained_sd: Dict[str, torch.Tensor],
    finetuned_sd: Dict[str, torch.Tensor],
    only_2d: bool,
) -> Tuple[List[float], int]:
    norms = []
    used_weight_elements = 0
    for key, pre_tensor in pretrained_sd.items():
        if key not in finetuned_sd:
            continue
        ft_tensor = finetuned_sd[key]
        if not _is_weight_tensor(key, pre_tensor, only_2d=only_2d):
            continue
        if pre_tensor.shape != ft_tensor.shape:
            continue

        pre = pre_tensor.detach().float().cpu()
        ft = ft_tensor.detach().float().cpu()
        diff = ft - pre
        frob = torch.linalg.norm(diff, ord="fro").item()
        norms.append(float(frob))
        used_weight_elements += int(pre.numel())
    return norms, used_weight_elements


def _count_total_weight_elements(state_dict: Dict[str, torch.Tensor], only_2d: bool) -> int:
    total = 0
    for key, tensor in state_dict.items():
        if _is_weight_tensor(key, tensor, only_2d=only_2d):
            total += int(tensor.numel())
    return total


def _mean_std(values: List[float]) -> Tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float32)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=False).item())
    return mean, std


def _format_pm(mean: float, std: float, precision: int) -> str:
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _load_target_encoder_checkpoint(model: str, checkpoint_root: str, train_dataset: str, encoder_mode: str):
    ckpt_dir = os.path.join(checkpoint_root, model, train_dataset)
    if encoder_mode in {"normal", "zeroshot"}:
        return _load_checkpoint(os.path.join(ckpt_dir, "zeroshot.pt"))
    if encoder_mode == "finetuned":
        return _load_checkpoint(os.path.join(ckpt_dir, "finetuned.pt"))
    if encoder_mode == "adapted_finetuned":
        return _load_checkpoint(os.path.join(ckpt_dir, "adapted_finetuned.pt"))
    if encoder_mode == "adapted_finetuned_drift_per_task":
        return _load_checkpoint(os.path.join(ckpt_dir, "adapted_finetuned_drift_per_task.pt"))
    if encoder_mode == "adapted_finetuned_drift_per_class":
        return _load_checkpoint(os.path.join(ckpt_dir, "adapted_finetuned_drift_per_class.pt"))
    if encoder_mode == "adapted_finetuned_prompt_head":
        return _load_checkpoint(os.path.join(ckpt_dir, "adapted_finetuned_prompt_head.pt"))

    # merged_adapted_finetuned
    zeroshot_path = os.path.join(ckpt_dir, "zeroshot.pt")
    finetuned_paths = []
    for ds in DEFAULT_DATASETS:
        source_path = os.path.join(checkpoint_root, model, _ensure_train_dataset_name(ds), "adapted_finetuned.pt")
        if os.path.isfile(source_path):
            finetuned_paths.append(source_path)
    if len(finetuned_paths) == 0:
        raise FileNotFoundError(f"No adapted_finetuned checkpoints found for {model} to merge.")

    return merge_adapted_finetuned_visual_encoders(
        zeroshot_checkpoint=zeroshot_path,
        finetuned_checkpoints=finetuned_paths,
        scaling_coef=1.0 / len(finetuned_paths),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute average Frobenius norm of (selected_encoder - zeroshot) across "
            "vision encoder weight tensors, with std, reported as mean ± std."
        )
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument(
        "--datasets",
        type=lambda x: [part.strip() for part in x.split(",") if part.strip()],
        default=None,
        help="Comma-separated base dataset names. Default: standard 8 datasets.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="/data/139-1/users/selkarrat/checkpoints")
    parser.add_argument(
        "--encoder",
        choices=[
            "normal",
            "zeroshot",
            "finetuned",
            "adapted_finetuned",
            "adapted_finetuned_drift_per_task",
            "adapted_finetuned_drift_per_class",
            "adapted_finetuned_prompt_head",
            "merged_adapted_finetuned",
        ],
        default="adapted_finetuned",
        help="Target encoder mode to compare against zeroshot baseline.",
    )
    parser.add_argument(
        "--only-2d",
        action="store_true",
        help="Use only 2D weight tensors (strict matrix interpretation).",
    )
    parser.add_argument("--precision", type=int, default=6)
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS

    rows = []
    pooled_norms: List[float] = []

    for base_dataset in datasets:
        train_dataset = _ensure_train_dataset_name(base_dataset)
        ckpt_dir = os.path.join(args.checkpoint_root, args.model, train_dataset)

        zeroshot_path = os.path.join(ckpt_dir, "zeroshot.pt")
        target_label = args.encoder

        try:
            zeroshot_obj = _load_checkpoint(zeroshot_path)
            target_obj = _load_target_encoder_checkpoint(
                model=args.model,
                checkpoint_root=args.checkpoint_root,
                train_dataset=train_dataset,
                encoder_mode=args.encoder,
            )
            zeroshot_sd = _to_state_dict(zeroshot_obj)
            target_sd = _to_state_dict(target_obj)
            norms, used_weight_elements = _collect_frobenius_norms(
                zeroshot_sd, target_sd, only_2d=args.only_2d
            )
        except (FileNotFoundError, TypeError) as exc:
            print(f"Skipping {train_dataset}: {exc}")
            continue

        if len(norms) == 0:
            print(f"Skipping {train_dataset}: no matching weight tensors found.")
            continue

        total_weight_elements = _count_total_weight_elements(zeroshot_sd, only_2d=args.only_2d)
        if total_weight_elements == 0:
            print(f"Skipping {train_dataset}: no eligible weight elements found in zeroshot checkpoint.")
            continue

        used_pct = 100.0 * (used_weight_elements / total_weight_elements)
        mean, std = _mean_std(norms)
        rows.append((train_dataset, used_pct, mean, std))
        pooled_norms.extend(norms)

    if not rows:
        raise RuntimeError("No dataset results were produced.")

    header = f"{'Dataset':<12} {'UsedWeight%':>12} {f'Frobenius({target_label} - zeroshot)':>30}"
    print(header)
    print("-" * len(header))
    for dataset, used_pct, mean, std in rows:
        print(f"{dataset:<12} {used_pct:>11.2f}% {_format_pm(mean, std, args.precision):>30}")

    pooled_mean, pooled_std = _mean_std(pooled_norms)
    avg_of_means = sum(r[2] for r in rows) / len(rows)
    avg_of_stds = sum(r[3] for r in rows) / len(rows)

    print("-" * len(header))
    avg_used_pct = sum(r[1] for r in rows) / len(rows)
    print(f"{'AVG(per-dataset)':<12} {avg_used_pct:>11.2f}% {_format_pm(avg_of_means, avg_of_stds, args.precision):>30}")
    print(f"{'POOLED(all)':<12} {'n/a':>12} {_format_pm(pooled_mean, pooled_std, args.precision):>30}")


if __name__ == "__main__":
    main()
