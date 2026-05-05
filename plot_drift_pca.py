import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.modeling import ImageEncoder


def _ensure_train_dataset_name(dataset_name: str) -> str:
    return dataset_name if dataset_name.endswith("Val") else f"{dataset_name}Val"


def _project_pca_2d(*blocks):
    lengths = [b.shape[0] for b in blocks]
    x = torch.cat(blocks, dim=0)
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=2, center=False)
    z = x @ v[:, :2]

    out = []
    start = 0
    for length in lengths:
        out.append(z[start : start + length])
        start += length
    return out


def _effective_text_embeddings(head):
    embeds = head.weight.detach().cpu()
    if hasattr(head, "drift"):
        drift = head.drift.detach().cpu()
        if drift.ndim == 1:
            embeds = embeds + drift.unsqueeze(0)
        else:
            embeds = embeds + drift
    embeds = F.normalize(embeds, dim=-1)
    return embeds


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

    print("No cached normal head found. Building a new one with get_classification_head(...).")
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


def _collect_image_embeddings(
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

    features = []
    labels = []
    per_class_count = {}

    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(device)
            y = batch["labels"].cpu()

            feats = image_encoder(x)
            feats = F.normalize(feats, dim=-1).cpu()

            for i in range(feats.shape[0]):
                cls = int(y[i].item())
                cnt = per_class_count.get(cls, 0)
                if max_images_per_class > 0 and cnt >= max_images_per_class:
                    continue
                per_class_count[cls] = cnt + 1
                features.append(feats[i])
                labels.append(cls)

    if len(features) == 0:
        raise RuntimeError("No image embeddings were collected. Check dataset and sampling settings.")

    features = torch.stack(features, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels, dataset.classnames


def _colors_for_classes(num_classes):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(num_classes)]


def _scatter_image_points(ax, z_img, labels, colors):
    num_classes = len(colors)
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            ax.scatter(
                z_img[mask, 0],
                z_img[mask, 1],
                s=10,
                alpha=0.25,
                color=colors[cls],
            )


def _scatter_text_points(ax, z_txt, colors, marker, label):
    ax.scatter(
        z_txt[:, 0],
        z_txt[:, 1],
        s=110,
        marker=marker,
        c=colors,
        edgecolors="black",
        linewidths=0.7,
        label=label,
    )
    for i in range(z_txt.shape[0]):
        ax.text(z_txt[i, 0], z_txt[i, 1], str(i), fontsize=7, ha="left", va="bottom")


def _save_single_mode_plot(out_path, title, z_img, labels, z_txt, class_names):
    colors = _colors_for_classes(len(class_names))
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_image_points(ax, z_img, labels, colors)
    _scatter_text_points(ax, z_txt, colors, marker="X", label="class text prototype")

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _save_combined_plot(out_path, z_img, labels, z_normal, z_task, z_class, class_names):
    colors = _colors_for_classes(len(class_names))
    fig, ax = plt.subplots(figsize=(11, 8))

    _scatter_image_points(ax, z_img, labels, colors)
    _scatter_text_points(ax, z_normal, colors, marker="X", label="normal text")
    _scatter_text_points(ax, z_task, colors, marker="^", label="drift per-task text")
    _scatter_text_points(ax, z_class, colors, marker="P", label="drift per-class text")

    for i in range(len(class_names)):
        ax.plot(
            [z_normal[i, 0], z_task[i, 0]],
            [z_normal[i, 1], z_task[i, 1]],
            color=colors[i],
            alpha=0.45,
            linewidth=1.0,
        )
        ax.plot(
            [z_normal[i, 0], z_class[i, 0]],
            [z_normal[i, 1], z_class[i, 1]],
            color=colors[i],
            alpha=0.45,
            linewidth=1.0,
            linestyle="--",
        )

    ax.set_title("PCA Combined: Normal vs Drift Per-Task vs Drift Per-Class")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate PCA plots for normal head, drift per-task head, drift per-class head, "
            "and a combined comparison plot."
        )
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--dataset", type=str, required=True, help="Base dataset name, e.g. MNIST")
    parser.add_argument("--checkpoint-root", type=str, default="/data/139-1/users/selkarrat/checkpoints")
    parser.add_argument("--data-location", type=str, default="/data/139-1/datasets/merging")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=-1,
        help="Limit val image points per class for cleaner plots. Default uses all images.",
    )
    parser.add_argument("--output-dir", type=str, default="./plots")
    args = parser.parse_args()

    train_dataset_name = _ensure_train_dataset_name(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    image_encoder = _load_image_encoder(args.model, args.checkpoint_root, train_dataset_name)
    image_embeds, image_labels, class_names = _collect_image_embeddings(
        image_encoder=image_encoder,
        dataset_name=train_dataset_name,
        data_location=args.data_location,
        device=args.device,
        batch_size=args.batch_size,
        max_images_per_class=args.max_images_per_class,
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

    txt_normal = _effective_text_embeddings(normal_head)
    txt_task = _effective_text_embeddings(per_task_head)
    txt_class = _effective_text_embeddings(per_class_head)
    out_prefix = f"{args.model}_{train_dataset_name}"

    z_img_n, z_txt_n = _project_pca_2d(image_embeds, txt_normal)
    z_img_t, z_txt_t = _project_pca_2d(image_embeds, txt_task)
    z_img_c, z_txt_c = _project_pca_2d(image_embeds, txt_class)
    z_img_all, z_txt_n_all, z_txt_t_all, z_txt_c_all = _project_pca_2d(
        image_embeds,
        txt_normal,
        txt_task,
        txt_class,
    )

    _save_single_mode_plot(
        out_path=os.path.join(args.output_dir, f"{out_prefix}_pca_normal.png"),
        title=f"PCA: Normal Head ({args.model}, {train_dataset_name})",
        z_img=z_img_n,
        labels=image_labels,
        z_txt=z_txt_n,
        class_names=class_names,
    )
    _save_single_mode_plot(
        out_path=os.path.join(args.output_dir, f"{out_prefix}_pca_drift_per_task.png"),
        title=f"PCA: Drift Per-Task Head ({args.model}, {train_dataset_name})",
        z_img=z_img_t,
        labels=image_labels,
        z_txt=z_txt_t,
        class_names=class_names,
    )
    _save_single_mode_plot(
        out_path=os.path.join(args.output_dir, f"{out_prefix}_pca_drift_per_class.png"),
        title=f"PCA: Drift Per-Class Head ({args.model}, {train_dataset_name})",
        z_img=z_img_c,
        labels=image_labels,
        z_txt=z_txt_c,
        class_names=class_names,
    )
    _save_combined_plot(
        out_path=os.path.join(args.output_dir, f"{out_prefix}_pca_combined.png"),
        z_img=z_img_all,
        labels=image_labels,
        z_normal=z_txt_n_all,
        z_task=z_txt_t_all,
        z_class=z_txt_c_all,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
