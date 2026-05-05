from src.finetune import adapted_finetuning
from src.args import parse_arguments
import argparse
import os
import sys


def parse_runner_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--head-type",
        choices=["adapter", "normal", "drift_per_task", "drift_per_class", "prompt_head"],
        default="adapter",
        help=(
            "Head used while training the visual encoder: "
            "adapter, normal, drift_per_task, drift_per_class, prompt_head."
        ),
    )
    parser.add_argument(
        "--remove-previous-checkpoints",
        action="store_true",
        help="Remove the existing finetuned encoder checkpoint before training each dataset.",
    )
    runner_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return runner_args


runner_args = parse_runner_args()
data_location = "/data/139-1/datasets/merging"
models = ["ViT-B-32"]
datasets = ["Cars", "SVHN", "MNIST", "SUN397", "RESISC45", "GTSRB", "EuroSAT", "DTD"]

epochs = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SUN397": 14,
    "SVHN": 4,
}

for model in models:
    for dataset in datasets:
        print("=" * 100)
        print(f"Flexible-head encoder finetuning for {model} on {dataset} (head={runner_args.head_type})")
        print("=" * 100)
        args = parse_arguments()
        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.data_location = data_location
        args.train_dataset = dataset + "Val"
        args.batch_size = 128
        args.model = model
        args.save = f"/data/139-1/users/selkarrat/checkpoints/{model}"
        args.load = f"/data/139-1/users/selkarrat/checkpoints/{model}"

        train_dataset = args.train_dataset
        ckpdir = os.path.join(args.save, train_dataset)
        if runner_args.head_type == "adapter":
            target_ckpt = os.path.join(ckpdir, "adapted_finetuned.pt")
        elif runner_args.head_type == "normal":
            target_ckpt = os.path.join(ckpdir, "finetuned.pt")
        else:
            target_ckpt = os.path.join(ckpdir, f"adapted_finetuned_{runner_args.head_type}.pt")

        if runner_args.remove_previous_checkpoints and os.path.isfile(target_ckpt):
            os.remove(target_ckpt)
            print(f"Removed previous checkpoint: {target_ckpt}")

        adapted_finetuning(args, head_type=runner_args.head_type)
