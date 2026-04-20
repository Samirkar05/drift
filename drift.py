from src.finetune import train_drift
from src.args import parse_arguments
import argparse
import os
import sys


def parse_drift_runner_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--head-type",
        choices=["drift", "rigid"],
        default="drift",
        help="Training mode: drift -> train_drift(..., rigid_movement=False), rigid -> rigid_movement=True.",
    )
    parser.add_argument(
        "--remove-previous-checkpoints",
        action="store_true",
        help="Remove the existing target head checkpoint before training each dataset.",
    )
    runner_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return runner_args


runner_args = parse_drift_runner_args()
data_location = "/data/139-1/datasets/merging"
models = ['ViT-B-32']
datasets = ["Cars",'SVHN','MNIST', "SUN397","RESISC45","GTSRB","EuroSAT","DTD"] 

epochs = {
    'Cars': 35,
    'DTD': 76,
    'EuroSAT': 12,
    'GTSRB': 11,
    'MNIST': 5,
    'RESISC45': 15,
    'SUN397': 14,
    'SVHN': 4
}

for model in models:
    for dataset in datasets:
        print('='*100)
        print(f'Finetuning {model} on {dataset} ({runner_args.head_type})')
        print('='*100)
        args = parse_arguments()
        args.lr = 1e-3
        args.epochs = epochs[dataset]
        args.data_location = data_location
        args.train_dataset = dataset + 'Val'
        args.batch_size = 128 # 128 seems like GPUs can't handle this much batch size
        args.model = model
        args.save = f'/data/139-1/users/selkarrat/checkpoints/{model}'
        args.load = f'/data/139-1/users/selkarrat/checkpoints/{model}'

        train_dataset = args.train_dataset
        ckpdir = os.path.join(args.save, train_dataset)
        if runner_args.head_type == "rigid":
            target_ckpt = os.path.join(ckpdir, "trained_rigid_drift_head.pt")
            rigid_movement = True
        else:
            target_ckpt = os.path.join(ckpdir, "trained_drift_head.pt")
            rigid_movement = False

        if runner_args.remove_previous_checkpoints and os.path.isfile(target_ckpt):
            os.remove(target_ckpt)
            print(f"Removed previous checkpoint: {target_ckpt}")

        train_drift(args, rigid_movement=rigid_movement)
