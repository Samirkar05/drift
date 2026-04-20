from src.finetune import create_zeroshot_model
from src.args import parse_arguments
from src.eval import custom_evaluate
import argparse
import os
import sys
import torch
data_location = "/data/139-1/datasets/merging"
models = ['ViT-B-32']
datasets = ["Cars",'SVHN','MNIST', "SUN397","RESISC45","GTSRB","EuroSAT","DTD"] 

BASE_DIR = "/data/139-1/users/selkarrat/checkpoints"
RESULTS_DIR = f"/data/139-1/users/selkarrat/results/{models[0]}/drift_eval"


def parse_drift_eval_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--head-type",
        choices=["drift", "rigid"],
        default="drift",
        help="Which trained head to evaluate: drift -> trained_drift_head.pt, rigid -> trained_rigid_drift_head.pt.",
    )
    drift_eval_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return drift_eval_args


drift_eval_args = parse_drift_eval_args()
if drift_eval_args.head_type == "rigid":
    head_filename = "trained_rigid_drift_head.pt"
    run_tag = "rigid_drift"
else:
    head_filename = "trained_drift_head.pt"
    run_tag = "drift"

checkpoints = {}
task_vectors = {}
for model in models:
    checkpoints[model] = {}
    for dataset in datasets:
        source = os.path.join(BASE_DIR, f"{model}/{dataset}Val")
        checkpoints[model][dataset] = os.path.join(source, head_filename)

# Calculating finetuned models on all datasets
for model in models:
    for checkpoint_path in checkpoints[model].items():
        args = parse_arguments()
        args.eval_datasets = [checkpoint_path[0]]
        args.results_db = RESULTS_DIR
        args.data_location = data_location
        args.model = model
        args.save = f'/data/139-1/users/selkarrat/checkpoints/{model}'
        head_path = checkpoint_path[1]
        DIR = os.path.join(RESULTS_DIR, f"results_{run_tag}_{checkpoint_path[0]}")
        if os.path.isfile(DIR):
            print(f'\nResults already existing: \"results_{run_tag}_{checkpoint_path[0]}\".')
            print("Skipping evaluation...")
            continue
        if not os.path.isfile(head_path):
            print(f'\nCould not find checkpoint: "{head_path}"')
            print("Skipping evaluation...")
            continue

        print('='*100)
        print(f'Evaluating {model}, {run_tag} on {checkpoint_path[0]}, on all datasets')
        print('='*100)

        args.results_db = DIR
        zeroshot_path = create_zeroshot_model(args, train_dataset=checkpoint_path[0])
        image_encoder = torch.load(zeroshot_path)
        custom_evaluate(head_path, image_encoder, args)
