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
    parser.add_argument(
        "--delete-results",
        action="store_true",
        help="Delete existing result files for this run tag before evaluation.",
    )
    parser.add_argument(
        "--delete-dataset",
        choices=datasets,
        default=None,
        help="Delete results for one dataset only (requires --delete-results).",
    )
    parser.add_argument(
        "--delete-only",
        action="store_true",
        help="Delete results and exit without running evaluation.",
    )
    drift_eval_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return drift_eval_args


def _result_file_path(results_dir, run_tag, dataset):
    return os.path.join(results_dir, f"results_{run_tag}_{dataset}")


def delete_results(results_dir, run_tag, target_dataset=None):
    datasets_to_delete = [target_dataset] if target_dataset is not None else datasets
    deleted = 0

    for dataset in datasets_to_delete:
        result_file = _result_file_path(results_dir, run_tag, dataset)
        if os.path.isfile(result_file):
            os.remove(result_file)
            deleted += 1
            print(f'Deleted result file: "{result_file}"')
        else:
            print(f'No result file to delete for {dataset}: "{result_file}"')

    print(f"Deleted {deleted} result file(s).")
    return deleted


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

if drift_eval_args.delete_results:
    delete_results(RESULTS_DIR, run_tag, target_dataset=drift_eval_args.delete_dataset)
    if drift_eval_args.delete_only:
        print("Deletion requested with --delete-only; exiting.")
        sys.exit(0)

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
        DIR = _result_file_path(RESULTS_DIR, run_tag, checkpoint_path[0])
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
