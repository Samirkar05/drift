from src.finetune import create_zeroshot_model
from src.args import parse_arguments
from src.eval import custom_evaluate
import os
import torch
data_location = "/data/139-1/datasets/merging"
models = ['ViT-B-32']
#datasets = ['MNIST', 'SVHN', "Cars", "SUN397","RESISC45","GTSRB","EuroSAT","DTD"]
datasets = ["MNIST", "SVHN", "Cars", "SUN397"]
BASE_DIR = "/data/139-1/users/selkarrat/checkpoints"
RESULTS_DIR = f"/data/139-1/users/selkarrat/results/{models[0]}/drift_eval"
checkpoints = {}
task_vectors = {}
for model in models:
    checkpoints[model] = {}
    for dataset in datasets:
        source = os.path.join(BASE_DIR, f"{model}/{dataset}Val")
        checkpoints[model][dataset] = {
            "drift": os.path.join(source, "trained_drift_head.pt"),
        }

# Calculating finetuned models on all datasets
for model in models:
    for checkpoint_path in checkpoints[model].items():
        args = parse_arguments()
        args.eval_datasets = [checkpoint_path[0]]
        args.results_db = RESULTS_DIR
        args.data_location = data_location
        args.model = model
        args.save = f'/data/139-1/users/selkarrat/checkpoints/{model}'
        for checkpoint in checkpoint_path[1].items():
            if checkpoint[0] == "zeroshot":
                continue
            DIR= os.path.join(RESULTS_DIR, f"results_{checkpoint[0]}_{checkpoint_path[0]}")
            if os.path.isfile(DIR):
                print(f'\nResults already existing: \"results_{checkpoint[0]}_{checkpoint_path[0]}\".')
                print("Skipping evaluation...")
                continue

            print('='*100)
            print(f'Evaluating {model}, {checkpoint[0]} on {checkpoint_path[0]}, on all datasets')
            print('='*100)

            args.results_db = DIR
            zeroshot_path = create_zeroshot_model(args, train_dataset=checkpoint_path[0])
            image_encoder = torch.load(zeroshot_path)
            custom_evaluate(checkpoints[model][checkpoint_path[0]], image_encoder, args)
