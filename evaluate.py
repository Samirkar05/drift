from src.finetune import create_zeroshot_model
from src.args import parse_arguments
from src.eval import evaluate as evaluate_normal_head, custom_evaluate, custom_evaluate_adapter
from src.heads import get_classification_head
from merge_adapters import (
    adapter_iso_c_merging,
    adapter_weight_averaging,
    merge_adapted_finetuned_visual_encoders,
)
import argparse
import os
import sys
import torch

data_location = "/data/139-1/datasets/merging"
models = ["ViT-B-32"]
datasets = ["Cars", "SVHN", "MNIST", "SUN397", "RESISC45", "GTSRB", "EuroSAT", "DTD"]

BASE_DIR = "/data/139-1/users/selkarrat/checkpoints"
RESULTS_DIR = f"/data/139-1/users/selkarrat/results/{models[0]}/drift_eval"


def parse_eval_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--head",
        choices=[
            "normal",
            "prompt_head",
            "adapter",
            "merged_adapter",
            "iso_adapter",
            "drift_per_task",
            "drift_per_class",
        ],
        default="adapter",
        help=(
            "Head mode: normal -> standard classification head, prompt_head -> load prompt_csc_<dataset>.pt "
            "as the classification head, adapter -> dataset adapter head, "
            "merged_adapter -> averaged adapter head, iso_adapter -> Iso-C merged adapter head, "
            "drift_per_task/per_class -> drift heads."
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=[
            "normal",
            "zeroshot",
            "finetuned",
            "adapted_finetuned",
            "adapted_finetuned_drift_per_task",
            "adapted_finetuned_drift_per_class",
            "merged_adapted_finetuned",
        ],
        default="zeroshot",
        help=(
            "Encoder mode: normal/zeroshot -> zeroshot.pt, finetuned -> finetuned.pt, "
            "adapted_finetuned -> adapted_finetuned.pt, "
            "adapted_finetuned_drift_per_task/per_class -> adapted_finetuned_<head>.pt, "
            "merged_adapted_finetuned -> "
            "TaskVector-merged adapted_finetuned encoders applied to dataset zeroshot base."
        ),
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
    eval_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return eval_args


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


def _load_image_encoder(model, dataset, args, encoder_mode):
    if encoder_mode in {"normal", "zeroshot"}:
        zeroshot_path = create_zeroshot_model(args, train_dataset=dataset)
        return torch.load(zeroshot_path)

    if encoder_mode == "finetuned":
        finetuned_path = os.path.join(BASE_DIR, f"{model}/{dataset}Val", "finetuned.pt")
        if not os.path.isfile(finetuned_path):
            raise FileNotFoundError(f'Missing finetuned checkpoint: "{finetuned_path}"')
        return torch.load(finetuned_path)

    if encoder_mode == "adapted_finetuned":
        adapted_finetuned_path = os.path.join(BASE_DIR, f"{model}/{dataset}Val", "adapted_finetuned.pt")
        if not os.path.isfile(adapted_finetuned_path):
            raise FileNotFoundError(f'Missing adapted finetuned checkpoint: "{adapted_finetuned_path}"')
        return torch.load(adapted_finetuned_path)
    if encoder_mode == "adapted_finetuned_drift_per_task":
        drift_ft_path = os.path.join(BASE_DIR, f"{model}/{dataset}Val", "adapted_finetuned_drift_per_task.pt")
        if not os.path.isfile(drift_ft_path):
            raise FileNotFoundError(f'Missing drift-per-task adapted finetuned checkpoint: "{drift_ft_path}"')
        return torch.load(drift_ft_path)
    if encoder_mode == "adapted_finetuned_drift_per_class":
        drift_ft_path = os.path.join(BASE_DIR, f"{model}/{dataset}Val", "adapted_finetuned_drift_per_class.pt")
        if not os.path.isfile(drift_ft_path):
            raise FileNotFoundError(f'Missing drift-per-class adapted finetuned checkpoint: "{drift_ft_path}"')
        return torch.load(drift_ft_path)

    zeroshot_path = os.path.join(BASE_DIR, f"{model}/{dataset}Val", "zeroshot.pt")
    if not os.path.isfile(zeroshot_path):
        raise FileNotFoundError(f'Missing zeroshot checkpoint: "{zeroshot_path}"')

    finetuned_paths = [
        os.path.join(BASE_DIR, f"{model}/{source_dataset}Val", "adapted_finetuned.pt")
        for source_dataset in datasets
        if os.path.isfile(os.path.join(BASE_DIR, f"{model}/{source_dataset}Val", "adapted_finetuned.pt"))
    ]
    if len(finetuned_paths) == 0:
        raise FileNotFoundError(f"No adapted_finetuned checkpoints found for {model} to merge.")

    return merge_adapted_finetuned_visual_encoders(
        zeroshot_checkpoint=zeroshot_path,
        finetuned_checkpoints=finetuned_paths,
        scaling_coef=1.0 / len(finetuned_paths),
    )


def _load_drift_head_path(model, dataset, head_mode):
    dataset_ckpt_dir = os.path.join(BASE_DIR, f"{model}/{dataset}Val")
    if head_mode == "drift_per_task":
        filename = "trained_drift_head.pt"
    elif head_mode == "drift_per_class":
        filename = "trained_drift_head_per_class.pt"
    else:
        raise ValueError(f"Unsupported drift head mode: {head_mode}")

    drift_head_path = os.path.join(dataset_ckpt_dir, filename)
    if not os.path.isfile(drift_head_path):
        raise FileNotFoundError(f'Missing drift head checkpoint: "{drift_head_path}"')
    return drift_head_path


def _load_prompt_head_path(model, dataset):
    prompt_head_path = os.path.join(BASE_DIR, f"{model}/{dataset}Val", f"prompt_csc_{dataset}.pt")
    if not os.path.isfile(prompt_head_path):
        raise FileNotFoundError(f'Missing prompt head checkpoint: "{prompt_head_path}"')
    return prompt_head_path


def _load_prompt_head_as_classification_head(model, dataset, args):
    prompt_head_path = _load_prompt_head_path(model, dataset)
    prompt_head_obj = torch.load(prompt_head_path, map_location="cpu")

    # Build/load the standard head module, then overwrite its weights from prompt checkpoint.
    classification_head = get_classification_head(args, dataset, drift=False)

    if isinstance(prompt_head_obj, torch.Tensor):
        prompt_weights = prompt_head_obj
    elif isinstance(prompt_head_obj, dict):
        if "weight" not in prompt_head_obj:
            raise ValueError(
                f'Unsupported prompt head checkpoint format at "{prompt_head_path}": missing "weight" key.'
            )
        prompt_weights = prompt_head_obj["weight"]
        if "bias" in prompt_head_obj and prompt_head_obj["bias"] is not None:
            if classification_head.bias.shape != prompt_head_obj["bias"].shape:
                raise ValueError(
                    f'Prompt bias shape mismatch for {dataset}: expected {tuple(classification_head.bias.shape)}, '
                    f'got {tuple(prompt_head_obj["bias"].shape)}.'
                )
            classification_head.bias.data.copy_(prompt_head_obj["bias"].to(classification_head.bias.device))
    else:
        raise ValueError(
            f'Unsupported prompt head checkpoint type at "{prompt_head_path}": {type(prompt_head_obj)}'
        )

    if classification_head.weight.shape != prompt_weights.shape:
        raise ValueError(
            f'Prompt head shape mismatch for {dataset}: expected {tuple(classification_head.weight.shape)}, '
            f'got {tuple(prompt_weights.shape)}.'
        )

    classification_head.weight.data.copy_(prompt_weights.to(classification_head.weight.device))
    return classification_head


eval_args = parse_eval_args()
run_tag = f"{eval_args.head}_{eval_args.encoder}"
drift_heads = {"drift_per_task", "drift_per_class"}

if eval_args.head in drift_heads and eval_args.encoder not in {"normal", "zeroshot"}:
    raise ValueError("Drift heads currently support only normal/zeroshot encoder evaluation.")

if eval_args.delete_results:
    delete_results(RESULTS_DIR, run_tag, target_dataset=eval_args.delete_dataset)
    if eval_args.delete_only:
        print("Deletion requested with --delete-only; exiting.")
        sys.exit(0)

for model in models:
    merged_adapter = None
    if eval_args.head in {"merged_adapter", "iso_adapter"}:
        adapters = []
        for source_dataset in datasets:
            adapter_path = os.path.join(BASE_DIR, f"{model}/{source_dataset}Val", "trained_1_layer_mlp_adapter.pt")
            if os.path.isfile(adapter_path):
                adapters.append(torch.load(adapter_path, map_location="cpu"))

        if len(adapters) == 0:
            print(f'\nNo adapter checkpoints found for {model} to merge. Skipping model...')
            continue
        if eval_args.head == "merged_adapter":
            merged_adapter = adapter_weight_averaging(adapters)
        else:
            merged_adapter = adapter_iso_c_merging(adapters)

    for dataset in datasets:
        args = parse_arguments()
        args.eval_datasets = [dataset]
        args.results_db = RESULTS_DIR
        args.data_location = data_location
        args.model = model
        args.save = f"/data/139-1/users/selkarrat/checkpoints/{model}"

        result_file_path = _result_file_path(RESULTS_DIR, run_tag, dataset)
        if os.path.isfile(result_file_path):
            print(f'\nResults already existing: "results_{run_tag}_{dataset}".')
            print("Skipping evaluation...")
            continue

        adapter_path = os.path.join(BASE_DIR, f"{model}/{dataset}Val", "trained_1_layer_mlp_adapter.pt")
        if eval_args.head == "adapter" and not os.path.isfile(adapter_path):
            print(f'\nCould not find adapter checkpoint: "{adapter_path}"')
            print("Skipping evaluation...")
            continue

        try:
            image_encoder = _load_image_encoder(model, dataset, args, eval_args.encoder)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n{exc}")
            print("Skipping evaluation...")
            continue

        print("=" * 100)
        print(f"Evaluating {model}, head={eval_args.head}, encoder={eval_args.encoder} on {dataset}")
        print("=" * 100)

        args.results_db = result_file_path
        if eval_args.head == "normal":
            evaluate_normal_head(image_encoder, args)
        elif eval_args.head == "prompt_head":
            try:
                prompt_head = _load_prompt_head_as_classification_head(model, dataset, args)
            except (FileNotFoundError, ValueError) as exc:
                print(f"\n{exc}")
                print("Skipping evaluation...")
                continue
            custom_evaluate(prompt_head, image_encoder, args)
        elif eval_args.head in drift_heads:
            try:
                drift_head_path = _load_drift_head_path(model, dataset, eval_args.head)
            except FileNotFoundError as exc:
                print(f"\n{exc}")
                print("Skipping evaluation...")
                continue
            custom_evaluate(drift_head_path, image_encoder, args)
        elif eval_args.head == "adapter":
            custom_evaluate_adapter(
                adapter_path,
                image_encoder,
                dataset,
                args,
                use_adapter_head=True,
            )
        else:
            custom_evaluate_adapter(
                merged_adapter,
                image_encoder,
                dataset,
                args,
                use_adapter_head=True,
            )
