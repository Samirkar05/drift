from src.task_vectors import TaskVector
import torch
import os
import glob
import json
import shutil
from src.args import parse_arguments
from src.eval import evaluate
from src.main_results import read_accuracies_json
from main_IsoC import *
from metrics import *
from main_TSV import compute_and_sum_svd_mem_reduction
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt


data_location = "/data/139-1/datasets/merging"
BASE_DIR = "/data/139-1/users/selkarrat/checkpoints"
CUSTOM_DIR = "/data/139-1/users/selkarrat/checkpoints/ViT-B-16/custom"
RESULTS_DIR = "/data/139-1/users/selkarrat/results/results"
pretrained_checkpoint = "/data/139-1/users/selkarrat/checkpoints/ViT-B-16/MNISTVal/zeroshot.pt"
models = ['ViT-B-16']
#datasets = ['MNIST', 'SVHN', "Cars", "SUN397","RESISC45","GTSRB","EuroSAT","DTD"]
datasets = ['MNIST', 'SVHN', "Cars", "SUN397","RESISC45","GTSRB","EuroSAT","DTD"]


checkpoints = {}
task_vectors = {}
for model in models:
    checkpoints[model] = {}
    task_vectors[model] = {}

    for dataset in datasets:
        source = os.path.join(BASE_DIR, f"{model}/{dataset}Val")
        checkpoints[model][dataset] = {
            "zeroshot": os.path.join(source, "zeroshot.pt"),
            "finetuned": os.path.join(source, "finetuned.pt")
        }
        task_vectors[model][dataset] = TaskVector(checkpoints[model][dataset]["zeroshot"],checkpoints[model][dataset]["finetuned"])


scaling_coefs = [0.3,0.125]
results_dirs = [f"{RESULTS_DIR}_task-arithmetic", f"{RESULTS_DIR}_weight-averaging", f"{RESULTS_DIR}_Iso-C", f"{RESULTS_DIR}_TSV-LR", f"{RESULTS_DIR}_TSV-LR-Avg"]

def save_vector_checkpoint(vec: dict, checkpoint_name: str) -> str:
    save_dir = CUSTOM_DIR
    if not checkpoint_name:
        raise ValueError("checkpoint_name must be a non-empty string.")

    checkpoint_file = checkpoint_name if checkpoint_name.endswith(".pt") else f"{checkpoint_name}.pt"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, checkpoint_file)

    vector_cpu = {}
    for key, value in vec.items():
        if torch.is_tensor(value):
            vector_cpu[key] = value.detach().cpu()
        else:
            vector_cpu[key] = value

    torch.save(vector_cpu, checkpoint_path)
    print(f"Saved vector checkpoint to: {checkpoint_path}")
    return checkpoint_path

def load_vector_checkpoint_if_exists(checkpoint_name: str):
    save_dir = CUSTOM_DIR
    if not checkpoint_name:
        raise ValueError("checkpoint_name must be a non-empty string.")

    checkpoint_file = checkpoint_name if checkpoint_name.endswith(".pt") else f"{checkpoint_name}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_file)
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found, will compute vector: {checkpoint_path}")
        return None

    vec = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loaded vector checkpoint from: {checkpoint_path}")
    return vec

def _as_vector_dict(vector_like):
    if isinstance(vector_like, TaskVector):
        return vector_like.vector
    if isinstance(vector_like, dict):
        return vector_like
    raise TypeError(f"Unsupported vector type: {type(vector_like)}")

def _sorted_singular_spectrum(vector_like):
    vector_dict = _as_vector_dict(vector_like)
    all_singular_values = []
    for tensor in vector_dict.values():
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            continue
        s_values = torch.linalg.svdvals(tensor.detach().float().cpu())
        all_singular_values.append(s_values)

    if not all_singular_values:
        return torch.tensor([], dtype=torch.float32)

    concatenated = torch.cat(all_singular_values)
    return torch.sort(concatenated, descending=True).values

def _layer_singular_values(vector_like, layer_key: str):
    vector_dict = _as_vector_dict(vector_like)
    tensor = vector_dict[layer_key]
    return torch.linalg.svdvals(tensor.detach().float().cpu())

def replace_karcher_singular_values_with_isoc(karcher_vector, isoc_vector):
    karcher_dict = _as_vector_dict(karcher_vector)
    isoc_dict = _as_vector_dict(isoc_vector)
    mixed_vector = {}

    for key, k_tensor in karcher_dict.items():
        i_tensor = isoc_dict.get(key)
        if (
            i_tensor is None
            or not torch.is_tensor(k_tensor)
            or not torch.is_tensor(i_tensor)
            or k_tensor.ndim != 2
            or i_tensor.ndim != 2
            or k_tensor.shape != i_tensor.shape
        ):
            mixed_vector[key] = k_tensor
            continue

        k_dtype = k_tensor.dtype
        k_device = k_tensor.device
        k_work = k_tensor.detach().float()
        i_work = i_tensor.detach().to(device=k_device, dtype=torch.float32)

        Uk, _, Vhk = torch.linalg.svd(k_work, full_matrices=False)
        _, Si, _ = torch.linalg.svd(i_work, full_matrices=False)
        mixed = Uk @ torch.diag(Si) @ Vhk
        mixed_vector[key] = mixed.to(dtype=k_dtype, device=k_device)

    return mixed_vector

def plot_three_vector_random_layer_spectra(
    task_arithmetic_vector,
    isoc_vector,
    karcher_vector,
    save_path: str,
    num_layers: int = 2
) -> None:
    ta_dict = _as_vector_dict(task_arithmetic_vector)
    isoc_dict = _as_vector_dict(isoc_vector)
    karcher_dict = _as_vector_dict(karcher_vector)

    common_keys = set(ta_dict.keys()) & set(isoc_dict.keys()) & set(karcher_dict.keys())
    candidate_keys = sorted(
        key for key in common_keys
        if torch.is_tensor(ta_dict[key])
        and torch.is_tensor(isoc_dict[key])
        and torch.is_tensor(karcher_dict[key])
        and ta_dict[key].ndim == 2
        and isoc_dict[key].ndim == 2
        and karcher_dict[key].ndim == 2
    )

    if not candidate_keys:
        print("No common 2D layers found to plot singular value spectra.")
        return

    num_layers = min(num_layers, len(candidate_keys))
    perm = torch.randperm(len(candidate_keys))[:num_layers].tolist()
    selected_layers = [candidate_keys[idx] for idx in perm]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, num_layers, figsize=(8 * num_layers, 5), squeeze=False)
    axes = axes[0]

    for ax, layer_key in zip(axes, selected_layers):
        spectra = [
            ("Task Arithmetic", _layer_singular_values(task_arithmetic_vector, layer_key), "#1f77b4"),
            ("Iso-C", _layer_singular_values(isoc_vector, layer_key), "#d62728"),
            ("Karcher", _layer_singular_values(karcher_vector, layer_key), "#2ca02c"),
        ]
        for label, s_values, color in spectra:
            x = torch.arange(1, s_values.numel() + 1)
            mean_sv = float(s_values.mean().item())
            legend_label = f"{label} (mean={mean_sv:.3e})"
            ax.plot(x.numpy(), s_values.numpy(), label=legend_label, linewidth=1.8, color=color)

        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value")
        ax.set_title(layer_key)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Singular Value Spectrum on Random Layers: TA vs Iso-C vs Karcher")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"Saved random-layer singular spectrum plot to: {save_path}")
    print(f"Selected layers: {selected_layers}")

def evaluate_task_TA():
    # Calculating finetuned models on all datasets
    for model in models:
        args = parse_arguments()
        args.eval_datasets = datasets
        args.results_db = results_dirs[0]
        args.data_location = data_location
        args.model = model
        args.save = f'/data/139-1/users/selkarrat/checkpoints/{model}'

        print('='*100)
        print(f'Evaluating {model}, task arithmetic with scaling coefficient {scaling_coefs[1]}, on all datasets')
        print('='*100)

        if os.path.isfile(results_dirs[0]):
            print(f'\nResults already existing.')
            print("Skipping evaluation...")
            continue

        finetuned = [task_vectors[model][dataset] for dataset in datasets]
        task_vector_sum = sum(finetuned)
        image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coefs[1]) 
        evaluate(image_encoder,args)

def evaluate_Iso_C():
    # Calculating finetuned models on all datasets
    for model in models:
        args = parse_arguments()
        args.eval_datasets = datasets
        args.results_db = results_dirs[4]
        args.data_location = data_location
        args.model = model
        args.save = f'/data/139-1/users/selkarrat/checkpoints/{model}'

        print('='*100)
        print(f'Evaluating {model}, Iso-C with scaling coefficient on all datasets')
        print('='*100)

        if os.path.isfile(args.results_db ):
            print(f'\nResults already existing.')
            print("Skipping evaluation...")
            continue

        finetuned = [task_vectors[model][dataset] for dataset in datasets]
        vec = TSV_low_rank(finetuned,args,average_singular_values = True)
        TaskVector_specific = TaskVector(vector=vec)
        image_encoder = TaskVector_specific.apply_to(pretrained_checkpoint, scaling_coef=1/8) 
        evaluate(image_encoder,args)

def evaluate_TSVM():
    # Calculating finetuned models on all datasets
    for model in models:
        args = parse_arguments()
        args.eval_datasets = datasets
        args.results_db = f"{RESULTS_DIR}_TSV-M"
        args.data_location = data_location
        args.model = model
        args.save = f'/data/139-1/users/selkarrat/checkpoints/{model}'

        print('='*100)
        print(f'Evaluating {model}, TSV-M with scaling coefficient on all datasets')
        print('='*100)

        if os.path.isfile(args.results_db ):
            print(f'\nResults already existing.')
            print("Skipping evaluation...")
            continue

        finetuned = [task_vectors[model][dataset] for dataset in datasets]
        vec = compute_and_sum_svd_mem_reduction(finetuned, config=args)
        TaskVector_specific = TaskVector(vector=vec)
        image_encoder = TaskVector_specific.apply_to(pretrained_checkpoint, scaling_coef=1) 
        evaluate(image_encoder,args)

def evaluate_vector(
    vector,
    run_name: str = "custom-vector",
    scaling_coef: float = 1.0,
    model: str | None = None,
    eval_dataset_list: list[str] | None = None,
    results_db_path: str | None = None,
    device: str | None = None,
):
    args = parse_arguments()
    args.eval_datasets = eval_dataset_list if eval_dataset_list is not None else datasets
    args.results_db = results_db_path if results_db_path is not None else f"{RESULTS_DIR}_{run_name}"
    args.data_location = data_location
    args.model = model if model is not None else models[0]
    if device is not None:
        args.device = device
    args.save = f"/data/139-1/users/selkarrat/checkpoints/{args.model}"

    print("=" * 100)
    print(f"Evaluating {args.model}, vector run '{run_name}' with scaling coefficient {scaling_coef}")
    print("=" * 100)

    if os.path.isfile(args.results_db):
        print("\nResults already existing.")
        print("Skipping evaluation...")
        return args.results_db

    vector_dict = _as_vector_dict(vector)
    task_vector = TaskVector(vector=vector_dict)
    image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    evaluate(image_encoder, args)
    return args.results_db


def _sanitize_tag(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return safe or "x"


def _coef_to_tag(coef: float) -> str:
    return f"{coef:.1f}".replace("-", "m").replace(".", "p")


def _chunk_list(items: list, n_chunks: int) -> list[list]:
    if n_chunks <= 0:
        return [items]
    return [items[i::n_chunks] for i in range(n_chunks) if items[i::n_chunks]]


def _evaluate_isop_chunk(
    p_vector_pairs: list[tuple[float, dict]],
    gpu_id: int,
    model_name: str,
    eval_dataset_list: list[str],
    scaling_coef: float,
):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    for p, isop_vector in p_vector_pairs:
        evaluate_vector(
            vector=isop_vector,
            run_name=f"iso_p_{str(p).replace('.', 'p')}",
            scaling_coef=scaling_coef,
            model=model_name,
            eval_dataset_list=eval_dataset_list,
            device=device,
        )


def _evaluate_scaling_chunk(
    coefficients_chunk: list[float],
    vector,
    search_tag: str,
    scaling_dir: str,
    model_name: str,
    eval_dataset_list: list[str],
    gpu_id: int,
) -> list[tuple[float, str]]:
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    completed_trials = []
    for coef in coefficients_chunk:
        coef_value = float(coef)
        trial_run_name = f"{search_tag}_coef_{_coef_to_tag(coef_value)}"
        trial_results_path = os.path.join(scaling_dir, f"results_{trial_run_name}")
        output_path = evaluate_vector(
            vector=vector,
            run_name=trial_run_name,
            scaling_coef=coef_value,
            model=model_name,
            eval_dataset_list=eval_dataset_list,
            results_db_path=trial_results_path,
            device=device,
        )
        completed_trials.append((coef_value, output_path))
    return completed_trials


def _load_results_payload(results_path: str) -> dict:
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Results file is empty: {results_path}")

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        # Fallback for single-object JSON files.
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)


def _extract_top1_scores(payload: dict) -> dict[str, float]:
    scores = {}
    for key, value in payload.items():
        if isinstance(key, str) and key.endswith(":top1") and isinstance(value, (int, float)):
            dataset_name = key.split(":", 1)[0]
            scores[dataset_name] = float(value)
    return scores


def _subset_average(scores: dict[str, float], subset: list[str]) -> float:
    missing = [dataset_name for dataset_name in subset if dataset_name not in scores]
    if missing:
        raise ValueError(f"Missing datasets in scores: {missing}")
    return sum(scores[dataset_name] for dataset_name in subset) / len(subset)


def _best_existing_average(subset: list[str], results_root: str) -> dict:
    best = None
    for path in sorted(glob.glob(os.path.join(results_root, "results_*"))):
        try:
            payload = _load_results_payload(path)
            scores = _extract_top1_scores(payload)
            avg = _subset_average(scores, subset)
        except Exception:
            continue

        run_name = os.path.basename(path)
        if run_name.startswith("results_"):
            run_name = run_name[len("results_") :]

        row = {"run_name": run_name, "results_path": path, "average_top1": avg}
        if best is None or row["average_top1"] > best["average_top1"]:
            best = row

    if best is None:
        raise ValueError(
            f"No existing run in '{results_root}' has all selected datasets: {subset}"
        )
    return best


def scaling_searcher(
    vector,
    run_name: str = "custom-vector",
    model: str | None = None,
    eval_dataset_list: list[str] | None = None,
    coefficients: list[float] | None = None,
):
    selected_datasets = eval_dataset_list if eval_dataset_list is not None else datasets
    if not selected_datasets:
        raise ValueError("No datasets provided for scaling search.")

    if coefficients is None:
        coefficients = [round(step / 10, 1) for step in range(1, 21)]

    results_root = os.path.dirname(RESULTS_DIR)
    scaling_dir = os.path.join(results_root, "scaling")
    os.makedirs(scaling_dir, exist_ok=True)
    baseline = _best_existing_average(selected_datasets, results_root)

    dataset_tag = "_".join(_sanitize_tag(dataset_name) for dataset_name in selected_datasets)
    search_tag = f"{run_name}_search_all_{dataset_tag}"

    model_name = model if model is not None else models[0]
    coef_to_path: dict[float, str] = {}
    num_gpus = torch.cuda.device_count()
    worker_gpu_ids = list(range(min(4, num_gpus)))

    if worker_gpu_ids:
        coefficient_chunks = _chunk_list(coefficients, len(worker_gpu_ids))
        with ThreadPoolExecutor(max_workers=len(coefficient_chunks)) as executor:
            futures = []
            for gpu_id, chunk in zip(worker_gpu_ids, coefficient_chunks):
                futures.append(
                    executor.submit(
                        _evaluate_scaling_chunk,
                        chunk,
                        vector,
                        search_tag,
                        scaling_dir,
                        model_name,
                        selected_datasets,
                        gpu_id,
                    )
                )
            for future in futures:
                for coef_value, path in future.result():
                    coef_to_path[coef_value] = path
    else:
        for coef in coefficients:
            coef_value = float(coef)
            trial_run_name = f"{search_tag}_coef_{_coef_to_tag(coef_value)}"
            trial_results_path = os.path.join(scaling_dir, f"results_{trial_run_name}")
            output_path = evaluate_vector(
                vector=vector,
                run_name=trial_run_name,
                scaling_coef=coef_value,
                model=model_name,
                eval_dataset_list=selected_datasets,
                results_db_path=trial_results_path,
                device="cpu",
            )
            coef_to_path[coef_value] = output_path

    trials = []
    best_trial = None
    for coef in coefficients:
        coef_value = float(coef)
        trial_results_path = coef_to_path[coef_value]

        payload = _load_results_payload(trial_results_path)
        scores = _extract_top1_scores(payload)
        avg_top1 = _subset_average(scores, selected_datasets)
        row = {
            "coefficient": coef_value,
            "average_top1": avg_top1,
            "results_path": trial_results_path,
        }
        trials.append(row)

        if best_trial is None or row["average_top1"] > best_trial["average_top1"]:
            best_trial = row

    if best_trial is None:
        raise RuntimeError("No trial results were produced.")

    better_than_baseline = best_trial["average_top1"] > baseline["average_top1"]
    report = {
        "run_name": run_name,
        "model": model if model is not None else models[0],
        "selected_datasets": selected_datasets,
        "coefficients_tested": [float(c) for c in coefficients],
        "baseline": baseline,
        "trials": trials,
        "best_trial": best_trial,
        "better_than_baseline": better_than_baseline,
        "delta_vs_baseline": best_trial["average_top1"] - baseline["average_top1"],
    }

    report_path = os.path.join(scaling_dir, f"{run_name}_scaling_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Saved scaling search report to: {report_path}")

    if not better_than_baseline:
        print(
            "Best scaling coefficient did not beat existing baseline on selected datasets. "
            f"Best={best_trial['average_top1']:.6f}, baseline={baseline['average_top1']:.6f}"
        )

    best_results_output_path = os.path.join(scaling_dir, f"results_{run_name}")
    shutil.copyfile(best_trial["results_path"], best_results_output_path)
    print(f"Saved best-coefficient results to: {best_results_output_path}")

    best_meta_path = os.path.join(scaling_dir, f"{run_name}_best_scaling.json")
    best_meta = {
        "run_name": run_name,
        "selected_datasets": selected_datasets,
        "best_coefficient": best_trial["coefficient"],
        "best_average_top1": best_trial["average_top1"],
        "baseline_average_top1": baseline["average_top1"],
        "source_results_path": best_trial["results_path"],
    }
    with open(best_meta_path, "w", encoding="utf-8") as f:
        json.dump(best_meta, f, indent=2, sort_keys=True)
    print(f"Saved best scaling metadata to: {best_meta_path}")

    return report


"""         tensor_vec = list(vec.values())
        print(len(vec))
        print([tensor.shape for tensor in tensor_vec])
        print(tensor_vec[4]) """

def flatten_TaskVector(task_vector):
    tensor_list = list(task_vector.vector.values())
    component_tensors = [torch.flatten(torch.Tensor(T)) for T in tensor_list]
    flattened_task_vector = torch.cat(component_tensors)
    return flattened_task_vector

def cosine_similarity_matrix(vectors):
    # Build cosine similarity matrix.
    cosine_similarity_matrix = []
    for tvi in vectors:
        row = []
        for tvj in vectors:
            sim = torch.nn.functional.cosine_similarity(tvi, tvj, dim=0).item()
            row.append(sim)
        cosine_similarity_matrix.append(row)

    # Print as table.
    name_width = max(len(name) for name in datasets)
    cell_width = 9
    header = f"{'Dataset':<{name_width}} | " + " ".join(f"{name:>{cell_width}}" for name in datasets)
    print(header)
    print("-" * len(header))
    for row_name, row_vals in zip(datasets, cosine_similarity_matrix):
        row_str = f"{row_name:<{name_width}} | " + " ".join(f"{val:>{cell_width}.4f}" for val in row_vals)
        print(row_str)
    return cosine_similarity_matrix


def calculate_nai(dataset_names: list[str]) -> dict:
    accuracies = read_accuracies_json()
    # NAI = (acc(task_arithmetic) - acc(zeroshot)) / (acc(finetuned on dataset) - acc(zeroshot))
    zeroshot_acc = accuracies.get("zeroshot")

    if zeroshot_acc is None:
        return {}

    nai = {}
    for dataset in dataset_names:
        task_arithmetic_acc = accuracies.get(f"task_arithmetic_{dataset}", accuracies.get("task_arithmetic"))
        zeroshot_dataset_acc = accuracies.get(f"zeroshot_{dataset}", accuracies.get("zeroshot"))
        finetuned_acc = accuracies.get(dataset)
        if finetuned_acc is None or task_arithmetic_acc is None or zeroshot_dataset_acc is None:
            nai[dataset] = None
            continue
        numerator = task_arithmetic_acc - zeroshot_dataset_acc
        denominator = finetuned_acc - zeroshot_dataset_acc
        nai[dataset] = None if denominator == 0 else numerator / denominator

    task_arithmetic_avg = accuracies.get("task_arithmetic")
    database_finetuned_acc = accuracies.get("database_finetuned")
    if database_finetuned_acc is not None and task_arithmetic_avg is not None:
        numerator_overall = task_arithmetic_avg - zeroshot_acc
        denominator_overall = database_finetuned_acc - zeroshot_acc
        nai["overall"] = None if denominator_overall == 0 else numerator_overall / denominator_overall

    return nai


def plot_csm_vs_nai(csm_ta_row: list[float], nai: dict, dataset_names: list[str], save_path: str) -> None:
    x_vals = []
    y_vals = []
    labels = []
    for i, dataset in enumerate(dataset_names):
        nai_val = nai.get(dataset)
        if nai_val is None:
            continue
        x_vals.append(csm_ta_row[i])
        y_vals.append(nai_val)
        labels.append(dataset)

    if not x_vals:
        print("No valid points to plot for CSM vs NAI.")
        return

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 6))
    for idx, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
        plt.scatter(x, y, color=cmap(idx % 10), label=label, s=70)

    if len(x_vals) >= 2:
        x_tensor = torch.tensor(x_vals, dtype=torch.float32)
        y_tensor = torch.tensor(y_vals, dtype=torch.float32)
        x_centered = x_tensor - x_tensor.mean()
        y_centered = y_tensor - y_tensor.mean()
        denom = torch.sqrt((x_centered.pow(2).sum()) * (y_centered.pow(2).sum()))
        corr_text = "Pearson r = N/A"
        if denom.item() > 0:
            corr = (x_centered * y_centered).sum() / denom
            corr_text = f"Pearson r = {corr.item():.4f}"
    else:
        corr_text = "Pearson r = N/A"

    plt.xlabel("Cosine similarity with TA (csm_TA[-1])")
    plt.ylabel("NAI")
    plt.title("CSM(TA, Dataset TV) vs NAI")
    plt.grid(alpha=0.3)
    plt.legend(title=corr_text)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot to: {save_path}")


def plot_binet_cauchy_distance_matrix(task_names: list[str], matrix: torch.Tensor, save_path: str) -> None:
    plt.figure(figsize=(9, 7))
    im = plt.imshow(matrix.cpu().numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Average binet-cauchy distance")

    plt.xticks(range(len(task_names)), task_names, rotation=45, ha="right")
    plt.yticks(range(len(task_names)), task_names)
    plt.xlabel("Task vector j")
    plt.ylabel("Task vector i (as TA_vector)")
    plt.title("Pairwise Binet-Cauchy Distance Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j].item()
            text = "nan" if value != value else f"{value:.3f}"
            plt.text(j, i, text, ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"Saved binet-cauchy matrix plot to: {save_path}")


def plot_combined_distance_curves(
    task_names: list[str],
    ta_values: list[float],
    isoc_values: list[float],
    distance_name: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(task_names, ta_values, marker="o", linewidth=1.8, label="Task Arithmetic")
    plt.plot(task_names, isoc_values, marker="s", linewidth=1.8, label="IsoC")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Tasks")
    plt.ylabel("Average distance")
    plt.title(f"Task Arithmetic vs IsoC ({distance_name})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"Saved combined TA/IsoC plot to: {save_path}")

def plot_combined_distance_vs_nai(
    task_names: list[str],
    ta_distances: list[float],
    isoc_distances: list[float],
    ta_nai_by_dataset: dict,
    isoc_nai_by_dataset: dict,
    distance_name: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(8, 6))

    ta_x_vals = []
    ta_y_vals = []
    isoc_x_vals = []
    isoc_y_vals = []
    labels = []
    for idx, task in enumerate(task_names):
        if idx >= len(ta_distances) or idx >= len(isoc_distances):
            continue
        ta_x = ta_distances[idx]
        isoc_x = isoc_distances[idx]
        ta_y = ta_nai_by_dataset.get(task)
        isoc_y = isoc_nai_by_dataset.get(task)
        if ta_y is None or isoc_y is None:
            continue
        if ta_x != ta_x or isoc_x != isoc_x:
            continue

        ta_x_vals.append(float(ta_x))
        ta_y_vals.append(float(ta_y))
        isoc_x_vals.append(float(isoc_x))
        isoc_y_vals.append(float(isoc_y))
        labels.append(task)

    if not ta_x_vals:
        print("No valid points to plot for combined distance vs NAI.")
        return

    cmap = plt.get_cmap("tab10")
    for i, task in enumerate(labels):
        c = cmap(i % 10)
        plt.scatter(ta_x_vals[i], ta_y_vals[i], color=c, marker="o", s=70)
        plt.scatter(isoc_x_vals[i], isoc_y_vals[i], color=c, marker="s", s=70)

    method_handles = [
        plt.Line2D([], [], color="black", marker="o", linestyle="", label="Task Arithmetic"),
        plt.Line2D([], [], color="black", marker="s", linestyle="", label="IsoC"),
    ]
    dataset_handles = [plt.Line2D([], [], color=cmap(i % 10), marker="o", linestyle="", label=task) for i, task in enumerate(labels)]
    plt.legend(handles=method_handles + dataset_handles, ncol=2, fontsize=8)
    plt.xlabel(f"Average distance ({distance_name})")
    plt.ylabel("NAI")
    plt.title(f"Task Arithmetic vs IsoC: {distance_name} vs NAI")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"Saved combined distance-vs-NAI plot to: {save_path}")
