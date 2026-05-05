import copy
import os
import torch
from src.task_vectors import TaskVector


def adapter_weight_averaging(adapter_list):
    """Return a new adapter whose parameters are the average of all adapters in adapter_list."""
    if adapter_list is None or len(adapter_list) == 0:
        raise ValueError("adapter_list must contain at least one adapter.")

    for adapter in adapter_list:
        if not hasattr(adapter, "state_dict"):
            raise TypeError("All elements in adapter_list must be adapter modules.")

    averaged_adapter = copy.deepcopy(adapter_list[0])
    reference_state = adapter_list[0].state_dict()
    avg_state = {}

    with torch.no_grad():
        for key, ref_tensor in reference_state.items():
            if not torch.is_floating_point(ref_tensor):
                avg_state[key] = ref_tensor.clone()
                continue

            accum = ref_tensor.clone().float()
            for adapter in adapter_list[1:]:
                other_state = adapter.state_dict()
                if key not in other_state:
                    raise KeyError(f"Missing parameter '{key}' in one of the adapters.")
                accum += other_state[key].float()

            avg_state[key] = (accum / len(adapter_list)).to(ref_tensor.dtype)

    averaged_adapter.load_state_dict(avg_state, strict=True)
    return averaged_adapter


def adapter_iso_c_merging(adapter_list):
    """
    Return a new adapter merged with an Iso-C-style transform.

    For each floating-point parameter:
    - start from parameter-wise average across adapters
    - if the parameter is a matrix (2D), run SVD and replace singular values
      with their mean (same Iso-C equal-spectrum idea)
    - keep non-floating params unchanged from the first adapter
    """
    if adapter_list is None or len(adapter_list) == 0:
        raise ValueError("adapter_list must contain at least one adapter.")

    for adapter in adapter_list:
        if not hasattr(adapter, "state_dict"):
            raise TypeError("All elements in adapter_list must be adapter modules.")

    merged_adapter = copy.deepcopy(adapter_list[0])
    reference_state = adapter_list[0].state_dict()
    merged_state = {}

    with torch.no_grad():
        for key, ref_tensor in reference_state.items():
            if not torch.is_floating_point(ref_tensor):
                merged_state[key] = ref_tensor.clone()
                continue

            accum = ref_tensor.clone().float()
            for adapter in adapter_list[1:]:
                other_state = adapter.state_dict()
                if key not in other_state:
                    raise KeyError(f"Missing parameter '{key}' in one of the adapters.")
                accum += other_state[key].float()

            avg_tensor = accum / len(adapter_list)
            out_tensor = avg_tensor

            if avg_tensor.ndim == 2 and "text_projection" not in key:
                u, s, vh = torch.linalg.svd(avg_tensor, full_matrices=False)
                s_mean = torch.full_like(s, s.mean())
                out_tensor = torch.linalg.multi_dot((u, torch.diag(s_mean), vh))

            merged_state[key] = out_tensor.to(ref_tensor.dtype)

    merged_adapter.load_state_dict(merged_state, strict=True)
    return merged_adapter


def merge_adapted_finetuned_visual_encoders(
    zeroshot_checkpoint,
    finetuned_checkpoints,
    scaling_coef=None,
    output_path=None,
):
    """
    Merge visual encoders using TaskVector summation relative to one zeroshot checkpoint.

    The merged model is:
      zeroshot + scaling_coef * sum_i (finetuned_i - zeroshot)

    For equal-weight averaging over N finetuned encoders, use scaling_coef=1/N.
    If scaling_coef is None, this function defaults to 1/N.
    """
    if not os.path.isfile(zeroshot_checkpoint):
        raise FileNotFoundError(f"Missing zeroshot checkpoint: {zeroshot_checkpoint}")
    if finetuned_checkpoints is None or len(finetuned_checkpoints) == 0:
        raise ValueError("finetuned_checkpoints must contain at least one checkpoint.")

    task_vectors = []
    for finetuned_checkpoint in finetuned_checkpoints:
        if not os.path.isfile(finetuned_checkpoint):
            raise FileNotFoundError(f"Missing finetuned checkpoint: {finetuned_checkpoint}")
        task_vectors.append(
            TaskVector(
                pretrained_checkpoint=zeroshot_checkpoint,
                finetuned_checkpoint=finetuned_checkpoint,
            )
        )

    merged_task_vector = sum(task_vectors)
    if scaling_coef is None:
        scaling_coef = 1.0 / len(task_vectors)

    merged_model = merged_task_vector.apply_to(
        zeroshot_checkpoint, scaling_coef=scaling_coef
    )

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if hasattr(merged_model, "save"):
            merged_model.save(output_path)
        else:
            torch.save(merged_model, output_path)

    return merged_model
