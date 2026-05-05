import os
import torch
from tqdm import tqdm

import open_clip

from src.datasets.templates import get_templates
from src.datasets.registry import get_dataset

from src.modeling import (
    ClassificationHead,
    DriftClassificationHead,
    PerClassDriftClassificationHead,
    ImageEncoder,
)


def _normalize_drift_mode(drift):
    if drift is False or drift is None:
        return None
    if drift is True:
        return "per_task"
    if drift in {"per_task", "per_class"}:
        return drift
    raise ValueError(f"Unsupported drift mode: {drift}")


def build_classification_head(model, dataset_name, template, data_location, device, drift=None):
    drift_mode = _normalize_drift_mode(drift)
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location
    )
    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device) # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)
            
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)  # [C,1,D]
    zeroshot_weights = zeroshot_weights.squeeze(1)                      # [C,D]
    head_logit_scale = float(logit_scale.exp().detach().cpu().item())
    if drift_mode == "per_task":
        classification_head = DriftClassificationHead(
            normalize=True,
            weights=zeroshot_weights,
            biases=None,
            logit_scale=head_logit_scale,
        )
    elif drift_mode == "per_class":
        classification_head = PerClassDriftClassificationHead(
            normalize=True,
            weights=zeroshot_weights,
            biases=None,
            logit_scale=head_logit_scale,
        )
    else:
        classification_head = ClassificationHead(
            normalize=True,
            weights=zeroshot_weights,
            logit_scale=head_logit_scale,
        )

    return classification_head


def get_classification_head(args, dataset, drift=None):
    drift_mode = _normalize_drift_mode(drift)
    if drift_mode is not None:
        filename = os.path.join(args.save, f'trained_drift_head_{drift_mode}_{dataset}.pt')
        print("Skipping loading of classification head, since drift is enabled. Will build a new head with drift parameters.")
    else:
        filename = os.path.join(args.save, f'head_{dataset}.pt')
        if os.path.exists(filename):
            print(f'Classification head for {args.model} on {dataset} exists at {filename}')
            head = ClassificationHead.load(filename)
            if not hasattr(head, "logit_scale") or head.logit_scale is None:
                print("Loaded head is missing logit_scale; rebuilding head.")
            else:
                if torch.is_tensor(head.logit_scale):
                    head.logit_scale = float(head.logit_scale.detach().cpu().item())
                else:
                    head.logit_scale = float(head.logit_scale)
                # CLIP-style heads generally require a high temperature scale (often ~100).
                # A stale scale near 1 usually indicates an old checkpoint built before fixes.
                if head.logit_scale < 10.0:
                    print(f"Loaded head logit_scale={head.logit_scale:.4f} looks stale; rebuilding head.")
                else:
                    return head
        print(f'Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch.')

    model = ImageEncoder(args, keep_lang=True).model
    template = get_templates(dataset)
    classification_head = build_classification_head(
        model,
        dataset,
        template,
        args.data_location,
        args.device,
        drift=drift_mode,
    )
    os.makedirs(args.save, exist_ok=True)
    classification_head.save(filename)
    return classification_head
