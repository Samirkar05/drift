import os
import time

import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier, AdapterImageClassifier, TaskAdapter
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head


import src.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import utils


def create_zeroshot_model(args, train_dataset=None, overwrite=False):
    """
    Build and save a zero-shot image encoder checkpoint.

    Saves to:
    - {args.save}/{train_dataset}/zeroshot.pt if train_dataset is provided
    - {args.save}/zeroshot.pt otherwise
    """
    if args.save is None:
        raise ValueError("args.save must be set to save the zero-shot model.")

    dataset_name = train_dataset if train_dataset is not None else args.train_dataset
    if dataset_name is not None:
        target_dir = os.path.join(args.save, dataset_name)
    else:
        target_dir = args.save

    os.makedirs(target_dir, exist_ok=True)
    zs_path = os.path.join(target_dir, "zeroshot.pt")

    if os.path.exists(zs_path) and not overwrite:
        print(f"Zero-shot model already exists at {zs_path}.")
        return zs_path

    print("Building zero-shot image encoder.")
    image_encoder = ImageEncoder(args, keep_lang=False)
    image_encoder.save(zs_path)
    return zs_path

def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if checkpoints already exist
    zs_path = os.path.join(args.save, train_dataset, 'checkpoint_0.pt')  
    ft_path = os.path.join(args.save, train_dataset, f'checkpoint_{args.epochs}.pt')
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        print("Loading image encoder.")
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    # Save the starting zero-shot checkpoint in the model folder.
    if args.save is not None:
        create_zeroshot_model(args, train_dataset=train_dataset)

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    # Evaluate
    image_encoder = model.module.image_encoder
    evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(ckpdir, 'zeroshot.pt')  
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)
        return zs_path, ft_path


from src.modeling import ClassificationHead, ImageClassifier

def train_drift(args, rigid_movement= False):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)
    if rigid_movement is not False:
        print("Training with rigid movement (no drift).")
        trained_head_path = os.path.join(ckpdir, 'trained_rigid_drift_head.pt')
    else:
        trained_head_path = os.path.join(ckpdir, 'trained_drift_head.pt')

    # The zero-drift initialization head is managed by get_classification_head().
    # train_drift() saves only the learned drift head to a separate checkpoint.
#    if os.path.exists(trained_head_path):
#        print(f'Skipping fine-tuning because {trained_head_path} exists.')
#        return trained_head_path

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        print("Loading image encoder.")
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    if rigid_movement is not False:
        classification_head = get_classification_head(args, train_dataset, drift=False)
    else:
        classification_head = get_classification_head(args, train_dataset, drift=True)

    model = ImageClassifier(image_encoder, classification_head)

    if rigid_movement is False:
        model.freeze_head()
    
    model.freeze_encoder()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)

    prev_val_acc = None
    best_val_acc = -1.0
    best_epoch = -1
    best_train_loss = None
    best_val_loss = None

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        train_loss_sum = 0.0
        train_batches = 0
        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)
            train_loss_sum += loss.item()
            train_batches += 1

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        train_loss_epoch = train_loss_sum / train_batches if train_batches > 0 else 0.0

        model.eval()
        val_loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')

                logits = model(inputs)
                val_loss = loss_fn(logits, labels)
                preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss_sum += val_loss.item()
                val_batches += 1

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss_epoch = val_loss_sum / val_batches if val_batches > 0 else 0.0

        print(f"Validation summary for epoch {epoch}:")
        print(f"Current validation accuracy: {100.0 * val_acc:.2f}%")
        print(f"Current validation loss: {val_loss_epoch:.6f}")
        print(f"Average training loss this epoch: {train_loss_epoch:.6f}")

        if prev_val_acc is None:
            print("Previous validation accuracy: N/A (first epoch)")
        else:
            print(f"Previous validation accuracy: {100.0 * prev_val_acc:.2f}%")
            if val_acc > prev_val_acc:
                print("Validation accuracy is going up compared to previous epoch.")
            elif val_acc < prev_val_acc:
                print("Validation accuracy went down compared to previous epoch.")
            else:
                print("Validation accuracy stayed the same as previous epoch.")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_loss = train_loss_epoch
            best_val_loss = val_loss_epoch
            if args.save is not None:
                model.module.classification_head.save(trained_head_path)
                print(f"Saved new best checkpoint to {trained_head_path}")
            print(f"Best validation accuracy so far: {100.0 * best_val_acc:.2f}% (new best)")
        else:
            print(f"Best validation accuracy so far: {100.0 * best_val_acc:.2f}%")

        prev_val_acc = val_acc

    if best_epoch >= 0:
        print("Best checkpoint summary:")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation accuracy: {100.0 * best_val_acc:.2f}%")
        print(f"Training loss at best epoch: {best_train_loss:.6f}")
        print(f"Validation loss at best epoch: {best_val_loss:.6f}")

    if args.save is not None:
        return trained_head_path


def train_1_layer_mlp(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)
    trained_adapter_path = os.path.join(ckpdir, "trained_1_layer_mlp_adapter.pt")

    assert train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        print("Loading image encoder.")
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    classification_head = get_classification_head(args, train_dataset, drift=False)
    embedding_dim = classification_head.weight.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    adapter = TaskAdapter(dim=embedding_dim, normalize_output=True)

    model = AdapterImageClassifier(image_encoder, adapter, classification_head)
    model.freeze_base()
    

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        create_zeroshot_model(args, train_dataset=train_dataset)

    prev_val_acc = None
    best_val_acc = -1.0
    best_epoch = -1
    best_train_loss = None
    best_val_loss = None

    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        model.module.image_encoder.eval()
        model.module.classification_head.eval()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        train_loss_sum = 0.0
        train_batches = 0
        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)
            train_loss_sum += loss.item()
            train_batches += 1

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        train_loss_epoch = train_loss_sum / train_batches if train_batches > 0 else 0.0

        model.eval()
        val_loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')

                logits = model(inputs)
                val_loss = loss_fn(logits, labels)
                preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss_sum += val_loss.item()
                val_batches += 1

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss_epoch = val_loss_sum / val_batches if val_batches > 0 else 0.0

        print(f"Validation summary for epoch {epoch}:")
        print(f"Current validation accuracy: {100.0 * val_acc:.2f}%")
        print(f"Current validation loss: {val_loss_epoch:.6f}")
        print(f"Average training loss this epoch: {train_loss_epoch:.6f}")

        if prev_val_acc is None:
            print("Previous validation accuracy: N/A (first epoch)")
        else:
            print(f"Previous validation accuracy: {100.0 * prev_val_acc:.2f}%")
            if val_acc > prev_val_acc:
                print("Validation accuracy is going up compared to previous epoch.")
            elif val_acc < prev_val_acc:
                print("Validation accuracy went down compared to previous epoch.")
            else:
                print("Validation accuracy stayed the same as previous epoch.")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_loss = train_loss_epoch
            best_val_loss = val_loss_epoch
            if args.save is not None:
                model.module.adapter.save(trained_adapter_path)
                print(f"Saved new best checkpoint to {trained_adapter_path}")
            print(f"Best validation accuracy so far: {100.0 * best_val_acc:.2f}% (new best)")
        else:
            print(f"Best validation accuracy so far: {100.0 * best_val_acc:.2f}%")

        prev_val_acc = val_acc

    if best_epoch >= 0:
        print("Best checkpoint summary:")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation accuracy: {100.0 * best_val_acc:.2f}%")
        print(f"Training loss at best epoch: {best_train_loss:.6f}")
        print(f"Validation loss at best epoch: {best_val_loss:.6f}")

    if args.save is not None:
        return trained_adapter_path
