import torch
import numpy as np
import os
import math
import yaml
import argparse
import collections
import time
import csv


from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from torch.amp import autocast, GradScaler

from deepfakes_dataset import DeepFakesDataset
from utils import check_correct, shuffle_dataset, get_n_params


BASE_DIR = '../deep_fakes_explain/'
DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")

MODELS_PATH = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_CSV = os.path.join(BASE_DIR, "training_results.csv")
EPOCH_1_SUMMARY = os.path.join(BASE_DIR, "epoch_1_summary.txt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--patience', type=int, default=5)

    opt = parser.parse_args()

    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    from convnext_crossvit import ConvNeXtCrossViT
    model = ConvNeXtCrossViT().to(device)

    print("Model Parameters:", get_n_params(model))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight-decay'])
    )

    # scheduler initialization moved after data loading to calculate total steps

    writer = SummaryWriter(LOG_DIR)

    scaler = GradScaler(enabled=True)

    start_epoch = 0
    best_val_acc = 0
    previous_val_acc = 0
    not_improved_loss = 0
    previous_loss = math.inf


    print("\nScanning dataset...\n")

    train_dataset = []
    validation_dataset = []

    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(TRAINING_DIR, folder)
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    train_dataset.append((os.path.join(root, f), label))

    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(VALIDATION_DIR, folder)
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    validation_dataset.append((os.path.join(root, f), label))

    train_dataset = shuffle_dataset(train_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)

    train_images = np.asarray([x[0] for x in train_dataset])
    train_labels = np.asarray([x[1] for x in train_dataset])

    val_images = np.asarray([x[0] for x in validation_dataset])
    val_labels = np.asarray([x[1] for x in validation_dataset])

    train_samples = len(train_images)
    validation_samples = len(val_images)

    print("Train images:", train_samples)
    print("Validation images:", validation_samples)

    print("\nCalculating class weights...\n")

    counter = collections.Counter(train_labels)

    print("__TRAINING STATS__")
    print(counter)

    if counter[1] == 0:
        class_weight = 1.0
    else:
        class_weight = counter[0] / counter[1]

    print("Calculated pos_weight (for logging):", class_weight)

    # Removed pos_weight here because WeightedRandomSampler is natively oversampling 
    # the rare class. Doing both causes the model to guess Real for 100% of images.
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_dataset = DeepFakesDataset(
        train_images,
        train_labels,
        config['model']['image-size']
    )

    # Balanced sampling
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = torch.from_numpy(class_weights[train_labels]).float()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    validation_dataset = DeepFakesDataset(
        val_images,
        val_labels,
        config['model']['image-size'],
        mode='validation'
    )

    dl = DataLoader(
        train_dataset,
        batch_size=int(config['training']['bs']),
        sampler=sampler,
        num_workers=opt.workers,
        pin_memory=True,
        persistent_workers=(opt.workers > 0),
        prefetch_factor=4 if opt.workers > 0 else None
    )

    val_dl = DataLoader(
        validation_dataset,
        batch_size=int(config['training']['bs']),
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
        persistent_workers=(opt.workers > 0),
        prefetch_factor=4 if opt.workers > 0 else None
    )

    accumulation_steps = config['training'].get('accumulation_steps', 2)

    # Calculate total steps for OneCycleLR
    total_steps = (len(dl) * opt.num_epochs) // accumulation_steps
    
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config['training']['lr']),
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )

    print("\nStarting training...\n")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Label smoothing: {config['training'].get('label_smoothing', 0.1)}")
    print(f"Effective batch size: {int(config['training']['bs']) * accumulation_steps}")

    auto_resume_path = os.path.join(MODELS_PATH, "last_checkpoint.pth")
    if opt.resume == '' and os.path.exists(auto_resume_path):
        print(f"Auto-resume found checkpoint at {auto_resume_path}")
        opt.resume = auto_resume_path

    if opt.resume != '':
        if os.path.exists(opt.resume):
            checkpoint = torch.load(opt.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_val_acc' in checkpoint:
                best_val_acc = checkpoint['best_val_acc']
                previous_val_acc = checkpoint.get('previous_val_acc', best_val_acc)
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        else:
            print(f"Resume path {opt.resume} does not exist.")

    # Initialize CSV header if file does not exist
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'precision', 'recall', 'f1', 'time_min'])

    for epoch in range(start_epoch, opt.num_epochs):

        epoch_start_time = time.time()

        if not_improved_loss == opt.patience:
            print("Early stopping triggered.")
            break

        model.train()

        total_loss = 0
        counter_batches = 0
        train_correct = 0
        train_seen = 0

        optimizer.zero_grad()

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{opt.num_epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):

            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

            # Manual Label Smoothing (for compatibility with older Torch versions)
            smoothing = config['training'].get('label_smoothing', 0.1)
            smoothed_labels = labels * (1 - smoothing) + 0.5 * smoothing

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(images)
                loss = loss_fn(preds, smoothed_labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            corrects, _, _ = check_correct(
                preds.detach().cpu(),
                labels.detach().cpu()
            )

            batch_size = images.size(0)

            train_correct += corrects
            train_seen += batch_size
            total_loss += loss.item() * accumulation_steps
            counter_batches += 1

            # Get current LR for logging
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=loss.item() * accumulation_steps, lr=f"{current_lr:.6f}")

        train_accuracy = train_correct / train_seen
        total_loss /= counter_batches

        model.eval()

        val_loss = 0
        val_correct = 0
        val_counter = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images, labels in tqdm(val_dl, desc="Validation", leave=False):

                images = images.to(device, non_blocking=True)
                labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = model(images)
                    loss = loss_fn(preds, labels)

                probs = torch.sigmoid(preds)
                pred_labels = (probs > 0.5).int()

                all_preds.extend(pred_labels.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

                val_correct += (pred_labels == labels).sum().item()

                val_loss += loss.item()
                val_counter += 1

        val_loss /= val_counter
        val_accuracy = val_correct / len(all_labels)

        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        # scheduler.step() moved to training loop

        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
        writer.add_scalar("F1", f1, epoch)

        epoch_time = time.time() - epoch_start_time
        remaining_epochs = opt.num_epochs - (epoch + 1)

        eta_minutes = (epoch_time * remaining_epochs) / 60
        epoch_minutes = epoch_time / 60

        print(
            f"\nEpoch {epoch+1}/{opt.num_epochs} "
            f"| epoch_time: {epoch_minutes:.2f} min "
            f"| ETA: {eta_minutes:.2f} min "
            f"| loss:{total_loss:.4f} "
            f"| acc:{train_accuracy:.4f} "
            f"| val_loss:{val_loss:.4f} "
            f"| val_acc:{val_accuracy:.4f} "
            f"| precision:{precision:.4f} "
            f"| recall:{recall:.4f} "
            f"| f1:{f1:.4f}"
        )

        if epoch > start_epoch and val_accuracy < previous_val_acc * 0.90:
            print(f"\nAccuracy dropped significantly ({previous_val_acc:.4f} -> {val_accuracy:.4f}). Stopping training to prevent divergence.")
            break

        os.makedirs(MODELS_PATH, exist_ok=True)
        
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'previous_val_acc': val_accuracy,
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict()
        }

        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy
            checkpoint_dict['best_val_acc'] = best_val_acc

            torch.save(
                checkpoint_dict,
                os.path.join(MODELS_PATH, "best_model.pth")
            )

        if val_loss >= previous_loss:
            not_improved_loss += 1
        else:
            not_improved_loss = 0

        previous_loss = val_loss
        previous_val_acc = val_accuracy

        torch.save(checkpoint_dict, os.path.join(MODELS_PATH, f"checkpoint_epoch_{epoch+1}.pth"))
        torch.save(checkpoint_dict, os.path.join(MODELS_PATH, "last_checkpoint.pth"))

        # Log results to CSV
        with open(RESULTS_CSV, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                epoch + 1,
                f"{total_loss:.4f}",
                f"{train_accuracy:.4f}",
                f"{val_loss:.4f}",
                f"{val_accuracy:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{epoch_minutes:.2f}"
            ])

        # Store epoch 1 results specifically
        if epoch + 1 == 1:
            with open(EPOCH_1_SUMMARY, 'w') as f:
                f.write("--- EPOCH 1 RESULTS SUMMARY ---\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuration: {opt.config}\n")
                f.write(f"Train Loss: {total_loss:.4f}\n")
                f.write(f"Train Accuracy: {train_accuracy:.4f}\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")
                f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Epoch Time: {epoch_minutes:.2f} min\n")
                f.write("-------------------------------\n")
            print(f"Epoch 1 results stored in {EPOCH_1_SUMMARY}")