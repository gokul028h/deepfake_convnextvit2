import torch
import numpy as np
import os
import glob
import math
import yaml
import argparse
import collections

from deepfakes_dataset import DeepFakesDataset
from progress.bar import ChargingBar
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import check_correct, shuffle_dataset, get_n_params


BASE_DIR = '../deep_fakes_explain/'
DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")

MODELS_PATH = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")


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

    from convnext_crossvit import ConvNeXtCrossViT

    model = ConvNeXtCrossViT().to(device)

    print("Model Parameters:", get_n_params(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight-decay']
    )

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['step-size'],
        gamma=config['training']['gamma']
    )

    writer = SummaryWriter(LOG_DIR)

    if opt.resume != '':
        checkpoint = torch.load(opt.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded.")

    print("\nScanning dataset...\n")

    train_dataset = []
    validation_dataset = []

    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(TRAINING_DIR, folder)

        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    train_dataset.append((os.path.join(root,f),label))

    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(VALIDATION_DIR, folder)

        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    validation_dataset.append((os.path.join(root,f),label))


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

    print("Weights:", class_weight)

    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([class_weight]).to(device)
    )

    train_dataset = DeepFakesDataset(
        train_images,
        train_labels,
        config['model']['image-size']
    )

    validation_dataset = DeepFakesDataset(
        val_images,
        val_labels,
        config['model']['image-size'],
        mode='validation'
    )

    dl = DataLoader(
        train_dataset,
        batch_size=config['training']['bs'],
        shuffle=True,
        num_workers=opt.workers
    )

    val_dl = DataLoader(
        validation_dataset,
        batch_size=config['training']['bs'],
        shuffle=False,
        num_workers=opt.workers
    )

    print("\nStarting training...\n")

    previous_loss = math.inf
    not_improved_loss = 0
    best_val_acc = 0

    for epoch in range(opt.num_epochs):

        if not_improved_loss == opt.patience:
            print("Early stopping triggered.")
            break

        model.train()

        total_loss = 0
        counter_batches = 0
        train_correct = 0

        bar = ChargingBar('EPOCH #' + str(epoch + 1), max=len(dl))

        for images, labels in dl:

            images = images.to(device)

            labels = labels.unsqueeze(1).float().to(device)

            preds = model(images)

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            corrects, pred_labels, true_labels = check_correct(
                preds.detach().cpu(),
                labels.detach().cpu()
            )

            train_correct += corrects
            total_loss += loss.item()
            counter_batches += 1

            bar.next()

        bar.finish()

        train_accuracy = train_correct / train_samples
        total_loss /= counter_batches

        model.eval()

        val_loss = 0
        val_correct = 0
        val_counter = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for images, labels in val_dl:

                images = images.to(device)
                labels = labels.unsqueeze(1).float().to(device)
                preds = model(images)
                loss = loss_fn(preds, labels)
                corrects, pred_labels, true_labels = check_correct(
                    preds.cpu(),
                    labels.cpu()
                )

                all_preds.extend(pred_labels)
                all_labels.extend(true_labels)

                val_correct += corrects
                val_loss += loss.item()
                val_counter += 1

        val_loss /= val_counter
        val_accuracy = val_correct / validation_samples

        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        scheduler.step()

        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
        writer.add_scalar("F1", f1, epoch)

        print(
            f"\nEpoch {epoch+1}/{opt.num_epochs} "
            f"loss:{total_loss:.4f} "
            f"acc:{train_accuracy:.4f} "
            f"val_loss:{val_loss:.4f} "
            f"val_acc:{val_accuracy:.4f} "
            f"precision:{precision:.4f} "
            f"recall:{recall:.4f} "
            f"f1:{f1:.4f}"
        )

        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy

            os.makedirs(MODELS_PATH, exist_ok=True)

            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                },
                os.path.join(MODELS_PATH, "best_model.pth")
            )

        if val_loss >= previous_loss:
            not_improved_loss += 1
        else:
            not_improved_loss = 0

        previous_loss = val_loss