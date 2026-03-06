import os
import sys
import cv2
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import csv

from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, f1_score
from tqdm import tqdm
from progress.bar import Bar
from multiprocessing.pool import Pool
from multiprocessing import Manager
from functools import partial
from shutil import copyfile

import ttach as tta

# allow importing model files
sys.path.append(os.path.abspath("."))

from evit_model10 import EfficientViT
from transforms.albu import IsotropicResize
from utils import get_method, custom_round, custom_video_round

from albumentations import Compose, PadIfNeeded

RESULTS_DIR = "results"
BASE_DIR = "../deep_fakes_explain"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "validation_set")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "tests")

TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])


def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    fpr, tpr, _ = metrics.roc_curve(correct_labels, preds)
    model_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{model_name} (AUC={model_auc:.3f})")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')

    plt.savefig(os.path.join(
        OUTPUT_DIR,
        f"{model_name}_acc{accuracy*100:.2f}_loss{loss:.4f}_f1{f1:.4f}.jpg"
    ))
    plt.clf()


def read_frames(video_path, videos):

    method = get_method(video_path, DATA_DIR)

    if "Original" in video_path:
        label = 0.
    else:
        label = 1.

    selected_frames = []

    frames = os.listdir(video_path)
    frames_number = len(frames)

    if frames_number == 0:
        return

    frames_interval = max(1, int(frames_number / opt.frames_per_video))

    frames = frames[::frames_interval][:opt.frames_per_video]

    video = []

    for frame in frames:

        img_path = os.path.join(video_path, frame)
        image = cv2.imread(img_path)

        if image is None:
            continue

        transform = create_base_transform(config['model']['image-size'])
        image = transform(image=image)['image']

        video.append(image)

    if len(video) > 0:
        videos.append((video, label, video_path, frames))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset', type=str, default='All')
    parser.add_argument('--frames_per_video', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--config', type=str)

    opt = parser.parse_args()

    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if not os.path.exists(opt.model_path):
        print("ERROR: Model checkpoint not found.")
        exit()

    channels = 1280

    model = EfficientViT(config=config, channels=channels, selected_efficient_net=0)

    checkpoint = torch.load(opt.model_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Test Time Augmentation wrapper
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 90]),
    ])

    model = tta.ClassificationTTAWrapper(model, tta_transforms)

    print("Model loaded.")

    mgr = Manager()
    videos = mgr.list()
    paths = []

    if opt.dataset == 'All':
        folders = ["Original", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "Deepfakes"]
    else:
        folders = [opt.dataset, "Original"]

    for folder in folders:

        method_folder = os.path.join(TEST_DIR, folder)

        if not os.path.exists(method_folder):
            continue

        for video_folder in os.listdir(method_folder):
            paths.append(os.path.join(method_folder, video_folder))

    print("Total videos:", len(paths))

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(paths)) as pbar:
            for _ in p.imap_unordered(partial(read_frames, videos=videos), paths):
                pbar.update()

    videos = list(videos)

    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    frames = [row[0] for row in videos]

    preds = []

    bar = Bar('Predicting', max=len(frames))

    for index, video in enumerate(frames):

        faces_preds = []

        for i in range(0, len(video), opt.batch_size):

            batch = video[i:i + opt.batch_size]

            batch = torch.tensor(np.asarray(batch))
            batch = batch.permute(0, 3, 1, 2).float().to(device)

            with torch.no_grad():

                pred = model(batch)

                pred = torch.sigmoid(pred)

            faces_preds.extend(pred.cpu().numpy())

        video_pred = np.mean(faces_preds)

        preds.append(video_pred)

        bar.next()

    bar.finish()

    preds = np.asarray(preds)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    tensor_labels = torch.tensor(correct_test_labels).unsqueeze(1).float()
    tensor_preds = torch.tensor(preds).unsqueeze(1)

    loss = loss_fn(tensor_preds, tensor_labels).item()

    accuracy = accuracy_score(custom_round(preds), correct_test_labels)
    f1 = f1_score(correct_test_labels, custom_round(preds))

    print("Accuracy:", accuracy)
    print("Loss:", loss)
    print("F1:", f1)

    save_roc_curves(correct_test_labels, preds, "EfficientViT", accuracy, loss, f1)