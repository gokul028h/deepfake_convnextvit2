Explainable Deepfake Detection using ConvNeXt + Vision Transformer

A hybrid ConvNeXt–Vision Transformer deepfake detector with attention-based explainability for analyzing manipulated face images extracted from videos.

This project implements a deepfake detection framework that:

Uses ConvNeXt + Vision Transformer architecture for feature extraction and classification

Supports training, evaluation, and inference

Generates explainable attention heatmaps highlighting manipulated regions

Works with multiple datasets including FaceForensics++ deepfake methods

The system provides a pipeline for:

Video preprocessing

Face extraction

Model training

Model evaluation

Attention-based explainability

Project Overview

Deepfake detection is a critical task in combating misinformation and media manipulation.
This project combines modern convolutional networks (ConvNeXt) with transformer attention mechanisms to detect facial manipulations and provide interpretable explanations of predictions.

The explainability module produces relevancy heatmaps derived from transformer attention layers, helping visualize which facial regions influence the model’s decisions.

Repository Structure
deepfake_convnextvit
│
├── deep_fakes_explain
│   ├── dataset
│   │   ├── training_set
│   │   ├── validation_set
│   │   └── test_set
│   │
│   ├── FaceForensics
│   └── models
│
├── preprocessing
│   ├── detect_faces.py
│   ├── extract_crops.py
│   └── face_detector.py
│
├── model_test_train
│   ├── train_model.py
│   ├── test_model.py
│   ├── convnext_crossvit.py
│   ├── deepfakes_dataset.py
│   └── configs
│
├── explain_model
│   ├── explain_model.py
│   ├── examples
│   └── explanation
│
├── environment.yml
└── README.md
Installation
1. Clone the repository
git clone https://github.com/gokul028h/deepfake_convnextvit.git
cd deepfake_convnextvit
2. Create the Python environment

Using Conda:

conda env create -f environment.yml
conda activate deepfake_env

or install manually using pip if needed.

Dataset Setup

The repository does not include datasets due to size limitations.

Download the datasets separately and place them inside:

deep_fakes_explain/dataset/

Required structure:

dataset
├── training_set
│   ├── Deepfakes
│   └── Original
│
├── validation_set
│   ├── Deepfakes
│   └── Original
│
└── test_set
    ├── Deepfakes
    └── Original

Each dataset folder should contain video folders with extracted face frames:

Deepfakes
 ├ video_001
 │  ├ 0_0.png
 │  ├ 1_0.png
 │  └ ...
 ├ video_002
 │  ├ 0_0.png
 │  └ ...
Data Preprocessing

To run deepfake detection, faces must first be extracted from the videos.

Detect Faces
cd preprocessing

python detect_faces.py \
--data_path ../deep_fakes_explain/dataset \
--dataset FACEFORENSICS

Detected bounding boxes are saved to:

deep_fakes_explain/dataset/boxes/
Extract Face Crops
python extract_crops.py \
--data_path ../deep_fakes_explain/dataset \
--output_path ../deep_fakes_explain/dataset/training_set \
--dataset FACEFORENSIC

Repeat extraction for:

training_set
validation_set
test_set
Training the Model

Navigate to the training module:

cd model_test_train

Run training:

python train_model.py \
--config configs/explained_architecture.yaml \
--dataset Deepfakes
Optional Parameters
--num_epochs      Number of epochs (default: 100)
--workers         Data loader workers (default: 16)
--resume          Resume from checkpoint
--dataset         Dataset to train on
--max_videos      Limit number of videos
--patience        Early stopping patience

Trained models are saved to:

deep_fakes_explain/models/
Testing the Model

To evaluate the trained model:

cd model_test_train

python test_model.py \
--model_path ../deep_fakes_explain/models/convnext_crossvit_checkpoint \
--config configs/explained_architecture.yaml

Outputs include:

Accuracy

AUC score

F1 score

ROC curve

Prediction summaries

Results are saved in:

model_test_train/results/tests
Explainability (Attention Visualization)

The explainability module generates heatmaps showing important facial regions used for classification.

Navigate to:

cd explain_model

Place input images inside:

explain_model/examples

Run explanation:

python explain_model.py

Output visualizations will appear in:

explain_model/explanation
Hardware Requirements

Minimum (for testing):

CPU

16GB RAM

Recommended (for training):

NVIDIA GPU (RTX / Tesla)

16+ CPU cores

32GB+ RAM

Large-scale training may require:

2 GPUs

100GB RAM

Model Architecture

The model combines:

ConvNeXt

Used for hierarchical convolutional feature extraction.

Vision Transformer

Processes patch embeddings with multi-head attention.

Cross-Attention Fusion

Combines convolutional features with transformer representations.

Attention Explainability

Relevancy maps are computed from transformer attention weights.

Features

Hybrid ConvNeXt + ViT architecture

Attention-based explainability

Multi-dataset support

Training and evaluation pipelines

GPU acceleration

Deepfake method generalization

Future Improvements

Temporal deepfake detection

Multi-modal detection (audio + video)

Improved explainability maps

Lightweight model deployment

Benchmarking across more datasets

Credits

This implementation builds upon research in:

Vision Transformers

ConvNeXt architecture

Deepfake detection frameworks

Transformer explainability techniques

Author

Gokul

GitHub
https://github.com/gokul028h

License

This project is provided for research and educational purposes.
