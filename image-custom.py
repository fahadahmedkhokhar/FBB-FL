
from collections import OrderedDict, Counter
from typing import List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from models.plmodels import *
import sklearn
import csv
import os

NUM_CLASSES = 9
MAX_EPOCHS = 1
DATASET_PATH = "dataset/testing"
DEVICE = torch.device("cuda")
print(f"Training on {DEVICE}")

# Dataset loader
def load_datasets(partition_id: int, num_partitions: int):
    root_dir = DATASET_PATH+"/train"
    full_dataset = ImageFolder(root=root_dir)
    global NUM_CLASSES
    NUM_CLASSES = len(full_dataset.classes)
    partition_size = len(full_dataset) // num_partitions
    remainder = len(full_dataset) % num_partitions
    partition_lengths = [partition_size + 1 if i < remainder else partition_size for i in range(num_partitions)]
    partitions = random_split(full_dataset, partition_lengths, generator=torch.Generator().manual_seed(42))
    partition = partitions[partition_id]
    train_size = int(0.8 * len(partition))
    val_size = len(partition) - train_size
    partition_train_test = random_split(partition, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    pytorch_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class CustomTransformWrapper(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    trainloader = DataLoader(CustomTransformWrapper(partition_train_test[0], pytorch_transforms), batch_size=4, shuffle=True)
    valloader = DataLoader(CustomTransformWrapper(partition_train_test[1], pytorch_transforms), batch_size=4)
    test_dataset = ImageFolder(root=DATASET_PATH+"/test", transform=pytorch_transforms)
    testloader = DataLoader(test_dataset, batch_size=4)

    return trainloader, valloader, testloader

# Model and training functions
def get_dnn_classifiers():
    models = []
    model_name = ['VGG11','DenseNet121','GoogLeNet','ResNet50','ResNet101', 'AlexNet']
    for model in model_name:
        model = ImageClassifier(model, num_classes=NUM_CLASSES, learning_rate=1e-4, max_epochs=MAX_EPOCHS)
        models.append(model)
    return models

def extract_labels(dataloader):
    labels_list = []
    for batch in dataloader:
        y = batch[1]
        labels_list.extend(y.numpy())
    return np.unique(np.array(labels_list))

def build_confidence(dataloader, y_proba, y_pred) -> pd.DataFrame:
    label_tags = extract_labels(dataloader)
    x_test, y_test = [], []
    for batch in dataloader:
        x_batch = batch[0]
        y_batch = batch[1]
        y_test.extend(y_batch.numpy())
        x_test.extend(range(len(x_batch)))

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    out_df = pd.DataFrame()
    out_df['true_label'] = list(map(lambda x: label_tags[x], y_test))
    out_df['predicted_label'] = list(map(lambda x: label_tags[x], y_pred))
    out_df['is_misclassification'] = np.where(out_df['true_label'] != out_df['predicted_label'], 1, 0)
    out_df['probabilities'] = [np.array2string(y_proba[i], separator=";") for i in range(len(y_proba))]
    return out_df

def train_models_on_partitions(partition_id: int, num_partitions: int):
    classifiers = get_dnn_classifiers()
    base_labels = None
    all_predictions = {}
    for pid in range(num_partitions):
        print(f"\n--- Client {pid} using model {classifiers[pid].model_name} ---")
        trainloader, valloader, testloader = load_datasets(pid, num_partitions)
        model = classifiers[pid]
        model.fit(trainloader, valloader)
        y_pred = model.predict(testloader)
        y_proba = model.predict_proba(testloader)
        if base_labels is None:
            base_labels = build_confidence(testloader, y_proba, y_pred)[['true_label']]
        model_column_name = model.model_name + f"_Client_{pid}"
        model_proba_column_name = model.model_name + f"_Client_{pid}_proba"
        all_predictions[model_column_name] = y_pred
        all_predictions[model_proba_column_name] = np.max(y_proba, axis=1)
    final_df = pd.DataFrame(all_predictions)
    final_df = pd.concat([final_df, base_labels], axis=1)
    return final_df

# Step 2: Apply UM
def cal_um(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    model_prediction_cols = [col for col in df.columns if not col.endswith("_proba") and "_Client_" in col]
    model_probability_cols = [col for col in df.columns if col.endswith("_proba") and "_Client_" in col]

    def get_um_and_max_prob(row):
        predictions = [row[col] for col in model_prediction_cols]
        probabilities = [row[col] for col in model_probability_cols]
        pred_counter = Counter(predictions)
        majority_pred, _ = pred_counter.most_common(1)[0]
        max_prob = max(prob for pred, prob in zip(predictions, probabilities) if pred == majority_pred)
        return pd.Series({"final_prediction": majority_pred, "final_probability": max_prob})

    df[["final_prediction", "final_probability"]] = df.apply(get_um_and_max_prob, axis=1)
    df.to_csv(output_csv, index=False)
    print("✅ Prediction and Uncertainity Measure calculated and saved.")

# Step 3: Compute Omission Metrics
def compute_omission_metrics(y_true, y_wrapper, y_clf, reject_tag='omission'):
    met_dict = {}
    met_dict['alpha'] = sklearn.metrics.accuracy_score(y_true, y_clf)
    met_dict['eps'] = 1 - met_dict['alpha']
    met_dict['phi'] = np.count_nonzero(y_wrapper == reject_tag) / len(y_true)
    met_dict['alpha_w'] = sum(y_true == y_wrapper) / len(y_true)
    met_dict['eps_w'] = 1 - met_dict['alpha_w'] - met_dict['phi']
    met_dict['phi_c'] = sum(np.where((y_wrapper == reject_tag) & (y_clf == y_true), 1, 0)) / len(y_true)
    met_dict['phi_m'] = sum(np.where((y_wrapper == reject_tag) & (y_clf != y_true), 1, 0)) / len(y_true)
    met_dict['eps_gain'] = 0 if met_dict['eps'] == 0 else (met_dict['eps'] - met_dict['eps_w']) / met_dict['eps']
    met_dict['phi_m_ratio'] = 0 if met_dict['phi'] == 0 else met_dict['phi_m'] / met_dict['phi']
    return met_dict

def save_or_append_dict_plain(data, file_path):
    headers = data.keys()
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
            writer.writerows([data])
    else:
        with open(file_path, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
            writer.writeheader()
            writer.writerows([data])

if __name__ == "__main__":
    partition_id = 6
    num_partitions = 6
    df = train_models_on_partitions(partition_id, num_partitions)
    confidence_file = "confidence_report.csv"
    majority_file = "um_output.csv"
    df.to_csv(confidence_file, index=False)
    cal_um(confidence_file, majority_file)

    df_majority = pd.read_csv(majority_file)
    y_true = df_majority['true_label'].to_numpy()
    y_clf = df_majority['final_prediction'].to_numpy()
    final_probability = df_majority['final_probability'].to_numpy()
    df_majority['Predicted_FCC'] = np.where(final_probability >= 0.9, y_clf, 'omission')
    y_wrapper = df_majority['Predicted_FCC'].apply(lambda x: int(float(x)) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)

    metrics = compute_omission_metrics(y_true, y_wrapper, y_clf)
    save_or_append_dict_plain(metrics, "results.txt")
