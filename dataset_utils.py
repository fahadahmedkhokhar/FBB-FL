import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from models.plmodels import ImageClassifier

DATASET_PATH = "dataset/testing"
MAX_EPOCHS = 1
NUM_CLASSES = 9  # Set dynamically later if needed
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_datasets(partition_id: int, num_partitions: int):
    root_dir = DATASET_PATH + "/train"
    full_dataset = ImageFolder(root=root_dir)

    # Calculate NUM_CLASSES dynamically
    global NUM_CLASSES
    NUM_CLASSES = len(full_dataset.classes)

    # Split into partitions
    partition_size = len(full_dataset) // num_partitions
    remainder = len(full_dataset) % num_partitions
    partition_lengths = [partition_size + 1 if i < remainder else partition_size for i in range(num_partitions)]
    partitions = random_split(full_dataset, partition_lengths, generator=torch.Generator().manual_seed(42))
    partition = partitions[partition_id]

    # Further split into train/val
    train_size = int(0.8 * len(partition))
    val_size = len(partition) - train_size
    train_subset, val_subset = random_split(partition, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Custom wrapper to apply transform after split
    class TransformWrapper(Dataset):
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

    trainloader = DataLoader(TransformWrapper(train_subset, transform), batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(TransformWrapper(val_subset, transform), batch_size=BATCH_SIZE)
    test_dataset = ImageFolder(root=DATASET_PATH + "/test", transform=transform)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader


def get_dnn_classifiers():
    model_names = ['VGG11', 'DenseNet121', 'GoogLeNet', 'ResNet50', 'ResNet101', 'AlexNet']
    classifiers = []
    for name in model_names:
        model = ImageClassifier(model_name=name, num_classes=NUM_CLASSES, learning_rate=1e-4, max_epochs=MAX_EPOCHS)
        classifiers.append(model)
    return classifiers

def load_partition(partition_id: int, num_partitions: int):
    csv_path = "dataset/Metro/MetroPT2_shuffled.csv"
    label_column = "multilabel"
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df[label_column] = le.fit_transform(df[label_column])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_fraction = 0.2
    test_size = int(len(df) * test_fraction)
    test_df = df.iloc[:test_size]
    train_val_df = df.iloc[test_size:]

    partition_size = len(train_val_df) // num_partitions
    partitions = [train_val_df.iloc[i * partition_size: (i + 1) * partition_size] for i in range(num_partitions)]
    client_df = partitions[partition_id]

    train_df, val_df = train_test_split(client_df, test_size=0.2, random_state=42)

    def split_features_labels(df):
        X = df.drop(columns=[label_column])
        y = df[label_column]
        return X.values, y.values

    return split_features_labels(train_df), split_features_labels(val_df), split_features_labels(test_df)
