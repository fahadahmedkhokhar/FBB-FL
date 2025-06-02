import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
