# federated_learning_tabular_pipeline.py

from collections import OrderedDict, Counter
from typing import List
import pandas as pd
import numpy as np
import torch
import flwr
import sklearn
import csv
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


DEVICE = torch.device("cuda")
print(f"Training on {DEVICE}")

# Dataset loader for tabular data
def load_datasets(partition_id: int, num_partitions: int):
    csv_path = "dataset/Error Detection/arancino_all_scikit.csv"
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
        return X, y
    X_train, y_train = split_features_labels(train_df)
    X_val, y_val = split_features_labels(val_df)
    X_test, y_test = split_features_labels(test_df)
    return X_train, y_train, X_val, y_val, X_test, y_test

# Classifier models
def get_dnn_classifiers():
    models = [
        GradientBoostingClassifier(n_estimators=30),
        GradientBoostingClassifier(n_estimators=100),
        DecisionTreeClassifier(),
        LinearDiscriminantAnalysis(),
        RandomForestClassifier(n_estimators=100),
        GaussianNB(),
    ]
    return models

def build_confidence(y_true, y_proba, y_pred):
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    out_df = pd.DataFrame()
    out_df["true_label"] = y_true
    out_df["predicted_label"] = y_pred
    out_df["is_misclassification"] = (y_true != y_pred).astype(int)
    out_df["probabilities"] = [np.array2string(probs, separator=";") for probs in y_proba]
    return out_df

def train_models_on_partitions(partition_id: int, num_partitions: int):
    classifiers = get_dnn_classifiers()
    base_labels = None
    all_predictions = {}
    for pid in range(num_partitions):
        print(f"\n--- Client {pid} using model {classifiers[pid].__class__.__name__} ---")
        X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(pid, num_partitions)
        model = classifiers[pid]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        if base_labels is None:
            base_labels = build_confidence(y_test, y_proba, y_pred)[['true_label']]
        model_column_name = model.__class__.__name__ + f"_Client_{pid}"
        model_proba_column_name = model.__class__.__name__ + f"_Client_{pid}_proba"
        all_predictions[model_column_name] = y_pred
        all_predictions[model_proba_column_name] = np.max(y_proba, axis=1)
    final_df = pd.DataFrame(all_predictions)
    final_df = pd.concat([final_df, base_labels], axis=1)
    return final_df

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
    print("✅ Prediction and Uncertainty Measure calculated and saved.")

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
