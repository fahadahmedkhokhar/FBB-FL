import flwr as fl
import sys
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from models.plmodels import ImageClassifier
from dataset_utils import load_datasets, get_dnn_classifiers


class TorchImageClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_partitions: int):
        self.cid = cid
        self.trainloader, self.valloader, self.testloader = load_datasets(cid, num_partitions)

        model_list = get_dnn_classifiers()
        self.model = model_list[cid % len(model_list)]
        self.fitted = False

    def get_parameters(self, config):
        # No parameters exchanged in classical models
        return []

    def fit(self, parameters, config):
        self.model.fit(self.trainloader, self.valloader)
        self.fitted = True
        return [], len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # Ensure the model is trained before evaluating
        if not self.fitted:
            self.model.fit(self.trainloader, self.valloader)
            self.fitted = True

        preds = self.model.predict(self.testloader)
        probas = self.model.predict_proba(self.testloader)

        metrics = {
            "predictions": json.dumps(preds.tolist()),
            "probabilities": json.dumps(probas.max(axis=1).tolist())
        }

        if self.cid == 0:
            y_true = []
            for _, labels in self.testloader:
                y_true.extend(labels.numpy())  # accumulate ground-truth labels
            metrics["true_labels"] = json.dumps(list(map(int, y_true)))



        print("Fahad", len(preds) )
        return float(accuracy_score(y_true, preds)), len(self.testloader.dataset), metrics


if __name__ == "__main__":
    cid = int(sys.argv[1])
    num_clients = int(sys.argv[2])
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=TorchImageClient(cid, num_clients))