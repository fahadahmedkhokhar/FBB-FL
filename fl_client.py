import flwr as fl
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from dataset_utils import load_partition


class SklearnClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_partitions: int):
        self.cid = cid
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = load_partition(cid, num_partitions)

        # List of different models for each client
        model_list = [
            GradientBoostingClassifier(n_estimators=30),
            GradientBoostingClassifier(n_estimators=100),
            DecisionTreeClassifier(),
            LinearDiscriminantAnalysis(),
            RandomForestClassifier(n_estimators=100),
            GaussianNB()
        ]

        # Assign a model based on client ID
        self.model = model_list[cid % len(model_list)]
        self.fitted = False

    def get_parameters(self, config):
        # No parameters exchanged in classical models
        return []

    def fit(self, parameters, config):
        self.model.fit(self.X_train, self.y_train)
        self.fitted = True
        return [], len(self.X_train), {}

    def evaluate(self, parameters, config):
        # Ensure the model is trained before evaluating
        if not self.fitted:
            self.model.fit(self.X_train, self.y_train)
            self.fitted = True

        preds = self.model.predict(self.X_test)
        probas = self.model.predict_proba(self.X_test)

        metrics = {
            "predictions": json.dumps(preds.tolist()),
            "probabilities": json.dumps(probas.max(axis=1).tolist())
        }

        if self.cid == 0:
            metrics["true_labels"] = json.dumps(self.y_test.tolist())

        return float(accuracy_score(self.y_test, preds)), len(self.X_test), metrics


if __name__ == "__main__":
    cid = int(sys.argv[1])
    num_clients = int(sys.argv[2])
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=SklearnClient(cid, num_clients))
