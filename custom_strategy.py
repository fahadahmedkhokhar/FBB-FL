from flwr.server.strategy import FedAvg
from flwr.common import EvaluateRes
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import json
from collections import Counter
import sklearn

class MyCustomStrategy(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        print(f"🔄 Round {rnd}: skipping aggregation (non-NN models).")
        return [], {}

    def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[Any, EvaluateRes]],
            failures: List[Any],
    ) -> Optional[float]:
        print(f"📊 [Round {rnd}] Custom UM + Omission evaluation")

        if not results:
            print("⚠️ No results received. Skipping.")
            return super().aggregate_evaluate(rnd, results, failures)

        all_preds = []
        all_probas = []
        true_labels = None

        for i, (client_proxy, eval_res) in enumerate(results):
            metrics = eval_res.metrics
            print(f"Client {i} metrics: {list(metrics.keys())}")

            preds = json.loads(metrics["predictions"])
            probas = json.loads(metrics["probabilities"])
            all_preds.append([str(p) for p in preds])
            all_probas.append([float(p) for p in probas])

            if true_labels is None and "true_labels" in metrics:
                true_labels = [int(p) for p in json.loads(metrics["true_labels"])]

        if true_labels is None:
            print("❌ true_labels missing. UM + omission skipped.")
            return super().aggregate_evaluate(rnd, results, failures)

        # Build prediction matrix
        preds_matrix = np.array(all_preds).T
        probas_matrix = np.array(all_probas).T

        final_preds = []
        final_probs = []

        for i in range(len(true_labels)):
            row_preds = preds_matrix[i]
            row_probas = probas_matrix[i]
            counter = Counter(row_preds)
            majority_label, _ = counter.most_common(1)[0]
            filtered_probs = [
                float(prob) for pred, prob in zip(row_preds, row_probas)
                if str(pred) == str(majority_label)
            ]
            max_prob = max(filtered_probs) if filtered_probs else 0.0
            final_preds.append(majority_label)
            final_probs.append(max_prob)

        # Build final dataframe
        df = pd.DataFrame({
            "true_label": true_labels,
            "final_prediction": final_preds,
            "final_probability": final_probs
        })

        # === Apply omission threshold
        threshold = 0.9
        df["Predicted_FCC"] = np.where(df["final_probability"] >= threshold, df["final_prediction"], "omission")

        y_true = df["true_label"].to_numpy()
        y_clf = df["final_prediction"].astype(int).to_numpy()
        y_wrapper = df["Predicted_FCC"].apply(
            lambda x: int(float(x)) if isinstance(x, str) and x.replace(".", "", 1).isdigit() else x
        ).to_numpy()

        # === Compute omission metrics
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

        metrics = compute_omission_metrics(y_true, y_wrapper, y_clf)
        print(f"📈 Omission Metrics (Round {rnd}): {metrics}")

        # === Save results
        df.to_csv(f"um_round_{rnd}.csv", index=False)
        with open(f"omission_metrics_round_{rnd}.txt", "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}\t{v}\n")

        return super().aggregate_evaluate(rnd, results, failures)
