# FBB-FL: Federated Learning with Uncertainty-Based Client Selection

This repository implements a **modular federated learning pipeline** using [Flower](https://flower.dev/) for training classical machine learning models (e.g., Decision Trees, Random Forest, Gradient Boosting, etc.) on **tabular data** and Deep Learning models (e.g. AlexNet, ResNet, GoogLeNet etc) on **Image data** . It also introduces a **custom server-side strategy** based on **Uncertainty Measures (UMs)** to intelligently select predictions based on maximum confidence at the client level.

---

## 🚀 Features

- 🔁 Federated learning using Flower with multiple scikit-learn and PyTorch models.
- 📊 Tabular dataset loader and Image dataset loader with stratified client partitions.
- 🔍 Client-side computation of Uncertainty Measures (UMs).
- 🧠 Server-side selection of predictions based on UM-max strategy.
- 📉 Evaluation of aggregated accuracy and omission metrics.
- 📁 Clean modular structure with utilities for training, evaluation, and data handling.

---

## 📂 Repository Structure

```
FBB-FL/
├── dataset/
│   └── Tabular/    # Tabular dataset
│   └── Image/      # Image dataset
├── federated_learning_tabular_pipeline.py  # Main training pipeline
├── image_custom.py                          # Code of Image Dataset
├── tabular_custom.py                        # Code of Tabular Dataset
├── models/
│   └── model_utils.py                        # Classifier factory
├── results.txt                           # Output metrics
├── confidence_report.csv                 # Per-model predictions and probabilities
├── um_output.csv                         # UM-based final predictions
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/fahadahmedkhokhar/FBB-FL.git
cd FBB-FL
```

### 2. Install dependencies

Create a virtual environment and install the required packages:

```bash
conda create -n fbb-fl python=3.10
conda activate fbb-fl

pip install -r requirements.txt
# OR manually install
pip install scikit-learn pandas numpy flwr
```

---

## 🧪 Running the Project

### Run the desired dataset base python file

```bash
python image_custom.py
```


This script computes:
- UM-based majority predictions
- Confidence scores
- Omission metrics (alpha, epsilon, phi, etc.)

---

## 📈 Example Output

```
✅ UM-Max Accuracy: 0.9432
✅ Prediction and Uncertainty Measure calculated and saved.
```

---

## 📊 Omission Metrics

After running the pipeline, omission metrics are saved to `results.txt`:

| Metric       | Description                              |
|--------------|------------------------------------------|
| alpha        | Accuracy of original classifier          |
| alpha_w      | Accuracy after applying wrapper (UM)     |
| phi          | Fraction of rejected samples (omissions) |
| eps_gain     | Accuracy gain from rejection strategy    |

---

## 🧠 Custom Strategy Details

The custom Flower strategy (`UMSelectionStrategy`) modifies the default `FedAvg` by:
- Skipping parameter averaging
- Collecting UMs and predictions from each client
- Selecting per-sample predictions from the client with the **maximum UM value**

---

## 📝 Author

**Fahad Ahmed Khokhar**  
[GitHub](https://github.com/fahadahmedkhokhar) | [LinkedIn](https://www.linkedin.com/in/fahadahmedkhokhar)

---

## 📜 License

This project is licensed under the MIT License.
