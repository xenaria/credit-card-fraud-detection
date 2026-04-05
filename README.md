
# Credit Card Fraud Detection (Machine Learning Project)
Credit Card Fraud Detection project for SUTD 50.039 Theory and Practice of Deep Learning.

Using [Credit Card Fraud  Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) from Kaggle.

This project explores multiple machine learning approaches for detecting fraudulent credit card transactions on an imbalanced dataset. The goal is to compare traditional models, deep learning, anomaly detection, and state-of-the-art techniques to identify the most effective solution.



## Project Overview

Fraud detection is a highly imbalanced classification problem where fraudulent transactions are extremely rare. This project investigates different strategies to handle class imbalance and improve detection performance.

We implement and compare:

- Baseline Neural Network
- Weighted Neural Network (Cost-Sensitive Learning)
- Autoencoder (Anomaly Detection)
- XGBoost (State-of-the-Art for Tabular Data)



## Dataset

- Credit Card Fraud Detection Dataset
- Highly imbalanced:
  - Normal transactions ≫ Fraud transactions
- Features are anonymized (PCA-transformed)



## Methods

### 1. Baseline Neural Network
- Standard binary classifier
- Uses BCE loss without weighting
- Serves as a reference model



### 2. Weighted Neural Network
- Uses `BCEWithLogitsLoss(pos_weight=...)`
- Handles imbalance by penalizing fraud misclassification
- Includes:
  - Early stopping
  - Learning rate scheduler
  - Threshold tuning


### 3. Autoencoder (Anomaly Detection)
- Trained only on normal transactions
- Detects fraud via reconstruction error
- Threshold-based classification



### 4. XGBoost (SOTA)
- Gradient boosting model optimized for tabular data
- Uses `scale_pos_weight` for imbalance handling
- No manual training loop required
- Strong performance in real-world fraud detection systems


## Evaluation Metrics

Due to class imbalance, multiple metrics are used:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC (most important)
- Balanced Accuracy
- Confusion Matrix (normalized)



## Key Results

| Model | Precision | Recall | F1 Score | PR-AUC |
|------|----------|--------|----------|--------|
| Weighted NN | 0.7941 | **0.8265** | 0.8100 | 0.7387 |
| XGBoost | **0.9390** | 0.7857 | **0.8556** | **0.8697** |



### Insights

- **XGBoost achieved the best overall performance**, with the highest F1 score and PR-AUC.
- It significantly reduced false positives while maintaining strong fraud detection capability.
- The weighted neural network achieved higher recall but at the cost of more false alarms.
- Autoencoder performed well for anomaly detection but lacked precision compared to supervised models.
- GNN showed limited effectiveness due to lack of relational data.



## Model Trade-offs

| Model | Strength | Weakness |
|------|--------|---------|
| Weighted NN | High recall | More false positives |
| XGBoost | High precision & F1 | Slightly lower recall |
| Autoencoder | Detects anomalies | Less accurate classification |


## Key Takeaway

XGBoost is the most suitable model for this task due to its strong performance on tabular data, ability to handle imbalance, and superior precision-recall balance.



## Tech Stack

- Python 3.10
- PyTorch
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn



##  Project Structure
```
├── data/ # Dataset (ignored in Git)
├── notebooks/ # Jupyter notebooks
│ ├── baseline.ipynb
│ ├── weighted_model.ipynb
│ ├── autoencoder.ipynb
│ ├── xgboost.ipynb
│ └── gnn.ipynb
├── models/ # Saved models
├── utils/ # Helper functions
├── README.md
└── requirements.txt
```

## How to Run

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run notebooks in order



## Authors

|             Names            |         | 
|-----------------------------| ------- |
| Pagdilao Geoffrey Cyd Caraig    | |
| Lindero Dianthe Marithe Lumagui | |
| Karen Neo                       | |



## References

- https://kumo.ai/resources/learn/fraud-detection-xgboost-vs-gnn/
- https://arxiv.org/abs/2306.12251