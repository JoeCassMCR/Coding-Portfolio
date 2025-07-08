# Credit Card Fraud Detection

This project detects fraudulent credit card transactions using a calibrated XGBoost model, with support from SMOTE-based oversampling, cost-sensitive threshold tuning, and ensemble learning.

## Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains 284,807 transactions with anonymized features (`V1` to `V28`) and a binary fraud label.

## Models & Methods
| Component           | Description                                     |
|---------------------|-------------------------------------------------|
| XGBoost             | Tuned via GridSearch with ROC-AUC scoring       |
| RandomForest        | Used in ensemble, with SMOTE                    |
| Calibration         | Isotonic regression on validation set           |
| Threshold Tuning    | Based on custom cost matrix (FN ≫ FP)           |
| Samplers Compared   | SMOTE, SMOTEENN, ADASYN                         |
| Ensemble            | Soft-voting average of Calibrated XGB + RF      |

## Results

| Model            | ROC-AUC | PR-AUC | Precision | Recall | F₁ Score |
|------------------|---------|--------|-----------|--------|----------|
| XGB (Calibrated) | 0.9706  | 0.7759 | 0.6752    | 0.8061 | 0.7349   |
| Ensemble (XGB+RF)| 0.9827  | 0.8535 | 0.7757    | 0.8469 | 0.8098   |

Threshold used: `0.1333` (from cost-based tuning)

## Features Used
- **Original**: `V1`–`V28`, `Amount`, `Time`
- **Engineered**: 
  - `log_amount = log1p(Amount)`
  - `hour_of_day = (Time // 3600) % 24`

## Highlights
- Full ML pipeline with schema validation (`pandera`)
- Robust handling of imbalanced data (SMOTE/ADASYN)
- Cost-sensitive classification using threshold tuning
- Calibrated probability predictions
- Test-time ensemble boosting performance
- FastAPI endpoint (`serve.py`) for real-time fraud prediction

## How to Run

```bash
pip install -r requirements.txt
python train.py --cost-fp 1 --cost-fn 10
```

Or tune just the threshold:

```bash
python threshold_tuner.py --model-path models/fraud_xgb_model.joblib --cost-fp 1 --cost-fn 10
```

Serve the model via API:

```bash
uvicorn serve:app --reload
```

## Input Format (`/predict`)
```json
{
  "V": [v1, v2, ..., v28],
  "log_amount": 6.21,
  "hour_of_day": 14
}
```

## Output Format
```json
{
  "probability": 0.732,
  "is_fraud": true
}
```

## Requirements
```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
joblib
fastapi
uvicorn
pandera
pytest
```

## Optional Enhancements
- Add SHAP for explainability
- Auto-retraining pipeline with drift detection
- Docker deployment (`Dockerfile` included)
