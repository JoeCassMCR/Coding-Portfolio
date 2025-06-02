# Predictive Maintenance Pipeline

This project implements a complete predictive‐maintenance workflow on CMAPSS turbofan engine data. It covers:

- **Unsupervised anomaly detection**: Isolation Forest, LOF, One‐Class SVM  
- **Hyperparameter tuning** (Isolation Forest, Random Forest)  
- **Ensembling** of unsupervised and supervised detectors  
- **Supervised anomaly classification**: Random Forest on “true anomalies”  
- **Remaining Useful Life (RUL) regression**:  
  - Gradient Boosting Regressor (GBR)  
  - XGBoost  
  - Neural‐Net autoencoder + GBR  
- **Visualization** of ROC/PR curves, detection delays, and anomaly flags on individual units  

---

## Dataset

- **Source**: CMAPSS Turbofan Engine Degradation Simulation Data Set  
- **File**: `data/train_FD001.txt`  
  - Contains 21 sensor readings and 3 operating settings per time‐cycle for multiple engine units  
  - We label the **last 15 cycles** of each unit (RUL ≤ 15) as “true anomalies”

---

## Models Evaluated

| Model                          | Description                                                                                       |
|--------------------------------|---------------------------------------------------------------------------------------------------|
| **Isolation Forest**           | Tree‐ensemble unsupervised detector; uses `decision_function` to score anomalies.                 |
| **Local Outlier Factor (LOF)** | Density‐based unsupervised detector; uses `decision_function` (novelty=True) to score anomalies. |
| **One‐Class SVM**              | Kernel‐based SVM on “normal” data; `decision_function` inverted for anomaly scoring.             |
| **Isolation Forest _tuned_**   | Tuned version using grid‐search on n_estimators & contamination to maximize mean anomaly score.  |
| **Random Forest Classifier**   | Supervised classifier trained on “true_anomaly” labels (last 15 cycles).                          |
| **GBR (RUL)**                  | Gradient Boosting Regressor to predict Remaining Useful Life (RUL).                                |
| **XGBoost (RUL)**              | XGBoost Regressor to predict RUL.                                                                  |
| **NN Autoencoder + GBR (RUL)**  | Autoencoder bottleneck features fed into GBR to predict RUL.                                      |

---

## Results

### Unsupervised Anomaly Detection (Before/After Tuning)

| Model                | Precision (anomaly=1) | Recall (anomaly=1) | F1‐Score (anomaly=1) | Avg Detection Delay (cycles) |
|----------------------|-----------------------|--------------------|----------------------|------------------------------|
| Isolation Forest     | 0.6368                | 0.1644             | 0.2613               | 179.6                        |
| LOF                  | 0.1090                | 0.0281             | 0.0447               | 135.0                        |
| One‐Class SVM        | 0.1386                | 0.5450             | 0.2210               | 188.1                        |
| **IsolationForest _tuned_** | 0.5381         | 0.6538             | 0.5903               | —                            |
| **LOF _tuned_**             | 0.0776         | 1.0000             | 0.1440               | —                            |
| **OneClassSVM _tuned_**     | 0.1389         | 0.6012             | 0.2257               | —                            |

> *Tuned versions use thresholds chosen to maximize F1 on the Precision‐Recall curve.*

---

### Supervised Random Forest (Anomaly Classification)

- **Precision (anomaly=1):** 0.8885  
- **Recall (anomaly=1):** 0.8219  
- **F1‐Score (anomaly=1):** 0.8539  
- **Accuracy:** 0.9782  

---

### RUL Regression

| Model                      | RMSE (Test Split) |
|----------------------------|-------------------|
| Gradient Boosting (GBR)    | 39.80             |
| XGBoost                    | 38.26             |
| NN Autoencoder + GBR         | 44.34             |

---

## Graphs

All plots are saved under the `graphs/` directory at runtime. Key figures include:

1. **ROC Curves**  
   - File: `graphs/roc_curves.png`  
   - Compares ROC for Isolation Forest, LOF, One‐Class SVM (unsupervised).

2. **Precision–Recall Curves**  
   - File: `graphs/pr_curves.png`  
   - Compares PR curves and average precision for each unsupervised model.

3. **Detection Delay Bar Chart**  
   - File: `graphs/detection_delay.png`  
   - Shows average detection delay (cycles) per model.

4. **Anomaly Comparison on Unit 5**  
   - File: `graphs/unit_5_anomaly_comparison.png`  
   - Overlays `sensors_mean_all` time series with red/blue/green flags for each tuned model.

---

## Highlights

- **Feature Engineering**  
  - Rolling‐window mean & std (window = 5) per sensor  
  - 1‐cycle difference (lag), 2‐cycle slope, acceleration (second difference)  
  - Sensor_2 / Sensor_3 ratio, global mean & std of all rolling means  
  - Variance threshold (drop features with σ = 0.01) & correlation threshold (drop corr > 0.95)

- **Unsupervised → Supervised**  
  - Benchmarked IsolationForest, LOF, OneClassSVM on unlabeled data  
  - Tuned IsolationForest hyperparameters to improve F1  
  - Trained a supervised RandomForestClassifier on “true_anomaly” labels (last 15 cycles)

- **RUL Prediction**  
  - Compared three approaches (GBR, XGBoost, Autoencoder + GBR) on RUL regression  
  - Reported standalone RMSE for each

- **Logging & Reproducibility**  
  - All constants, thresholds, hyperparameter lists, paths are in `config.py`  
  - Pipeline logs to both console and `pipeline.log`  

---

## How to Run

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Place the Data**  
   Ensure `train_FD001.txt` is in the `data/` folder:  
   ```
   predictive_maintenance/
   ├── data/
   │   └── train_FD001.txt
   └── [other files…]
   ```

3. **Run the Full Pipeline**  
   ```bash
   python main.py
   ```  
   By default, it will:
   - Load & preprocess data  
   - Engineer features & label “true anomalies” (RUL ≤ 15)  
   - Train IsolationForest, LOF, OneClassSVM  
   - Evaluate raw & tuned detectors (classification, ROC, PR, delay)  
   - Produce anomaly‐comparison plot for Unit 5  
   - Tune IsolationForest & RandomForest (unless skipped)  
   - Ensemble tuned IsolationForest & RF predictions  
   - Train supervised RF classifier on “true_anomaly” labels  
   - Train RUL regressors (GBR, XGBoost, NN+GBR)  

   All plots appear under `graphs/`, and metrics/logs are written to `pipeline.log`.

4. **Optional Flags**  
   - **Skip hyperparameter tuning** (faster run):  
     ```bash
     python main.py --skip-tuning
     ```  
   - **Only evaluate & plot** (requires saved models):  
     ```bash
     python main.py --only-evaluate
     ```

---

## Next Steps / Improvements

- **Reduce Detection Delay (IsolationForest _tuned)**  
  - Further threshold tuning or more discriminative features (e.g., higher‐order slopes, additional sensor ratios).

- **Expanded Feature Engineering**  
  - Add higher‐order lags (e.g., 3, 4, 5‐cycle differences) or domain‐specific sensor combinations.

- **Cross‐Validated Hyperparameter Search**  
  - Perform k-fold grid‐search (rather than “mean score” proxy) for IsolationForest & RandomForest to improve precision/recall and RMSE.

- **Advanced Ensembles**  
  - Combine tuned IsolationForest, LOF, and supervised RF (e.g., raise anomaly only when ≥ 2 detectors agree) to reduce false positives.

- **Neural RUL Models**  
  - Try LSTM/GRU sequence modeling on raw sensor time series and compare RMSE against current regressors.

- **Separate Training Scripts for Each Component**  
  - Implement distinct `train_unsupervised.py`, `train_supervised.py`, and `train_rul.py` scripts to modularize and simplify usage.

---

Feel free to browse the code, run experiments, and adapt any component to your own predictive‐maintenance projects.
