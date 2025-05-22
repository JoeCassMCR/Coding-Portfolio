# Stock-Prediction

This project forecasts Apple's (AAPL) stock price using GRU and LSTM neural networks. It includes both single-feature and multi-feature prediction models, comparing performance via MSE and R².

## Data Source
- Ticker: AAPL
- Range: 2020–2024
- Downloaded via `yfinance`

## Features Used
### Single Feature Model:
- Close Price

### Multi-Feature Model:
- MA20, MA50, EMA20
- Bollinger Bands (Upper & Lower)
- RSI, MACD, ATR
- Stochastic Oscillator
- Lagged Close

## Results

| Model        | MAPE     | R² Score |
|--------------|---------|----------|
| GRU (Single) | *0.02*  | *0.95*   |
| LSTM (Single)| *0.02*  | *0.95*   |
| GRU (Multi)  | *0.0139*  | *0.99793*   |
| LSTM (Multi) | *0.0156*  | *0.9739*   |

Visual comparisons in `assets/`.
### Graphs

#### Single Feature Graphs

#####Train and Validation Graph
![Train and Validation Graph](https://github.com/user-attachments/assets/d83ea1b2-4029-4188-b37a-4a5939305fcf)

##### Predictions
![Predictions](https://github.com/user-attachments/assets/528b3e00-8761-4d37-b8bc-8b32a2d5a602)

#### Multiple Feature Graphs

#####Train and Validation Graph
![Train and Loss Validation](https://github.com/user-attachments/assets/3d1f83e2-b054-4185-a392-cece4472d116)

##### Predictions
![Predictions](https://github.com/user-attachments/assets/0101f18e-325b-4d69-9bac-867570f651d9)


## How to Run

```bash
pip install -r ../requirements.txt
python Stocks_GRU_LSTM_SingleFeature.py
python Stocks_GRU_LSTM_MultiFeature.py
```
