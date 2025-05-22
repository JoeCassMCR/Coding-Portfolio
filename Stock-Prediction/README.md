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

![Train and Loss Validation](https://github.com/user-attachments/assets/2d42cb8f-abe3-453e-8890-534ed89a3877)
![Predictions](https://github.com/user-attachments/assets/bc4830df-090b-40bd-9be6-30768bcf52f2)

#### Multiple Feature Graphs
![Train and Loss Validation](https://github.com/user-attachments/assets/c3d97404-c37d-42c8-89a7-ea7f04fad426)
![Predictions](https://github.com/user-attachments/assets/98f1c1f1-e9ed-4f35-8322-4fe3af6119ba)


## How to Run

```bash
pip install -r ../requirements.txt
python Stocks_GRU_LSTM_SingleFeature.py
python Stocks_GRU_LSTM_MultiFeature.py
```
