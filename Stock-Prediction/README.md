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

| Model        | MSE     | R² Score |
|--------------|---------|----------|
| GRU (Single) | *0.XX*  | *0.XX*   |
| LSTM (Single)| *0.XX*  | *0.XX*   |
| GRU (Multi)  | *0.XX*  | *0.XX*   |
| LSTM (Multi) | *0.XX*  | *0.XX*   |

Visual comparisons in `assets/`.

## How to Run

```bash
pip install -r ../requirements.txt
python Stocks_GRU_LSTM_SingleFeature.py
python Stocks_GRU_LSTM_MultiFeature.py
```