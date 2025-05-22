import yfinance as yf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dropout, Dense,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from typing import Union



def ensure_series(x: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


# 1) Download & flatten
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-12-31", auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 2) Feature Engineering
data['MA20']  = ensure_series(data['Close'].rolling(20).mean())
data['MA50']  = ensure_series(data['Close'].rolling(50).mean())
data['EMA20'] = ensure_series(data['Close'].ewm(span=20, adjust=False).mean())

std20 = ensure_series(data['Close'].rolling(20).std())
data['BB_Upper'] = data['MA20'] + 2 * std20
data['BB_Lower'] = data['MA20'] - 2 * std20

data['Lagged_Close'] = data['Close'].shift(1)

hl = data['High'] - data['Low']
hc = (data['High'] - data['Close'].shift()).abs()
lc = (data['Low'] - data['Close'].shift()).abs()
tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
data['ATR'] = ensure_series(tr.rolling(14).mean())

high14 = ensure_series(data['High'].rolling(14).max())
low14  = ensure_series(data['Low'].rolling(14).min())
stoch  = (data['Close'] - low14) / (high14 - low14) * 100
data['Stochastic_Oscillator'] = ensure_series(stoch)

# RSI
delta = data['Close'].diff()
gain  = ensure_series(delta.where(delta > 0, 0).rolling(14).mean())
loss  = ensure_series((-delta.where(delta < 0, 0)).rolling(14).mean())
rs    = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12   = ensure_series(data['Close'].ewm(span=12, adjust=False).mean())
ema26   = ensure_series(data['Close'].ewm(span=26, adjust=False).mean())
macd    = ema12 - ema26
signal  = ensure_series(macd.ewm(span=9, adjust=False).mean())
data['MACD']        = macd
data['Signal_Line'] = signal

# 3) Clean
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# 4) Scale
features = [
    'Close','MA20','MA50','EMA20','BB_Upper','BB_Lower',
    'Lagged_Close','ATR','Stochastic_Oscillator','RSI','MACD','Signal_Line'
]
sc_close    = MinMaxScaler()
sc_features = MinMaxScaler()

scaled_close = sc_close.fit_transform(data[['Close']])
scaled_other = sc_features.fit_transform(data[features[1:]])
# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
reduced = pca.fit_transform(scaled_other)
print(f"PCA reduced features from {scaled_other.shape[1]} to {reduced.shape[1]} dimensions.")

# Use reduced PCA features + scaled_close
scaled_data  = np.hstack([scaled_close, reduced])  # shape: (n_samples, 1 + n_pca_components)


# 5) Supervised dataset
def create_dataset(arr: np.ndarray, time_step: int = 7):
    X, y = [], []
    for i in range(len(arr) - time_step):
        X.append(arr[i : i + time_step])
        y.append(arr[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, time_step=7)
split = int(len(X) * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# 6) Define two models: GRU and LSTM
def make_model(layer):
    m = Sequential([
        Input(shape=(X_tr.shape[1], X_tr.shape[2])),
        Bidirectional(layer(128, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(layer(64)),
        Dropout(0.2),
        Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

gru_model  = make_model(GRU)
lstm_model = make_model(LSTM)

# 7) Train
epochs, batch = 100, 16
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

h_gru  = gru_model.fit(X_tr, y_tr, validation_data=(X_te,y_te),
                       epochs=epochs, batch_size=batch, verbose=2,
                       callbacks=[early_stop])
h_lstm = lstm_model.fit(X_tr, y_tr, validation_data=(X_te,y_te),
                        epochs=epochs, batch_size=batch, verbose=2,
                        callbacks=[early_stop])

# 8) Predict & rescale
def predict_rescale(m):
    y_s = m.predict(X_te)
    return sc_close.inverse_transform(y_s)

y_gru   = predict_rescale(gru_model)
y_lstm  = predict_rescale(lstm_model)
y_actual= sc_close.inverse_transform(y_te.reshape(-1,1))

# 9) Combined Train & Val Loss for GRU vs LSTM
plt.figure(figsize=(12, 6))
plt.plot(h_gru.history['loss'],     linestyle='-',  label='GRU Train Loss')
plt.plot(h_gru.history['val_loss'], linestyle='--', label='GRU Val Loss')
plt.plot(h_lstm.history['loss'],     linestyle='-',  label='LSTM Train Loss')
plt.plot(h_lstm.history['val_loss'], linestyle='--', label='LSTM Val Loss')
plt.title('Train & Validation Loss: GRU vs LSTM with Multiple Features')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig("loss_comparison_gru_lstm_Features.png")
plt.close()

# 10) Combined Predictions vs Actual (first 100 days)
plt.figure(figsize=(12, 6))
plt.plot(y_actual[:100],    linestyle='-',  label='Actual')
plt.plot(y_gru[:100],       linestyle='--', label='GRU Predicted')
plt.plot(y_lstm[:100],      linestyle='-.', label='LSTM Predicted')
plt.title('First 100 Days: Actual vs GRU & LSTM Predictions with Multiple Features')
plt.xlabel('Time Step')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig("predictions_vs_actual_gru_lstm_Features.png")
plt.close()

# 11) Metrics
for name, pred in [("GRU", y_gru), ("LSTM", y_lstm)]:
    mse = mean_squared_error(y_actual, pred)
    r2  = r2_score(y_actual, pred)
    print(f"{name} → MSE: {mse:.4f},  R²: {r2:.4f}")
