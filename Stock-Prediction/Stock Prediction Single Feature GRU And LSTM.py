import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

base_directory = os.path.abspath(".")  
assets_path = os.path.join(base_directory, "Assets")
os.makedirs(assets_path, exist_ok=True)

# Downloading the data
ticker = "aapl"
data = yf.download(ticker, start="2020-01-01", end="2024-12-31")
print(data.head())
print(data.isnull().sum())

# Scaling the 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
print(data_scaled[:5])

# Function to create datasets
def create_dataset(data, time_step=7):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled)
print(X.shape, y.shape)

# Splitting data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshaping for RNNs
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)
print(X_train.shape, X_test.shape)


# --- Build & Train GRU ---
GRUModel = Sequential([
    GRU(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
GRUModel.compile(optimizer='adam', loss='mean_squared_error')
history_gru = GRUModel.fit(
    X_train, y_train,
    epochs=15, batch_size=16,
    validation_data=(X_test, y_test)
)

# GRU Predictions & Metrics
y_pred_gru = GRUModel.predict(X_test)
y_pred_rescaled_gru = scaler.inverse_transform(y_pred_gru)
y_test_rescaled      = scaler.inverse_transform(y_test.reshape(-1,1))

GRUmape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled_gru)
GRUr2  = r2_score(y_test_rescaled, y_pred_rescaled_gru)
print(f"\nGRU MAPE: {GRUmape:.2f}, R2: {GRUr2:.2f}")


# --- Build & Train LSTM ---
LSTMModel = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
LSTMModel.compile(optimizer='adam', loss='mean_squared_error')
history_lstm = LSTMModel.fit(
    X_train, y_train,
    epochs=15, batch_size=16,
    validation_data=(X_test, y_test)
)

# LSTM Predictions & Metrics
y_pred_lstm = LSTMModel.predict(X_test)
y_pred_rescaled_lstm = scaler.inverse_transform(y_pred_lstm)

LSTMmape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled_lstm)
LSTMr2  = r2_score(y_test_rescaled, y_pred_rescaled_lstm)
print(f"\nLSTM MAPE: {LSTMmape:.2f}, R2: {LSTMr2:.2f}")


# --- 1) Combined Train & Validation Loss (GRU vs LSTM) ---
plt.figure(figsize=(12, 6))
plt.plot(history_gru.history['loss'],      '-',  label='GRU Train Loss')
plt.plot(history_gru.history['val_loss'],  '--', label='GRU Val Loss')
plt.plot(history_lstm.history['loss'],     '-',  label='LSTM Train Loss')
plt.plot(history_lstm.history['val_loss'], '--', label='LSTM Val Loss')
plt.title('Train & Validation Loss: GRU vs LSTM')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(assets_path, "loss_comparison_gru_lstm_Feature.png"))
plt.close()


# --- 2) Combined Predictions vs Actual (first 100 timesteps) ---
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:100],                 '-',  label='Actual')
plt.plot(y_pred_rescaled_gru[:100],             '--', label='GRU Predicted')
plt.plot(y_pred_rescaled_lstm[:100],            '-.', label='LSTM Predicted')
plt.title('First 100 Days: Actual vs GRU & LSTM Predictions')
plt.xlabel('Time Steps'); plt.ylabel('Stock Price')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(assets_path, "predictions_vs_actual_gru_lstm_Features.png"))
plt.close()
