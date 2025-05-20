import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout


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

# Reshaping for the GRU neural network
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_train.shape, X_test.shape)

# GRU Model
GRUModel = Sequential()
GRUModel.add(GRU(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
GRUModel.add(Dropout(0.2))
GRUModel.add(GRU(units=64, return_sequences=False))
GRUModel.add(Dropout(0.2))
GRUModel.add(Dense(units=1))

GRUModel.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training the GRU Model
history = GRUModel.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

# Predictions
y_pred = GRUModel.predict(X_test)

# Inverse scaling predictions and actual values
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Display results
print("\nFirst 5 Predictions vs Actuals:")
for i in range(5):
    print(f"Predicted: {y_pred_rescaled[i][0]:.2f}, Actual: {y_test_rescaled[i][0]:.2f}")

# Evaluate performance
GRUmse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
GRUr2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"\nGRU Mean Squared Error: {GRUmse:.2f}")
print(f"GRU R2 Score: {GRUr2:.2f}")

# LSTM Model
LSTMModel = Sequential()
LSTMModel.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
LSTMModel.add(Dropout(0.2))
LSTMModel.add(LSTM(units=64, return_sequences=False))
LSTMModel.add(Dropout(0.2))
LSTMModel.add(Dense(units=1))

LSTMModel.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training the GRU Model
history = LSTMModel.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

# Predictions
y_pred = LSTMModel.predict(X_test)

# Inverse scaling predictions and actual values
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Display results
print("\nFirst 5 Predictions vs Actuals:")
for i in range(5):
    print(f"Predicted: {y_pred_rescaled[i][0]:.2f}, Actual: {y_test_rescaled[i][0]:.2f}")

# Evaluate performance
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"\nLSTM Mean Squared Error: {mse:.2f}")
print(f"LSTM R2 Score: {r2:.2f}")


print(f"\nGRU Mean Squared Error: {GRUmse:.2f}")
print(f"GRU R2 Score: {GRUr2:.2f}")




# Plot training and validation loss for GRU
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss - GRU')
plt.plot(history.history['val_loss'], label='Validation Loss - GRU')
plt.title('GRU Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Training the LSTM Model again to capture its history
history_lstm = LSTMModel.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

# Plot training and validation loss for LSTM
plt.figure(figsize=(12, 6))
plt.plot(history_lstm.history['loss'], label='Train Loss - LSTM')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss - LSTM')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predictions vs. actual values for GRU
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:100], label='Actual')
plt.plot(y_pred_rescaled[:100], label='Predicted - GRU')
plt.title('GRU Predictions vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Re-predict with the LSTM model to plot predictions vs actuals
y_pred_lstm = LSTMModel.predict(X_test)
y_pred_rescaled_lstm = scaler.inverse_transform(y_pred_lstm)

# Plot predictions vs. actual values for LSTM
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:100], label='Actual')
plt.plot(y_pred_rescaled_lstm[:100], label='Predicted - LSTM')
plt.title('LSTM Predictions vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Compare errors between GRU and LSTM
models = ['GRU', 'LSTM']
mse_values = [GRUmse, mean_squared_error(y_test_rescaled, y_pred_rescaled_lstm)]
r2_values = [GRUr2, r2_score(y_test_rescaled, y_pred_rescaled_lstm)]

plt.figure(figsize=(12, 6))
plt.bar(models, mse_values, color=['blue', 'orange'])
plt.title('Mean Squared Error Comparison')
plt.ylabel('Mean Squared Error')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(models, r2_values, color=['blue', 'orange'])
plt.title('R2 Score Comparison')
plt.ylabel('R2 Score')
plt.show()

