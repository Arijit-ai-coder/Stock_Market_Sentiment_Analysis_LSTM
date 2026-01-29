print("Program started successfully")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import yfinance as yf
import os

# =============================
# CREATE FOLDERS
# =============================
os.makedirs("model", exist_ok=True)

# =============================
# BRAND SELECTION
# =============================
BRANDS = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA"
}

print("\nAvailable Brands:")
for i, brand in enumerate(BRANDS.keys(), 1):
    print(f"{i}. {brand}")

choice = int(input("\nSelect brand number: "))
brand_name = list(BRANDS.keys())[choice - 1]
ticker = BRANDS[brand_name]

print(f"\nSelected Brand: {brand_name} ({ticker})")

# =============================
# STOCK DATA
# =============================
stock = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Fix MultiIndex issue
if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.get_level_values(0)

stock.reset_index(inplace=True)
stock = stock[['Date', 'Close']]

# Normalize stock dates
stock['Date'] = pd.to_datetime(stock['Date']).dt.date

# =============================
# NEWS DATA
# =============================
news_data = pd.read_csv("data/news_data.csv")

news_data['Date'] = pd.to_datetime(news_data['Date']).dt.date

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

news_data['sentiment'] = news_data['Headline'].apply(get_sentiment)

# Aggregate sentiment per day
news_data = news_data.groupby('Date', as_index=False)['sentiment'].mean()

# =============================
# MERGE DATA (CORRECT WAY)
# =============================
data = pd.merge(stock, news_data, on='Date', how='left')

# Fill missing sentiment safely
data['sentiment'] = data['sentiment'].fillna(0)

data = data[['Close', 'sentiment']]
data.dropna(inplace=True)

print("Merged data shape:", data.shape)

# =============================
# SCALING
# =============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =============================
# CREATE LSTM DATA
# =============================
X, y = [], []
time_steps = 60

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i - time_steps:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
print("Total samples:", len(X))

# =============================
# TRAIN TEST SPLIT
# =============================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =============================
# LSTM MODEL
# =============================
model = Sequential()
model.add(LSTM(50, return_sequences=True,
               input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# =============================
# PREDICTION
# =============================
predicted = model.predict(X_test).flatten()

# =============================
# EVALUATION METRICS (SAFE)
# =============================
mask = ~np.isnan(y_test) & ~np.isnan(predicted)

rmse = np.sqrt(mean_squared_error(y_test[mask], predicted[mask]))
mae = mean_absolute_error(y_test[mask], predicted[mask])

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# =============================
# VISUALIZATION
# =============================
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Price")
plt.plot(predicted, label="Predicted Price")
plt.title(f"{brand_name} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

plt.savefig(f"model/{ticker}_prediction.png")
plt.show()

# =============================
# SAVE MODEL
# =============================
model.save(f"model/{ticker}_lstm_model.h5")
print("Model and prediction graph saved successfully")
