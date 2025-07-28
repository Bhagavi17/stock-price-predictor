import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf
import numpy as np

# App Title
st.title("📈 Stock Price Predictor - AAPL")

# Download data
st.write("⏳ Downloading AAPL stock data...")
df = yf.download('AAPL', start='2019-01-01', end='2024-12-31')
df = df[['Close']].dropna()
df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

# Features and target
X = df[['Prev_Close']]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write("**RMSE:**", round(rmse, 2))
st.write("**R² Score:**", round(r2, 2))

# Plot
st.subheader("📉 Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.index, y_test, label='Actual')
ax.plot(y_test.index, y_pred, label='Predicted')
ax.legend()
ax.set_title("AAPL: Actual vs Predicted Closing Prices")
st.pyplot(fig)

