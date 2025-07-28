import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="📈 Stock Price Predictor")

st.title("📊 Stock Price Predictor App")
st.write("Predicts future stock prices using a Random Forest Regressor.")

@st.cache_data
def load_data():
    df = yf.download('AAPL', start='2019-01-01', end='2024-12-31')
    df = df[['Close']].dropna()
    df['Prev_Close'] = df['Close'].shift(1)
    return df.dropna()

df = load_data()

st.write("### Sample Data")
st.dataframe(df.tail(10))

X = df[['Prev_Close']]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.success(f"✅ Model trained! RMSE: {rmse:.2f}, R² Score: {r2:.2f}")

# Plot
st.write("### 📈 Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
ax.plot(y_test.index, y_pred, label='Predicted', linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.set_title("AAPL: Actual vs Predicted Closing Prices")
ax.legend()
st.pyplot(fig)
