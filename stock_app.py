import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Predictor")
st.info("🔁 If you get a rate limit error, please wait 1-2 minutes and click 'Rerun' (top-right).")

# Download and cache data
@st.cache_data
def load_stock_data():
    return yf.download('AAPL', start='2019-01-01', end='2024-12-31')

# Try downloading
try:
    df = load_stock_data()
    if df.empty:
        st.error("❌ No data received. Rate limit hit. Please try again later.")
        st.stop()
except Exception as e:
    st.error(f"❌ Download failed: {e}")
    st.stop()

# Prepare data
df = df[['Close']].dropna()
df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

X = df[['Prev_Close']]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.success(f"✅ Model trained! RMSE: {rmse:.2f}, R² Score: {r2:.2f}")

# Plot results
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.index, y_test, label='Actual', color='blue')
ax.plot(y_test.index, y_pred, label='Predicted', color='orange')
ax.set_title("AAPL: Actual vs Predicted Closing Prices")
ax.legend()
st.pyplot(fig)
