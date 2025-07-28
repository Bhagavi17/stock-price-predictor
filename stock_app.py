import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf

# Download data for example stock
df = yf.download('AAPL', start='2019-01-01', end='2024-12-31')
df = df[['Close']].dropna()
df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

X = df[['Prev_Close']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.title("AAPL: Actual vs Predicted Closing Prices")
plt.savefig("stock_prediction.png")
