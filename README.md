## Task 2: Stock Price Predictor

- Model: Random Forest
- Data Source: Yahoo Finance (AAPL)

To run:
```bash
pip install pandas matplotlib scikit-learn yfinance
python stock_predictor.py
```
# 📈 Stock Price Predictor

This Streamlit app predicts future closing prices for AAPL stock using a Random Forest Regressor.

## Features
- Downloads AAPL data from Yahoo Finance
- Preprocesses data with previous-day close
- Splits into train/test sets (80/20)
- Trains a Random Forest model
- Evaluates with RMSE and R²
- Visualizes Actual vs Predicted prices

## Requirements
Install all dependencies with:
