# NSE Equity Volatility & Valuation Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Architecture](https://img.shields.io/badge/Architecture-Ensemble%20Regression-orange)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)
## ⚠️CAUTION: My statement: "I built a stochastic modeling pipeline to experiment with feature engineering and volatility prediction."
##  System Abstract
This repository houses a **quantitative predictive modeling framework** designed to forecast short-term price volatility for NSE-listed equities (specifically *Zen Technologies*). 

Unlike traditional price-target models, this system minimizes variance by training on **relative percentage returns** rather than absolute price values. It leverages an **Ensemble Learning architecture (Random Forest Regressor)** to detect non-linear dependencies between momentum indicators (RSI, SMA, EMA) and future price action, outputting a probabilistic **BUY/SELL/HOLD** signal based on a volatility confidence threshold.

##  Key Technical Features
*   **Stochastic Data Pipeline:** Automated ingestion of OHLCV market data via `yfinance` APIs, supporting both historical backtesting (2000–Present) and real-time inference.
*   **Vectorized Feature Engineering:** Utilizes `pandas` and the `ta` library to synthesize technical vectors:
    *   **SMA (5-day):** Trend smoothing for short-term signal detection.
    *   **EMA (10-day):** Weighted moving average to prioritize recent price action.
    *   **RSI (14-day):** Momentum oscillator to identify overbought/oversold conditions.
*   **Ensemble Regression:** Implements `RandomForestRegressor` (100 estimators) to mitigate overfitting common in single decision trees when applied to noisy financial time-series data.
*   **Risk-Adjusted Decision Logic:** Generates signals only when predicted volatility exceeds a specific confidence threshold (>1%), effectively filtering out market noise.

##  Methodology & Mathematical Logic
The system frames the market prediction problem as a supervised regression task, isolating price velocity from absolute value.

### 1. Target Variable Normalization
To ensure the model is robust against historical price inflation and stock splits, the target variable $y$ is defined as the relative percentage change ($R_t$):

$$ y_t = \frac{P_{t+1} - P_t}{P_t} $$

Where $P_t$ is the closing price at time $t$.

### 2. Feature Vector Space
The input matrix $X$ is constructed from a rolling window of technical indicators, capturing both trend direction and momentum magnitude:

$$ X_t = \{ \text{SMA}_5(P), \text{EMA}_{10}(P), \text{RSI}_{14}(P) \} $$

### 3. Optimization Function
The Random Forest regressor optimizes for minimal **Mean Squared Error (MSE)** during training to reduce the variance of the predicted return $\hat{y}$:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

##  Repository Structure

| File | Description |
| :--- | :--- |
| `zentec_stock_data.csv` | **Data Layer:** Raw OHLCV dataset fetched from NSE (2014–2025). |
| `train_zentec_percent.py` | **Training Pipeline:** Preprocessing, feature extraction, train-test splitting (80/20), and model serialization. |
| `predict_zentec_percent.py` | **Inference Engine:** Fetches live trailing 60-day data, regenerates features, and computes next-day directional probability. |
| `anscom_zentec_model_percent.pkl` | **Serialized Model:** The optimized Random Forest model artifact. |

##  Usage & Installation

### 1. Prerequisites
Ensure the quantitative stack is installed:
```bash
pip install pandas yfinance ta scikit-learn joblib
2. Data Ingestion (Optional)
To refresh the historical dataset:
code
Python
# Run the extraction script
import yfinance as yf
data = yf.download("ZENTEC.NS", start="2000-01-01", end="2025-07-15")
data.to_csv("zentec_stock_data.csv")
3. Model Training
Execute the training pipeline to generate the .pkl artifact:
code
Bash
python train_zentec_percent.py
Output: Reports MSE (Mean Squared Error) and saves the model.
4. Real-Time Inference
Run the prediction engine to generate today's signal:
code
Bash
python predict_zentec_percent.py
   Sample Output (Inference)
code
Text
📆 Date: 2025-07-15
📉 Current Price: ₹1837.80
📈 Predicted % Change: 2.41%
🔮 Predicted Next Price: ₹1882.09
🟢 Signal: BUY
