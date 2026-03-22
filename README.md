# Forex Exchange Rate Prediction using Deep Learning

Forecasting the EUR/DZD (Euro to Algerian Dinar) exchange rate using multiple Deep Learning architectures.

## Models
- Simple RNN
- LSTM
- GRU
- Transformer

## Dataset
- **Name:** Forex Exchange Rates Since 2004 (Updated Daily)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/asaniczka/forex-exchange-rate-since-2004-updated-daily)
- **Target currency:** DZD (Algerian Dinar) vs EUR

## Pipeline
- Extensive EDA (ACF/PACF, entropy, Kaboudan metric, Mann-Kendall, ADF test)
- First-order differencing + Min-Max scaling [-1, 1]
- Sliding window (input: 7 days → output: 1 day)
- Live validation against real market data (Dec 24, 2025)

## Key Results
- LSTM & GRU achieved lowest error (MAPE ~0.23%)
- Transformer showed better qualitative performance on volatility peaks

## How to Run
- Open on Kaggle — dataset loads automatically via `kagglehub`
