# PV Power Production Forecasting — ML, Deep Learning & Transfer Learning

Forecasting photovoltaic power production across multiple horizons and methods
on real operational data. M.Eng. thesis, Renewable Energy Systems. The work is
structured in three complementary tracks.

## Track A — Day-Ahead (24h) Forecasting with a Physics-Informed Digital Twin
- **Approach:** A physics-informed digital twin combined with an LSTM, using
  Single Diode Model (SDM) features and monthly calibration, for 24-hour-ahead
  forecasting driven by numerical weather prediction (NWP).
- **Results:** Monthly-calibrated annual validation R² ≈ 0.72 (RMSE 106.5 kW,
  MAE 82.7 kW); the calibrated digital twin reproduced the annual energy yield
  closely.
- **Key insight:** Day-ahead accuracy was limited by NWP (weather-forecast)
  failures — on stable days R² ≈ 0.35, while the full 29-day aggregate was
  dominated by 8 NWP-failure days, identifying weather-forecast quality as the
  main bottleneck for day-ahead PV forecasting.

## Track B — One-Hour-Ahead Forecasting
- **Approach:** Benchmarked classical ML and deep-learning models on German
  electricity-market data (SMARD), via an automated API pipeline.
- **Best models:** R² ≈ 0.95 on the daytime test set.
- **Sensitivity analysis:** deep-learning models stayed robust under input
  noise; classical ML degraded more sharply.

## Track C — Transfer Learning (plants with limited history)
- **Problem:** New PV plants have little operating history, so models trained on
  their limited data alone perform poorly.
- **Approach:** Pre-trained models on a national-scale dataset (SMARD), then
  fine-tuned on a local site (Munich) with limited data. Compared scratch
  training, zero-shot transfer and fine-tuning across four architectures
  (LSTM, CNN, TCN, GRU).
- **Key results:**
  - Transfer learning (pre-train → fine-tune) beat training from scratch on
    3 of 4 architectures; CNN improved most (R² 0.11 → 0.56), as scratch CNN
    failed to converge on the small local dataset.
  - Fine-tuning consistently outperformed zero-shot (mean +0.104 R²) —
    local adaptation is always beneficial.
  - Conclusion: "train once nationally, deploy everywhere" is insufficient;
    site-level fine-tuning is essential for reasonable accuracy.

## Data
- SMARD (German electricity-market data) via the SMARD API, plus operational PV data
- Period 2019–2024 (Track B: train 2019–2022, test 2024; daytime hours)

## Pipeline
Automated data retrieval → cleaning & validation → feature engineering →
model training & benchmarking → evaluation (R², RMSE, MAE)

## Models
Classical ML: Linear/Ridge Regression, Random Forest, XGBoost, LightGBM, HistGradientBoosting
Deep Learning: LSTM, Bi-LSTM, Stacked LSTM, GRU, CNN, CNN-LSTM, TCN, Transformer

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · PyTorch · pvlib · Plotly
